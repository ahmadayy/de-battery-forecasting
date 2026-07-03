# de-battery-forecasting

**How much economic value does better forecasting unlock for battery arbitrage?**

This repository answers that question for the German day-ahead power market:

1. **Phase 1:** build a 15-node PyPSA-Eur dispatch model of Germany and
   validate it against real ENTSO-E 2023 data.
2. **Phases 2-3:** build a leakage-safe forecasting dataset from real ENTSO-E
   DE-LU market data (2018-2026) and train price forecasters (persistence and
   LightGBM baselines, a point LSTM, a quantile LSTM), with a single locked
   test evaluation.
3. **Phase 4:** feed the forecasts into a battery dispatch optimizer and
   measure what forecast accuracy is worth in EUR/MW/yr.

> **Status: Phases 1-4 complete. Headline result:** switching from a naive
> persistence forecast to the quantile-LSTM median is worth
> **+5,012 EUR/MW/yr** for a 1 MW / 2 MWh battery trading German day-ahead
> prices. The model captures 89.9% of the 80,916 EUR/MW/yr perfect-foresight
> ceiling; the naive forecast captures 83.7%.
> Source: `results/phase4/arbitrage_summary.json`.

The forecasting layer (Phases 2-4) runs on real DE-LU market data, not on
model prices. The Phase 1 dispatch model is the validated base for a deferred
nodal extension (see future work in `docs/ARCHITECTURE.md`).

## Architecture in one table

Two codebases, two environment managers, no shared Python interpreter. They
exchange data only through files on disk (`.nc`, `.csv`, `.parquet`).

| Concern          | Location     | Env manager | Notes                                         |
| ---------------- | ------------ | ----------- | --------------------------------------------- |
| Dispatch model   | `pypsa-eur/` | pixi        | Upstream PyPSA-Eur clone (gitignored, pinned) |
| Forecasting / ML | `ml/`        | mamba       | Our code, conda env `energy-ml`               |

Design rationale and per-phase decisions:
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Repository structure

```
config/      PyPSA-Eur config override (config.germany-15node.yaml)
scripts/     shell drivers (phase1_solve.sh: plan / cutout / solve)
analysis/    scripts that run in the pixi env:
             inspect_fleet.py, validate_dispatch.py, battery_arbitrage.py
ml/          forecasting pipeline, conda env energy-ml:
             pull_entsoe.py, build_features.py, train_baselines.py,
             train_lstm.py, train_lstm_quantile.py, evaluate_test.py,
             export_test_forecasts.py
pypsa-eur/   upstream PyPSA-Eur clone (gitignored, pinned commit)
data/        raw and processed data (gitignored)
docs/        ARCHITECTURE.md, phase2_leakage_audit.md
results/germany-15node/validation/   Phase 1 validation outputs
results/phase3/                      forecasting metrics and loss curves
results/phase4/                      arbitrage results and sample-week plot
notebooks/   scratch space (empty so far)
```

## Reproduce

### Phase 1: dispatch model

Prerequisites: about 24 GB RAM (the binding constraint), 50+ GB free disk, a
Gurobi license at `~/gurobi.lic`, and a gitignored `.env` at the repo root:

```
ENTSOE_API_TOKEN=<your token>
```

Clone PyPSA-Eur at the pinned commit (it is not tracked in this repo):

```bash
git clone https://github.com/PyPSA/pypsa-eur.git pypsa-eur
cd pypsa-eur && git checkout 9423f94cb87daaad811aec14492f633b2b86541d
# == v2026.02.0-76-g9423f94c
pixi install
cd ..
```

Build and solve. The driver never runs anything without an explicit mode. The
full pipeline downloads tens of GB and takes a while, so run it yourself:

```bash
scripts/phase1_solve.sh plan     # dry-run: list jobs, download nothing
scripts/phase1_solve.sh cutout   # download only the ~6.6 GB weather cutout
scripts/phase1_solve.sh solve    # full pipeline: retrieve, build, solve
```

Output network: `pypsa-eur/results/germany-15node/networks/base_s_15_elec_.nc`.
Memory and concurrency can be tuned, for example
`JOBS=8 MEM_MB=20000 scripts/phase1_solve.sh solve`.

Validate against ENTSO-E 2023:

```bash
cd pypsa-eur
pixi run python ../analysis/validate_dispatch.py
```

This loads the real solved network and real ENTSO-E data, compares prices and
the generation mix, and writes CSVs and a plot to
`results/germany-15node/validation/`. It fails loudly if either input is
missing. It never fabricates or substitutes placeholder values.

### Phase 2: data pipeline

```bash
mamba env create -f ml/environment.yml
mamba activate energy-ml
python ml/pull_entsoe.py --start 2018-10-01 --end 2026-07-01
python ml/build_features.py
```

`pull_entsoe.py` fetches three ENTSO-E DE-LU series (day-ahead price, load
forecast, wind/solar forecast) onto an hourly UTC grid, prints a coverage and
gap report, and never interpolates. `build_features.py` writes
`data/processed/dataset.parquet`, a manifest of explicitly dropped rows, and
`docs/phase2_leakage_audit.md`.

### Phase 3: forecasting models

```bash
python ml/train_baselines.py       # persistence + LightGBM, writes baselines.json
python ml/train_lstm.py            # point LSTM (MSE)
python ml/train_lstm_quantile.py   # quantile LSTM (pinball loss, 10/50/90)
python ml/evaluate_test.py         # final one-time test evaluation. Run once.
```

All model and hyperparameter decisions are made on the validation split. The
test split is evaluated exactly once, by the last script, after everything is
locked.

### Phase 4: battery arbitrage

```bash
python ml/export_test_forecasts.py                 # energy-ml env
cd pypsa-eur
pixi run python ../analysis/battery_arbitrage.py   # pixi env (gurobipy)
```

The export is inference only, from the frozen Phase 3 checkpoint, and asserts
that it reproduces the committed Phase 3 test metrics before writing anything.
The optimizer solves 1,020 small daily MILPs and aborts on any non-optimal
day.

## Results

### Phase 1: dispatch model vs ENTSO-E 2023

15-node Germany, electricity only, hourly, full year 2023. Prices in EUR/MWh:

| Metric | Model  | ENTSO-E |
| ------ | -----: | ------: |
| mean   |  81.88 |   95.18 |
| std    |  37.87 |   47.58 |
| min    |   0.02 | -500.00 |
| max    | 117.95 |  524.27 |

Hourly Pearson correlation between model and ENTSO-E prices is about 0.75.

Generation mix, full-year 2023 (TWh):

| Carrier   |  Model | ENTSO-E |   Diff |
| --------- | -----: | ------: | -----: |
| Nuclear   |  20.93 |    6.74 | +14.19 |
| Lignite   |  76.79 |   77.84 |  -1.05 |
| Hard Coal |  25.17 |   39.75 | -14.58 |
| Gas       |   2.91 |   55.94 | -53.03 |
| Oil       |   0.00 |    3.15 |  -3.15 |
| Biomass   |  59.45 |   37.47 | +21.98 |
| Hydro     |  28.32 |   25.27 |  +3.05 |
| Wind      | 180.79 |  143.01 | +37.78 |
| Solar     |  77.55 |   55.80 | +21.75 |
| Other     |   3.09 |    9.72 |  -6.63 |

The gaps are structural and documented, not tuned away. See Known limitations.

### Phase 2: dataset

- `data/processed/dataset.parquet`: 66,548 hourly rows, 25 columns,
  2018-10-08 21:00 to 2026-06-29 21:00 UTC. 22 features, the target
  `da_price`, plus `forecast_origin` and `split` columns.
- Features: 4 price lags (24/48/72/168 h), 4 rolling 7-day price statistics
  (window ends 24 h before delivery), 9 calendar fields (local
  Europe/Brussels time, nationwide German holidays, cyclical encodings), the
  day-ahead load forecast, the wind/solar forecast at D-1 vintage (3 fields),
  and residual load.
- Every feature is available before the 12:00 CET day-ahead auction on D-1.
  Publication timing was verified against Commission Regulation (EU)
  No 543/2013: the load forecast is due two hours before gate closure (used
  directly), the wind/solar forecast is only due at 18:00 on D-1 (after the
  gate), so the previous day's vintage is used instead.
- Two real data bugs were found and fixed during the raw pull:
  1. EPEX moved day-ahead settlement to 15-minute periods during 2025, so the
     price series mixes hourly and 15-minute resolution. A single global
     resolution check missed this. Aggregation is now per hour bin and prints
     the detected transition (2025-09-30 22:15 UTC).
  2. entsoe-py's `@year_limited` request splitting drops the hour 24 h before
     the end of each roughly one-year block (verified against the live API:
     a one-year query drops the hour, a three-month query returns it). The
     pull now uses sub-year chunks with overlap and deduplication.
  After both fixes, all three series are gap-free: 0 missing hours over
  2018-2026.
- 773 rows were dropped explicitly because ENTSO-E never published the
  underlying forecast values (by year: 2018: 720, 2022: 48, 2023: 2, 2024: 2,
  2025: 1). A targeted re-pull confirmed none are recoverable. Manifest:
  `data/processed/_dropped_nan_rows.csv`. Nothing is interpolated anywhere in
  the pipeline.
- Chronological split: train 53,855 rows (2018-10 to 2024-12), val 4,152
  (2025-01 to 2025-06), test 8,541 (2025-07 to 2026-06), with an 8-day embargo
  at each boundary. The embargo must cover the 191-hour maximum feature
  lookback; a 7-day embargo would be 23 hours short.

### Phase 3: forecast accuracy (EUR/MWh)

| Model                         | Val MAE | Val RMSE | Test MAE | Test RMSE |
| ----------------------------- | ------: | -------: | -------: | --------: |
| Persistence (price 24 h ago)  |   26.36 |    40.52 |    27.33 |     43.91 |
| LightGBM (fixed config)       |   22.53 |    32.83 |    24.17 |     36.97 |
| Point LSTM (MSE)              |   22.53 |    31.38 |     n/a  |      n/a  |
| Quantile LSTM, median         |   21.59 |    31.38 |    24.02 |     37.07 |

The test split was evaluated exactly once (8,207 scored rows), after all
decisions were locked on val. The point LSTM was not part of that locked
evaluation, so it has no test numbers.

Findings, stated plainly:

- The quantile-LSTM median beats LightGBM on test MAE by 0.61% (it was 4.16%
  on val). On test RMSE, LightGBM is slightly ahead (36.97 vs 37.07). The
  deep model's edge is real but small.
- Calibration degraded out of sample. The 10-90 interval covered 79.8% on val
  but 76.5% on test, against an 80% target. With 8,207 test rows that drop is
  roughly 7 to 8 standard errors: a real effect, not noise. Quantile crossing
  was 0% on both splits.
- Val-to-test MAE growth: persistence +3.6%, LightGBM +7.3%, quantile LSTM
  +11.2%. The LSTM generalized worst, consistent with mild overfitting to the
  validation period (its val loss bottoms at epoch 4 and rises afterwards).

Sources: `results/phase3/baselines.json`, `lstm_val_metrics.json`,
`quantile_val_metrics.json`, `test_metrics.json`.

### Phase 4: battery arbitrage

Setup: 1 MW / 2 MWh battery (2-hour duration per the German new-build trend,
Battery-Charts/RWTH; 85% round-trip efficiency per NREL ATB 2024), hourly
day-ahead products, daily schedules committed at the D-1 gate from each
scenario's price signal and settled at actual prices. 340 delivery days
(2025-07-17 to 2026-06-29), 1,020 daily MILPs, all solved to optimality at
MIPGap 0.

| Scenario             | Total EUR | EUR/MW/yr | % of perfect | Cycles/day |
| -------------------- | --------: | --------: | -----------: | ---------: |
| Perfect foresight    |    75,365 |    80,916 |        100.0 |       1.58 |
| Quantile-LSTM median |    67,742 |    72,731 |         89.9 |       1.62 |
| Persistence          |    63,073 |    67,719 |         83.7 |       1.57 |

- **Value of forecast quality: +5,012 EUR/MW/yr** (LSTM vs persistence). That
  closes 38% of the gap between naive forecasting and perfect foresight. The
  remaining gap to perfect is 8,185 EUR/MW/yr.
- Expected vs realized revenue is asymmetric. Persistence plans 75,100 EUR
  but realizes 63,073, a 19% overestimate: yesterday's spreads flatter
  tomorrow's reality. The LSTM plans 62,096 and realizes 67,742, an 8%
  underestimate: the median forecast damps price extremes, so its schedule is
  worth more at real prices than on paper.
- Physical sanity holds in every scenario: charging concentrates at low and
  negative prices (charge-weighted average 60 to 66 EUR/MWh, including paid
  charging in negative-price hours), discharging at peaks (134 to 141
  EUR/MWh). See `results/phase4/sample_week.png`.

Source: `results/phase4/arbitrage_summary.json`.

## Known limitations

Dispatch model (Phase 1):

- Gas underdispatch (2.9 vs 55.9 TWh): the electricity-only model has no
  operating-reserve or CHP heat coupling, so gas that runs for non-energy
  reasons in reality is displaced by cheaper units.
- Nuclear overdispatch (20.9 vs 6.7 TWh): annual fleet granularity models the
  roughly 4 GW that ran January to April 2023 as running the full year.
- Wind and solar overdispatch: Germany-only scope with no exports, so surplus
  renewables are consumed domestically.
- Model prices have no negative-price mechanism and no scarcity spikes
  (model min 0.02 vs real -500; model max 118 vs real 524 EUR/MWh).
- Cost vintage is 2025 (nearest available) against 2023 actuals, partially
  corrected with sourced 2023 fuel and CO2 prices; every override is traced
  to a source in `config/config.germany-15node.yaml`.

Forecasting and arbitrage (Phases 2-4):

- Zonal scope only: one DE-LU price. Nodal price and congestion forecasting
  on the 15-node base is deferred future work.
- Hourly products only: 15-minute day-ahead prices (since October 2025) are
  aggregated to hourly means. This understates achievable revenue in that era
  for all scenarios equally.
- Battery degradation is excluded. Revenues are optimistic, and the optimizer
  cycles about 1.6 times per day, above typical warranty-limited operation.
- Grid fees and levies are assumed zero (German storage exemption,
  EnWG 118(6)); no intraday recourse or revenue stacking.
- Interval calibration drifts out of sample: 76.5% coverage on test against
  the 80% target, a statistically significant drop.
- Perfect foresight is a ceiling under this operational convention (daily
  cycle, 50% start and end state of charge), not an unconstrained maximum.
- One test year (July 2025 to June 2026), one market regime. 773 hours are
  dropped from the dataset because ENTSO-E never published them, and 16 test
  days are excluded from the arbitrage window for data-completeness reasons
  (both documented).

## License

[MIT](LICENSE). `pypsa-eur/` is a separate upstream project with its own
license; it is not redistributed here (you clone it yourself).
