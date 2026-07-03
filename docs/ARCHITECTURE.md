# Architecture and design decisions

## Project goal

Quantify the EUR/MW/yr value of forecast quality for battery arbitrage in the
German power market. The pipeline has three parts: a validated multi-node
PyPSA-Eur dispatch model of Germany (Phase 1), a leakage-safe forecasting
dataset and price forecasters trained on real DE-LU market data (Phases 2
and 3), and a battery dispatch optimizer that converts forecast accuracy into
revenue (Phase 4). Headline answer: better forecasting is worth
+5,012 EUR/MW/yr for a 1 MW / 2 MWh battery; details in the phase log below.

## Two-codebase design

This repo keeps two separate codebases with two separate environment
managers. They never share a Python interpreter and communicate only via
files on disk.

| Concern          | Codebase     | Env manager | Notes                                                     |
| ---------------- | ------------ | ----------- | --------------------------------------------------------- |
| Dispatch model   | `pypsa-eur/` | pixi        | Upstream PyPSA-Eur clone (gitignored), its own pinned env |
| Forecasting / ML | `ml/`        | mamba       | Our code, conda env `energy-ml`                           |

Why: PyPSA-Eur ships and pins its own environment, and foreign package pins
break it easily. The ML stack (PyTorch, LightGBM) has no business inside it.
One interpreter per concern keeps each side reproducible on its own.

Data exchange contract: PyPSA-Eur writes networks and results as `.nc` and
`.csv`/`.parquet` files; the `ml/` code reads files, trains and forecasts, and
writes files back (for example `data/processed/test_forecasts.csv`, consumed
by the Phase 4 optimizer running in the pixi env). Every hand-off between
stages is a file that can be inspected and versioned. Do not `conda activate`
one env and import the other's packages; wanting a shared import is the
signal to define a file interface instead.

Folder layout:

```
pypsa-eur/   cloned PyPSA-Eur repo (managed by pixi; gitignored)
config/      PyPSA-Eur config overrides (config.germany-15node.yaml)
scripts/     shell drivers (phase1_solve.sh)
analysis/    scripts run inside the pixi env (validation, battery optimizer)
ml/          forecasting pipeline (mamba env: energy-ml)
data/        raw and processed data, large files (gitignored)
notebooks/   scratch space
results/     outputs (gitignored except deliberately committed results)
docs/        this file, phase2_leakage_audit.md
```

## Hardware constraints and their consequences

The machine is WSL2/Ubuntu with a 24 GB RAM ceiling and about 900 GB free
disk. Consequences:

- RAM bounds the dispatch problem size. 15 nodes at hourly resolution for one
  year, dispatch-only, is the deliberate starting point. More nodes, finer
  time resolution, or capacity expansion all grow the LP beyond the ceiling.
- `scripts/phase1_solve.sh` caps Snakemake's memory budget (`MEM_MB`,
  `SOLVE_MEM_MB`) so the build and solve stay inside 24 GB; the upstream
  solve-memory heuristic (38,775 MB) is overridden to 18,000 MB as a
  scheduling hint.
- PyTorch is the CPU build (no CUDA assumed under WSL2). This shaped Phase 3:
  a 2-layer, hidden-64 LSTM over 168-hour windows trains in minutes on CPU;
  larger architectures were not justified for roughly 54k training rows.
- Disk is not binding. The 6.6 GB weather cutout and tens of GB of PyPSA-Eur
  artifacts stay out of git.

## Phase 1 design decisions (dispatch model)

Scope: Germany only, 15 nodes, electricity only, full year 2023 hourly,
dispatch only. Rationale for the year: a prebuilt weather cutout exists for
2023 and ENTSO-E fully covers it for validation.

- Dispatch-only is enforced by two config switches: empty
  `extendable_carriers` (nothing can be built) and `transmission_limit: v1.0`
  (the grid is fixed; the upstream default `vopt` would enable expansion).
- Fleet vintage 2023 via `powerplants_filter`. This keeps the roughly 4 GW of
  nuclear that ran January to April 2023; annual granularity models it as
  running the full year, a documented limitation.
- Renewable capacities are scaled to IRENA 2023 Germany totals (solar to
  74.9 GW, onshore wind about 61 GW, offshore 8.5 GW) because
  powerplantmatching alone undercounts distributed PV by roughly half.
- Cost vintage is 2025 (technology-data has no 2023 file). Two data
  corrections were applied after validation diagnosis, both sourced, neither
  tuned to fit:
  1. CO2 price 83.66 EUR/t, the 2023 EU ETS annual average (ICE/EEX via
     Statista). The default config had no carbon price at all, which
     collapsed the price level to about 22 EUR/MWh.
  2. Coal fuel 14.62 EUR/MWh_th (VDKi Jahresbericht 2024) and lignite
     1.5 EUR/MWh_th (Oeko-Institut 2022 / BNetzA grid development plan,
     variable mining cost). The upstream TYNDP figures priced coal below
     lignite, inverting the German merit order.
- Validation outcome: price mean 81.9 vs 95.2 EUR/MWh, hourly correlation
  about 0.75, lignite generation within 1.4% of actual. Remaining gaps are
  structural and listed in the README.

## Phase 2 design decisions (leakage-safe dataset)

- Leakage anchor: 12:00 CET on day D-1, the EPEX day-ahead auction gate
  closure for DE-LU. A feature is admissible only if its value was knowable
  at or before that instant. Every dataset row records this instant in a
  `forecast_origin` column.
- Publication timing was verified against Commission Regulation (EU)
  No 543/2013 rather than assumed. Article 6(2)(b): the day-ahead load
  forecast is due no later than two hours before gate closure, so it is
  pre-gate and used directly. Article 14(2)(d): the wind/solar day-ahead
  forecast is only due at 18:00 Brussels time on D-1, after the gate, so the
  D-1 vintage (published the day before, provably pre-gate) is used instead,
  shifted by 24 hours.
- `residual_load` is the day-D load forecast minus the D-1-vintage RES
  forecast. It is leakage-safe by construction but an approximation of true
  day-D residual load, and is documented as such.
- Missing data policy: never interpolate. 773 rows whose forecast values
  ENTSO-E never published were dropped explicitly with a manifest
  (`data/processed/_dropped_nan_rows.csv`); a targeted re-pull of exactly
  those windows recovered zero of them.
- Split and embargo: chronological train (2018-10 to 2024-12), val (2025-01
  to 2025-06), test (2025-07 to 2026-06), with an 8-day embargo at each
  boundary. The maximum feature lookback is 191 hours (a 168-hour rolling
  window shifted 24 hours), so a 7-day embargo would leak by 23 hours.
- Two pull bugs were found and fixed, both verified against real data or the
  live API: per-hour-bin resolution handling for EPEX's 2025 switch to
  15-minute day-ahead settlement (a global modal check had masked the mixed
  resolution), and sub-year request chunks to avoid entsoe-py's
  `@year_limited` splitting, which drops the hour 24 hours before the end of
  each roughly one-year block. After the fixes all three raw series have zero
  missing hours over 2018-2026.

## Phase 3 design decisions (forecasting models)

- Baselines come first. Persistence (the price 24 hours earlier) and a
  fixed-config LightGBM define the bar; no deep-learning result counts unless
  it is compared against them.
- The LSTM consumes the same 22 engineered features as LightGBM, in 168-hour
  windows, so the comparison isolates the architecture rather than the
  information set.
- Sequences are built per split from contiguous hourly runs only. No input
  window crosses a dropped-row hole, the embargo, or a split boundary
  (sampled and checked: zero violations). Scalers are fit on train only.
- Model selection happens on val only. The reported model is the
  best-validation checkpoint, and the selected epoch is recorded in the
  metrics JSON (epoch 4 of 11 for the quantile model).
- Probabilistic output uses pinball loss at quantiles 0.1/0.5/0.9. The loss
  implementation was verified against hand-computed cases before training
  (zero at exact match, asymmetric penalties per quantile).
- The test split was evaluated exactly once, after all decisions were locked.
  No tuning followed the test evaluation.

## Phase 4 design decisions (battery arbitrage)

- Daily rolling commitment. For each delivery day, the schedule over that
  day's hours is optimized at the D-1 gate using the scenario's price signal,
  then held binding and settled at actual prices. A single full-horizon
  optimization over concatenated forecasts would use forecasts that do not
  exist at decision time, so it is not used.
- One MILP per delivery day (local Europe/Brussels calendar day, 23/24/25
  hours across DST). Binary charge/discharge exclusivity is required: at
  negative prices a pure LP charges and discharges simultaneously, acting as
  a paid resistor, which a single inverter cannot do. The evaluation window
  contains 465 negative-price hours (minimum -499 EUR/MWh), so this is not
  theoretical.
- State of charge starts and ends every day at 50%. The forecast horizon is
  24 hours, so valuing energy carried across days would need a next-day
  forecast that does not exist at the gate. The convention applies to all
  scenarios including perfect foresight, which is therefore a ceiling under
  this convention, not an unconstrained maximum.
- Battery parameters are sourced, not invented: 1 MW / 2 MWh. Two-hour
  duration follows the German new-build trend (Battery-Charts / RWTH Aachen
  fleet data; Modo Energy, February 2026: "Two-hour systems now dominate new
  builds"). Round-trip efficiency 85% per NREL ATB 2024 (basis: Cole and
  Karmakar 2023), split as sqrt(0.85) per leg.
- Exclusions, identical across scenarios: degradation, grid fees (German
  storage exemption, EnWG 118(6)), intraday recourse, revenue stacking, and
  15-minute products (the price series is hourly).
- Fairness: all three scenarios run on the same 340 complete delivery days
  with identical physics; only the decision price differs. All 1,020 MILPs
  solved to optimality at MIPGap 0, asserted per day.

## Phase status

- **Phase 0 complete.** Folder structure, git hygiene, environment specs,
  toolchain verification.
- **Phase 1 complete.** 15-node Germany dispatch for 2023 validated against
  ENTSO-E: price mean 81.9 vs 95.2 EUR/MWh, hourly correlation about 0.75,
  lignite 76.8 vs 77.8 TWh. Structural gaps documented (gas underdispatch
  from missing reserve/CHP coupling, nuclear from annual fleet granularity,
  wind/solar surplus from the no-export scope). Results in
  `results/germany-15node/validation/`.
- **Phase 2 complete.** Leakage-safe dataset from real ENTSO-E DE-LU data:
  66,548 hourly rows, 22 features, 2018-10 to 2026-06, anchored on the
  12:00 CET D-1 gate. 773 unpublished rows dropped explicitly with a
  manifest. Chronological split with an 8-day embargo. Dataset in
  `data/processed/dataset.parquet`; per-feature timing in
  `docs/phase2_leakage_audit.md`.
- **Phase 3 complete.** Baselines plus point LSTM and quantile LSTM; test
  evaluated once (n = 8,207). Test MAE: persistence 27.33, LightGBM 24.17,
  quantile-LSTM median 24.02 EUR/MWh. The LSTM edge over LightGBM is small
  (0.61% on test MAE, down from 4.16% on val) and LightGBM is slightly ahead
  on test RMSE. The 10-90 interval covers 76.5% on test against an 80% target
  (79.8% on val), a statistically significant drop of roughly 7 to 8 standard
  errors. Results in `results/phase3/`.
- **Phase 4 complete.** Battery arbitrage under three price signals, 340
  delivery days, all MILPs optimal. Perfect foresight 80,916 EUR/MW/yr;
  quantile-LSTM median 72,731 (89.9% of perfect); persistence 67,719 (83.7%).
  Value of forecast quality: +5,012 EUR/MW/yr; remaining gap to perfect:
  8,185 EUR/MW/yr. Results in `results/phase4/`.
- **Deferred future work:** risk-aware dispatch on the q10/q90 band
  (pessimistic prices; needs distributional metrics, and the test
  under-coverage means the band under-hedges as-is), nodal price and
  congestion forecasting on the 15-node base, 15-minute products,
  degradation-aware operation.

## Data sources and citations

- ENTSO-E Transparency Platform: DE-LU day-ahead prices, day-ahead total load
  forecast [6.1.B], day-ahead wind/solar generation forecast [14.1.D].
  https://transparency.entsoe.eu
- Commission Regulation (EU) No 543/2013, Articles 6(2)(b) and 14(2)(d),
  used to verify forecast publication deadlines against the auction gate.
  https://www.legislation.gov.uk/eur/2013/543
- powerplantmatching v0.8.1 archived dataset (plant fleet).
  https://data.pypsa.org/workflows/eur/powerplants/0.8.1/powerplants.csv
- IRENA renewable capacity statistics, 2023 Germany totals (via PyPSA-Eur's
  `estimate_renewable_capacities`).
- VDKi Jahresbericht 2024: 2023 cross-border price for steam coal
  (119 EUR/t SKE, converted to 14.62 EUR/MWh_th).
  https://www.kohlenimporteure.de/files/user_upload/jahresberichte/Jahresbericht-2024.pdf
- Oeko-Institut (2022) and BNetzA grid development plan: lignite variable
  mining cost, 1.5 EUR/MWh_th.
- EU ETS 2023 annual average allowance price, 83.66 EUR/t (ICE/EEX via
  Statista).
  https://www.statista.com/statistics/1465687/average-annual-eu-ets-allowance-prices/
- Weather: prebuilt atlite cutout `europe-2023-sarah3-era5` (ERA5 plus
  SARAH-3).
- technology-data v0.14.0, 2025 cost vintage (nearest available to 2023).
- NREL Annual Technology Baseline 2024, utility-scale battery storage: 85%
  round-trip efficiency (basis: Cole and Karmakar 2023, NREL/TP-6A40-85332).
  https://atb.nrel.gov/electricity/2024/utility-scale_battery_storage
- Battery-Charts, ISEA RWTH Aachen (German battery fleet statistics),
  https://battery-charts.de; Figgener et al., arXiv:2203.06762; Modo Energy,
  Germany Battery Buildout Report, February 2026 ("Two-hour systems now
  dominate new builds").
- Python `holidays` package for nationwide German public holidays.

## Reproduction pins and environment notes

PyPSA-Eur is an upstream clone, gitignored, pinned to an exact commit:

```
git clone https://github.com/PyPSA/pypsa-eur.git pypsa-eur
cd pypsa-eur && git checkout 9423f94cb87daaad811aec14492f633b2b86541d
# == v2026.02.0-76-g9423f94c
pixi install
```

The ML environment is created from `ml/environment.yml` (conda env
`energy-ml`: PyTorch CPU, LightGBM, entsoe-py, holidays, scientific stack).

Gurobi: the WLS academic license lives at `~/gurobi.lic` and is already
activated. Never read, print, edit, or ask about its contents. License
reachability is verified only via `gurobipy`, without printing credentials.

## Operating rules (for assistants and automation)

- Never auto-launch a long-running solve and wait on it. Any job expected to
  take multiple minutes or longer is written as a script for the user to run
  manually and report back.
- Show output after each major step; stop and surface the exact error on
  failure rather than guessing a fix.
- Ask before installing significant new software or downloading large files.
