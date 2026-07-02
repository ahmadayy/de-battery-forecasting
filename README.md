# de-battery-forecasting

**How much economic value does better forecasting unlock for battery arbitrage?**

This project quantifies the **€/MW/yr value of forecast quality** for battery
storage arbitrage in the German electricity market. The approach:

1. Build a multi-node **PyPSA-Eur** dispatch model of Germany (starting at 15
   nodes, scaling toward 15–20+).
2. Add a **deep-learning forecasting layer** (node-level price / congestion
   forecasting).
3. Feed those forecasts into a **battery storage trading/dispatch optimizer**
   and measure how forecast accuracy translates into arbitrage revenue.

> **Status:** Phase 1 complete — the 15-node Germany dispatch model is built and
> validated against ENTSO-E 2023 (numbers below). Phase 2 (forecasting +
> battery arbitrage) is next.

---

## Architecture: two codebases, never one interpreter

The repo deliberately keeps two separate codebases with two separate
environment managers. **They communicate only via files on disk** (`.nc`,
`.csv`, `.parquet`) — neither imports the other.

| Concern           | Location     | Env manager | Notes                                          |
| ----------------- | ------------ | ----------- | ---------------------------------------------- |
| Dispatch model    | `pypsa-eur/` | **pixi**    | Upstream PyPSA-Eur clone (gitignored, pinned)  |
| Forecasting / ML  | `ml/`        | **mamba**   | Our code, conda env `energy-ml`                |

Supporting our-code directories (all run against the `pypsa-eur/` pixi env):

```
config/     PyPSA-Eur config override (config.germany-15node.yaml)
scripts/    runnable shell drivers (phase1_solve.sh — build & solve)
analysis/   inspection & validation scripts (inspect_fleet, validate_dispatch)
data/       raw + processed data / caches (gitignored, kept via .gitkeep)
results/    figures/tables (gitignored except deliberately committed outputs)
notebooks/  analysis notebooks
docs/       write-ups
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design decisions and rationale.

---

## Reproduce Phase 1

### 1. Prerequisites

- **~24 GB RAM** (the binding constraint on model size), ~50+ GB free disk for
  the weather cutout and intermediate artifacts.
- A **Gurobi** license at `~/gurobi.lic` (the solve uses Gurobi).
- An `.env` file at the repo root (gitignored) with an ENTSO-E API token used
  by the validation step:

  ```
  ENTSOE_API_TOKEN=<your token>
  # optional, for renewables data:
  RENEWABLES_NINJA_TOKEN=<your token>
  ```

### 2. Clone PyPSA-Eur at the pinned commit

`pypsa-eur/` is an upstream clone and is **not** tracked in this repo.

```bash
git clone https://github.com/PyPSA/pypsa-eur.git pypsa-eur
cd pypsa-eur && git checkout 9423f94cb87daaad811aec14492f633b2b86541d
# == v2026.02.0-76-g9423f94c
pixi install
cd ..
```

### 3. Build & solve the 15-node dispatch model

The driver never runs automatically — you pass an explicit mode. The full
build+solve downloads tens of GB and runs for a while, so run it yourself:

```bash
scripts/phase1_solve.sh plan     # dry-run: list jobs, download nothing
scripts/phase1_solve.sh cutout   # download only the ~6.6 GB weather cutout
scripts/phase1_solve.sh solve    # full pipeline: retrieve -> build -> solve
```

Output network: `pypsa-eur/results/germany-15node/networks/base_s_15_elec_.nc`.
Concurrency/memory can be tuned via env vars, e.g. `JOBS=8 MEM_MB=20000 scripts/phase1_solve.sh solve`.

### 4. Validate against ENTSO-E 2023

```bash
cd pypsa-eur
pixi run python ../analysis/validate_dispatch.py
```

This loads the **real** solved network and **real** ENTSO-E data, compares
prices and the generation mix, and writes CSVs + a plot to
`results/germany-15node/validation/`. It fails loudly if either input is
missing — it never fabricates or substitutes placeholder values.

(Optional: `pixi run python ../analysis/inspect_fleet.py` dumps the German
power-plant fleet by fuel type to help choose the fleet-vintage filter.)

---

## Phase 1 results (15-node DE, electricity-only, year 2023)

Model vs ENTSO-E day-ahead prices, hourly, 2023:

| Metric (EUR/MWh) | Model | ENTSO-E |
| ---------------- | ----: | ------: |
| mean             | 81.88 |   95.18 |
| std              | 37.87 |   47.58 |
| min              |  0.02 | −500.00 |
| max              | 117.95 |  524.27 |

Hourly Pearson correlation (model vs ENTSO-E) ≈ **0.75** (printed by the
validation script).

Generation mix, full-year 2023 (TWh):

| Carrier   | Model | ENTSO-E |  Diff |
| --------- | ----: | ------: | ----: |
| Nuclear   | 20.93 |    6.74 | +14.19 |
| Lignite   | 76.79 |   77.84 |  −1.05 |
| Hard Coal | 25.17 |   39.75 | −14.58 |
| Gas       |  2.91 |   55.94 | −53.03 |
| Oil       |  0.00 |    3.15 |  −3.15 |
| Biomass   | 59.45 |   37.47 | +21.98 |
| Hydro     | 28.32 |   25.27 |  +3.05 |
| Wind      | 180.79 |  143.01 | +37.78 |
| Solar     | 77.55 |   55.80 | +21.75 |
| Other     |  3.09 |    9.72 |  −6.63 |

### Known limitations (structural, documented — not tuned away)

- **Gas underdispatch** (2.9 vs 55.9 TWh): no reserve / CHP-heat coupling, so
  gas that runs for non-energy reasons in reality is displaced by cheaper units.
- **Nuclear over** (20.9 vs 6.7 TWh): annual fleet granularity models the ~4 GW
  that ran Jan–Apr 2023 as running the full year.
- **Wind/solar over**: Germany-only scope with no exports, so surplus renewables
  are consumed domestically rather than exported.

A price-level gap is also expected from the cost vintage (2025 techno-economic
data — the nearest available — vs 2023 actuals), partially corrected with
independently-sourced 2023 fuel/CO₂ prices (see
[`config/config.germany-15node.yaml`](config/config.germany-15node.yaml) for
every override, each traced to a source).

---

## License

[MIT](LICENSE). Note that `pypsa-eur/` is a separate upstream project with its
own license; it is not redistributed here (you clone it yourself).
