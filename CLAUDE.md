# CLAUDE.md — de-battery-forecasting

## Project goal

Quantify the **€/MW/yr value of forecast quality** for battery arbitrage in
German electricity markets. We build a multi-node PyPSA-Eur dispatch model of
Germany (target **15 nodes** initially, scaling toward 15–20+), add a
deep-learning forecasting layer (node-level price / congestion forecasting),
and feed those forecasts into a battery storage trading/dispatch optimizer.
The headline question: how much economic value does better forecasting unlock
for battery arbitrage?

## Two-codebase architecture (they NEVER share a Python interpreter)

This repo deliberately keeps two separate codebases with two separate
environment managers. They communicate **only via files on disk**, never by
importing each other.

| Concern            | Codebase     | Env manager | Notes                                    |
| ------------------ | ------------ | ----------- | ---------------------------------------- |
| Dispatch model     | `pypsa-eur/` | **pixi**    | Upstream PyPSA-Eur clone (gitignored), its own pinned env |
| Forecasting / ML   | `ml/`        | **mamba**   | Our code, conda env `energy-ml`          |

- **Data exchange contract:** PyPSA-Eur writes network/results as `.nc`
  (NetCDF) network files and `.csv` / `.parquet` tables into `data/` (or its
  own `results/`). The `ml/` code reads those files, trains/forecasts, and
  writes `.csv` / `.parquet` back into `data/` for the optimizer / next solve.
- Do **not** try to `conda activate` one env and import the other's packages.
  If you find yourself wanting a shared import, that's a signal to instead
  define a file-based interface.

## Folder layout

```
pypsa-eur/   cloned PyPSA-Eur repo (managed by pixi; generated dirs gitignored)
config/      PyPSA-Eur config overrides (e.g. config.germany-15node.yaml)
scripts/     runnable shell drivers (e.g. phase1_solve.sh — build & solve)
analysis/    our inspection/validation scripts (run inside the pixi env)
ml/          our DL/forecasting code (mamba env: energy-ml)
data/        raw + processed data, large files (gitignored, kept via .gitkeep)
notebooks/   analysis notebooks
results/     figures/tables (gitignored except deliberately committed outputs)
docs/        write-ups
```

## Hardware constraints

- **24 GB RAM ceiling** (WSL2/Ubuntu), ~900 GB+ disk free.
- RAM is the binding constraint on model size. **Node count and time
  resolution must be chosen deliberately** — start at **15 nodes**. Increasing
  nodes, hours of resolution, or co-optimized storage all raise the LP/MILP
  size and memory footprint. Validate memory headroom before scaling up.

## Operating rules for Claude Code

- **Never auto-launch a long-running solve and wait on it.** Any solve or job
  expected to take multiple minutes or longer must be written as a **script
  for the user to run manually** and report results back. Claude Code prepares
  the script and explains how to run it; it does not block on the solve.
- Show output after each major step; stop and surface the exact error on
  failure rather than guessing a fix.
- Ask before installing significant new software or downloading large files.

## PyPSA-Eur version pin

`pypsa-eur/` is an upstream clone and is **gitignored** (not tracked in our
repo). To reproduce the exact version used here:

```
git clone https://github.com/PyPSA/pypsa-eur.git pypsa-eur
cd pypsa-eur && git checkout 9423f94cb87daaad811aec14492f633b2b86541d
# == v2026.02.0-76-g9423f94c
pixi install
```

## Gurobi license

- The Gurobi WLS academic license lives at `~/gurobi.lic` and is **already
  activated**. Never read, print, edit, or ask about its contents. License
  reachability is verified only via `gurobipy` (no credentials printed).

## Current phase status

- **Phase 0 — environment & toolchain setup (COMPLETE).** Folder structure, git,
  gitignore, env specs, tool verification.
- **Phase 1 complete — dispatch model validated, known limitations documented.**
  15-node Germany electricity-only dispatch for 2023, validated vs ENTSO-E:
  price mean 81.9 vs 95.2 EUR/MWh (hourly corr ~0.75); lignite 76.8 vs 77.8 TWh.
  Remaining gaps are structural and documented (gas underdispatch = no
  reserve/CHP coupling; nuclear over = annual fleet granularity; wind/solar over
  = Germany-only, no exports). Results in `results/germany-15node/validation/`.
- Next: **Phase 2** — forecasting layer + battery arbitrage on this dispatch base.
