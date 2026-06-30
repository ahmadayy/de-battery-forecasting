#!/usr/bin/env bash
# =============================================================================
# Phase 1 — build & solve the 15-node Germany DISPATCH-ONLY network (year 2023).
#
# This drives PyPSA-Eur via Snakemake inside its pixi environment. It NEVER runs
# automatically — you must pass an explicit mode. The full build+solve downloads
# ~tens of GB and runs for a while (see the Phase 1 notes); run it yourself and
# report back the tail of the output.
#
# Usage (from anywhere):
#   scripts/phase1_solve.sh plan      # dry-run: list all jobs, download NOTHING
#   scripts/phase1_solve.sh cutout    # download ONLY the 6.6 GB weather cutout
#   scripts/phase1_solve.sh solve     # full pipeline: retrieve -> build -> solve
#
# Optional overrides:  JOBS=8 MEM_MB=20000 scripts/phase1_solve.sh solve
# =============================================================================
set -euo pipefail

MODE="${1:-}"
JOBS="${JOBS:-8}"             # max parallel Snakemake jobs (16 cores available)
MEM_MB="${MEM_MB:-20000}"    # global memory budget; gates concurrency < 24 GB RAM
SOLVE_MEM_MB="${SOLVE_MEM_MB:-18000}"  # override solve_network's 38775 MB heuristic
                             # (a scheduling hint only; does NOT cap Gurobi or force swap)

export PATH="$HOME/.pixi/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYPSA_DIR="$PROJECT_ROOT/pypsa-eur"
CONFIG="../config/config.germany-15node.yaml"   # relative to PYPSA_DIR
TARGET="results/germany-15node/networks/base_s_15_elec_.nc"
CUTOUT="data/cutout/archive/v1.0/europe-2023-sarah3-era5.nc"

if [[ ! -d "$PYPSA_DIR" ]]; then
  echo "FATAL: pypsa-eur clone not found at $PYPSA_DIR" >&2; exit 1
fi
cd "$PYPSA_DIR"

case "$MODE" in
  plan)
    echo "[plan] Dry-run only — no downloads, no solve."
    exec pixi run snakemake -n "$TARGET" --configfile "$CONFIG"
    ;;
  cutout)
    echo "[cutout] Downloading ONLY the ~6.6 GB weather cutout for 2023."
    exec pixi run snakemake -j1 "$CUTOUT" --configfile "$CONFIG"
    ;;
  solve)
    echo "[solve] Full pipeline: retrieve -> build -> solve."
    echo "[solve] JOBS=$JOBS  MEM_MB=$MEM_MB  SOLVE_MEM_MB=$SOLVE_MEM_MB  (override via env vars)"
    echo "[solve] solve_network's 38775 MB heuristic is overridden to $SOLVE_MEM_MB MB"
    echo "[solve] (scheduling hint only; Gurobi still uses physical RAM as needed)"
    echo "[solve] Target: $TARGET"
    exec pixi run snakemake "$TARGET" \
        --configfile "$CONFIG" \
        -j "$JOBS" \
        --resources mem_mb="$MEM_MB" \
        --set-resources solve_network:mem_mb="$SOLVE_MEM_MB" \
        --rerun-incomplete \
        --keep-going
    ;;
  *)
    echo "Usage: $0 {plan|cutout|solve}" >&2
    echo "  plan    dry-run, no downloads" >&2
    echo "  cutout  download only the 6.6 GB cutout" >&2
    echo "  solve   full retrieve+build+solve" >&2
    exit 2
    ;;
esac
