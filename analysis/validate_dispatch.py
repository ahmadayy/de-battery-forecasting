#!/usr/bin/env python
"""
Validate the solved 15-node Germany dispatch network against real ENTSO-E data.

INTEGRITY CONTRACT (see project docs/ARCHITECTURE.md):
  * Uses the REAL solved network and REAL ENTSO-E data only.
  * NEVER fabricates, mocks, or substitutes placeholder values.
  * Fails LOUDLY (non-zero exit) if either input is missing or malformed,
    rather than "completing" on synthetic data.
  * Reports whatever the numbers actually are, including a poor match.

Run inside the pypsa-eur pixi env (has pypsa, entsoe-py, pandas, matplotlib):
    cd ~/projects/de-battery-forecasting/pypsa-eur
    pixi run python ../analysis/validate_dispatch.py

Reads ENTSOE_API_TOKEN from the environment (or the project .env). The token
value is NEVER printed or written to any output.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NETWORK_FILE = PROJECT_ROOT / "pypsa-eur" / "results" / "germany-15node" / "networks" / "base_s_15_elec_.nc"
OUT_DIR = PROJECT_ROOT / "results" / "germany-15node" / "validation"

# ENTSO-E bidding zone for Germany (DE-LU since Oct 2018).
PRICE_ZONE = "DE_LU"
GEN_ZONE = "DE_LU"

# ---- carrier / production-type -> common comparison category ----------------
MODEL_CARRIER_TO_CAT = {
    "nuclear": "Nuclear", "lignite": "Lignite", "coal": "Hard Coal",
    "CCGT": "Gas", "OCGT": "Gas", "oil": "Oil", "biomass": "Biomass",
    "onwind": "Wind", "offwind-ac": "Wind", "offwind-dc": "Wind",
    "offwind-float": "Wind", "solar": "Solar", "solar-hsat": "Solar",
    "ror": "Hydro", "hydro": "Hydro", "PHS": "Hydro",
    "geothermal": "Other", "waste": "Other",
}
ENTSOE_TYPE_TO_CAT = {
    "Nuclear": "Nuclear",
    "Fossil Brown coal/Lignite": "Lignite",
    "Fossil Hard coal": "Hard Coal",
    "Fossil Gas": "Gas", "Fossil Coal-derived gas": "Gas",
    "Fossil Oil": "Oil", "Fossil Oil shale": "Oil",
    "Biomass": "Biomass",
    "Hydro Run-of-river and poundage": "Hydro",
    "Hydro Water Reservoir": "Hydro",
    "Hydro Pumped Storage": "Hydro",
    "Wind Onshore": "Wind", "Wind Offshore": "Wind",
    "Solar": "Solar",
}
CATEGORIES = ["Nuclear", "Lignite", "Hard Coal", "Gas", "Oil",
              "Biomass", "Hydro", "Wind", "Solar", "Other"]


def die(msg: str) -> None:
    sys.exit(f"FATAL: {msg}")


def load_env_token() -> str:
    if not os.environ.get("ENTSOE_API_TOKEN"):
        envf = PROJECT_ROOT / ".env"
        if envf.exists():
            for line in envf.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    token = os.environ.get("ENTSOE_API_TOKEN")
    if not token:
        die("ENTSOE_API_TOKEN not found in environment or project .env.")
    return token  # never logged


def load_network():
    if not NETWORK_FILE.exists():
        die(f"solved network not found at {NETWORK_FILE} — run the solve first (scripts/phase1_solve.sh solve).")
    import pypsa
    n = pypsa.Network(str(NETWORK_FILE))
    if n.snapshots.empty or n.buses.empty:
        die("network loaded but has no snapshots/buses — malformed solve output.")
    if n.buses_t.marginal_price.empty:
        die("network has no marginal_price — was it actually solved (optimal)?")
    return n


def report_capacities(n) -> pd.Series:
    cap = n.generators.groupby("carrier").p_nom.sum().sort_values(ascending=False)
    print("\n--- Installed generator capacity by carrier (GW), from solved network ---")
    print((cap / 1e3).round(2).to_string())
    print("(Sanity check vs IRENA-2023 expectation: solar ~74.9, onwind ~61, offwind-ac ~8.5 GW)")
    return cap


def model_price(n) -> pd.Series:
    """Load-weighted average nodal marginal price -> hourly DE system price (EUR/MWh)."""
    mp = n.buses_t.marginal_price
    load = n.loads_t.p_set if not n.loads_t.p_set.empty else n.loads_t.p
    bus_of_load = n.loads.bus
    load_by_bus = load.T.groupby(bus_of_load).sum().T.reindex(columns=mp.columns).fillna(0.0)
    w = load_by_bus.div(load_by_bus.sum(axis=1), axis=0).fillna(0.0)
    price = (mp * w).sum(axis=1)
    price.index = pd.to_datetime(price.index)  # treat as UTC-naive
    return price


def model_generation_twh(n) -> pd.Series:
    weight = n.snapshot_weightings.generators
    out: dict[str, float] = {}
    if not n.generators_t.p.empty:
        e = n.generators_t.p.multiply(weight, axis=0).sum()  # MWh per generator
        by_car = e.groupby(n.generators.carrier).sum()
        for car, mwh in by_car.items():
            cat = MODEL_CARRIER_TO_CAT.get(car, "Other")
            out[cat] = out.get(cat, 0.0) + mwh
    if not n.storage_units.empty and not n.storage_units_t.p.empty:
        e = n.storage_units_t.p.clip(lower=0).multiply(weight, axis=0).sum()
        by_car = e.groupby(n.storage_units.carrier).sum()
        for car, mwh in by_car.items():
            cat = MODEL_CARRIER_TO_CAT.get(car, "Other")
            out[cat] = out.get(cat, 0.0) + mwh
    s = pd.Series(out) / 1e6  # MWh -> TWh
    return s.reindex(CATEGORIES).fillna(0.0)


def fetch_entsoe(token: str):
    from entsoe import EntsoePandasClient
    client = EntsoePandasClient(api_key=token)
    start = pd.Timestamp("2023-01-01", tz="Europe/Brussels")
    end = pd.Timestamp("2024-01-01", tz="Europe/Brussels")

    try:
        prices = client.query_day_ahead_prices(PRICE_ZONE, start=start, end=end)
    except Exception as e:
        die(f"ENTSO-E day-ahead price query failed: {type(e).__name__}: {e}")
    if prices is None or prices.empty:
        die("ENTSO-E returned no day-ahead prices for 2023 — cannot validate.")
    prices = prices.tz_convert("UTC").tz_localize(None)
    prices = prices[~prices.index.duplicated(keep="first")].resample("1h").mean()

    try:
        gen = client.query_generation(GEN_ZONE, start=start, end=end, psr_type=None)
    except Exception as e:
        die(f"ENTSO-E generation query failed: {type(e).__name__}: {e}")
    if gen is None or gen.empty:
        die("ENTSO-E returned no generation data for 2023 — cannot validate.")
    if isinstance(gen.columns, pd.MultiIndex):  # keep 'Actual Aggregated'
        keep = [c for c in gen.columns if c[1] == "Actual Aggregated"]
        gen = gen[keep]
        gen.columns = [c[0] for c in keep]
    gen = gen.tz_convert("UTC").tz_localize(None)
    gen = gen.resample("1h").mean()  # MW hourly -> sum gives MWh
    return prices, gen


def entsoe_generation_twh(gen: pd.DataFrame) -> pd.Series:
    out: dict[str, float] = {}
    for col in gen.columns:
        cat = ENTSOE_TYPE_TO_CAT.get(col, "Other")
        out[cat] = out.get(cat, 0.0) + gen[col].clip(lower=0).sum()  # MWh
    s = pd.Series(out) / 1e6  # -> TWh
    return s.reindex(CATEGORIES).fillna(0.0)


def main() -> None:
    token = load_env_token()
    n = load_network()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report_capacities(n)
    m_price = model_price(n)
    m_gen = model_generation_twh(n)

    r_price, r_gen_df = fetch_entsoe(token)
    r_gen = entsoe_generation_twh(r_gen_df)

    # ---- align prices on common hourly timestamps ----
    common = m_price.index.intersection(r_price.index)
    if len(common) == 0:
        die("no overlapping timestamps between model and ENTSO-E prices — check timezone/period alignment.")
    mp, rp = m_price.reindex(common), r_price.reindex(common)
    valid = mp.notna() & rp.notna()
    mp, rp = mp[valid], rp[valid]
    if len(mp) < 24:
        die(f"only {len(mp)} overlapping valid price hours — too few to validate.")

    corr = float(np.corrcoef(mp.values, rp.values)[0, 1])
    price_summary = pd.DataFrame({
        "model": [mp.mean(), mp.std(), mp.min(), mp.max()],
        "entsoe": [rp.mean(), rp.std(), rp.min(), rp.max()],
    }, index=["mean", "std", "min", "max"]).round(2)

    gen_summary = pd.DataFrame({"model_TWh": m_gen, "entsoe_TWh": r_gen}).round(2)
    gen_summary["diff_TWh"] = (gen_summary["model_TWh"] - gen_summary["entsoe_TWh"]).round(2)

    print("\n================ PRICE (EUR/MWh) ================")
    print(price_summary.to_string())
    print(f"\nHourly Pearson correlation (model vs ENTSO-E): {corr:.3f}")
    print(f"Mean absolute error: {np.abs(mp - rp).mean():.2f} EUR/MWh")
    print(f"Overlapping hours used: {len(mp)}")
    print("\n================ GENERATION MIX (TWh, full year 2023) ================")
    print(gen_summary.to_string())
    print(f"\nTotal model: {gen_summary.model_TWh.sum():.1f} TWh   "
          f"Total ENTSO-E: {gen_summary.entsoe_TWh.sum():.1f} TWh")

    price_summary.to_csv(OUT_DIR / "price_summary.csv")
    gen_summary.to_csv(OUT_DIR / "generation_summary.csv")

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(np.sort(mp.values)[::-1], label="model", lw=1)
    ax[0].plot(np.sort(rp.values)[::-1], label="ENTSO-E", lw=1)
    ax[0].set_title(f"DE price duration curve 2023 (corr={corr:.2f})")
    ax[0].set_xlabel("hours"); ax[0].set_ylabel("EUR/MWh"); ax[0].legend()

    x = np.arange(len(CATEGORIES)); w = 0.4
    ax[1].bar(x - w / 2, m_gen.values, w, label="model")
    ax[1].bar(x + w / 2, r_gen.values, w, label="ENTSO-E")
    ax[1].set_xticks(x); ax[1].set_xticklabels(CATEGORIES, rotation=45, ha="right")
    ax[1].set_ylabel("TWh"); ax[1].set_title("Generation mix 2023"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "validation.png", dpi=120)
    print(f"\nWrote plots + CSVs to {OUT_DIR}")
    print("\nNOTE: interpret honestly. A price-level gap is expected from cost-vintage "
          "(2025 projected fuel/CO2 vs 2023 actuals) and is a diagnosis target, not "
          "something to tune away. See Phase 1 notes.")


if __name__ == "__main__":
    main()
