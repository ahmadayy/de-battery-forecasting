#!/usr/bin/env python
"""
Read-only inspection of the German power-plant fleet under different
`powerplants_filter` settings, BEFORE committing to a config.

It mirrors the preprocessing in pypsa-eur/scripts/build_powerplants.py
(country -> alpha2, gas/biomass relabel) so the capacities shown match what
PyPSA-Eur would actually build the network from. It performs NO solve and
fabricates NOTHING: it loads the real powerplantmatching dataset and reports
whatever is there. If the data is missing or malformed it fails loudly.

Run inside the pypsa-eur pixi env:
    cd ~/projects/de-battery-forecasting/pypsa-eur
    pixi run python ../analysis/inspect_fleet.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

# powerplantmatching v0.8.1 — same dataset/version PyPSA-Eur retrieves
# (data.versions.csv: powerplants, 0.8.1, archive).
PPL_URL = "https://data.pypsa.org/workflows/eur/powerplants/0.8.1/powerplants.csv"
CACHE = Path(__file__).resolve().parent.parent / "data" / "cache" / "powerplants_0.8.1.csv"

COUNTRY = "DE"

# Candidate filters. Keys mirror exactly what would go into
# electricity.powerplants_filter in the config.
FILTERS = {
    "default (~2026 fleet)": "(DateOut > 2025 or DateOut != DateOut) and (DateIn < 2026 or DateIn != DateIn)",
    "aligned to 2023":       "(DateOut >= 2023 or DateOut != DateOut) and (DateIn <= 2023 or DateIn != DateIn)",
}


def load_ppl() -> pd.DataFrame:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"[inspect_fleet] downloading powerplants.csv (~21 MB) -> {CACHE}")
        urlretrieve(PPL_URL, CACHE)
    if not CACHE.exists() or CACHE.stat().st_size == 0:
        sys.exit(f"FATAL: powerplant data not available at {CACHE}")

    try:
        import powerplantmatching as pm
    except ImportError:
        sys.exit("FATAL: powerplantmatching not importable — run inside the pixi env.")

    ppl = (
        pd.read_csv(CACHE, index_col=0, header=[0])
        .pipe(pm.collection.parse_string_to_dict, ["projectID", "EIC"])
        .pipe(pm.collection.set_column_name, "Matched Data")
    )
    ppl = (
        ppl.powerplant.convert_country_to_alpha2()
        .query("Country == @COUNTRY")
        # no gas remap needed for Fueltype totals; only relabel biomass variants
        .replace({"Solid Biomass": "Bioenergy", "Biogas": "Bioenergy"})
    )
    if ppl.empty:
        sys.exit(f"FATAL: no {COUNTRY} powerplants found after country filter — data malformed?")
    for col in ("Fueltype", "Capacity", "DateIn", "DateOut"):
        if col not in ppl.columns:
            sys.exit(f"FATAL: expected column '{col}' missing from powerplant data.")
    return ppl


def summarize(ppl: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for label, query in FILTERS.items():
        sub = ppl.query(query)
        by_fuel = sub.groupby("Fueltype")["Capacity"].sum().round(0)
        out[label] = by_fuel
    table = pd.DataFrame(out).fillna(0.0)
    table.loc["— TOTAL —"] = table.sum()
    return table


def main() -> None:
    ppl = load_ppl()
    print(f"\nGermany ({COUNTRY}) powerplantmatching fleet — capacity in MW by Fueltype\n")
    table = summarize(ppl)
    pd.set_option("display.float_format", lambda x: f"{x:,.0f}")
    print(table.to_string())

    # Highlight the differences that matter for 2023 validation.
    diff = (table["aligned to 2023"] - table["default (~2026 fleet)"]).round(0)
    print("\nDelta (aligned-2023 minus default) by Fueltype, MW:\n")
    print(diff[diff != 0].to_string())

    # Nuclear sanity line (Germany shut its last reactors in April 2023).
    nuc = ppl.query("Fueltype == 'Nuclear'")
    if not nuc.empty:
        print("\nNuclear units present in raw DE data (DateIn / DateOut / MW):")
        print(nuc[["Capacity", "DateIn", "DateOut"]].sort_values("DateOut").to_string())
    else:
        print("\nNo nuclear units in raw DE powerplantmatching data.")

    print(
        "\nNOTE: powerplantmatching is plant-level; distributed solar/onshore wind "
        "may be undercounted here. This dump is for choosing the time filter, not a "
        "final capacity validation."
    )


if __name__ == "__main__":
    main()
