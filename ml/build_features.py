#!/usr/bin/env python
"""
Phase 2 — build the leakage-safe feature dataset for DE-LU day-ahead price
forecasting from the raw ENTSO-E parquets (data/raw/entsoe/).

Forecast setup / leakage anchor:
  * Forecast origin = 12:00 CET on day D-1 (EPEX day-ahead gate closure).
  * We predict every hour of day D. A feature is admissible only if its value
    was knowable at or before that origin.
  * Price lags (>=24h multiples), 7-day rolling stats windowed to end at the
    D-1 equivalent hour, calendar features (deterministic, local time), the
    D load forecast (pre-gate), and the D-1-vintage RES forecast are all
    admissible; see docs/phase2_leakage_audit.md.

Integrity contract (Phase 1/2 discipline, non-negotiable):
  * Real ENTSO-E data only; no fabrication, no interpolation, no mocking.
  * Fail loudly if inputs are missing/misaligned.
  * The common range is trimmed STRUCTURALLY (warm-up + non-overlapping tails).
    Rows with genuinely-missing raw forecast values (NaN) are then dropped
    EXPLICITLY with a written manifest and a printed count — never silently
    absorbed, never filled. Any NaN after that is a hard error.

Run inside the energy-ml env (pure local compute, no network):
    mamba run -n energy-ml python ml/build_features.py
"""
from __future__ import annotations

from pathlib import Path

import holidays
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data" / "raw" / "entsoe"
PROCESSED = PROJECT_ROOT / "data" / "processed"
DATASET = PROCESSED / "dataset.parquet"
MANIFEST = PROCESSED / "_dropped_nan_rows.csv"
AUDIT_DOC = PROJECT_ROOT / "docs" / "phase2_leakage_audit.md"

PRICE_LAGS = {
    "price_lag_24h": 24,
    "price_lag_48h": 48,
    "price_lag_72h": 72,
    "price_lag_168h": 168,
}
ROLL_WINDOW = 168   # 7 days, hourly
ROLL_SHIFT = 24     # shift so the window ends at the D-1 equivalent hour (no day-D leak)

LOCAL_TZ = "Europe/Brussels"   # DE-LU market local time; calendar features derived here
RES_RENAME = {
    "Solar": "solar_d1",
    "Wind Offshore": "wind_offshore_d1",
    "Wind Onshore": "wind_onshore_d1",
}

# Oldest hour any feature reads: rolling = price.shift(24).rolling(168) -> window
# price[t-191h .. t-24h]. So MAX_LOOKBACK = 191h.
MAX_LOOKBACK = pd.Timedelta(hours=ROLL_SHIFT + ROLL_WINDOW - 1)  # 191h

SPLIT_B1 = pd.Timestamp("2025-01-01")   # train | val (UTC)
SPLIT_B2 = pd.Timestamp("2025-07-01")   # val | test (UTC)
EMBARGO = pd.Timedelta("8D")            # 192h >= MAX_LOOKBACK (7d=168h would be 23h short)


def _require_contiguous_hourly(idx: pd.DatetimeIndex, name: str) -> None:
    exp = pd.date_range(idx.min(), idx.max(), freq="1h")
    missing = exp.difference(idx)
    if len(missing):
        raise SystemExit(
            f"FATAL: {name} index is not contiguous hourly — {len(missing)} missing hours "
            f"(first: {list(missing[:3])}). Features assume a gap-free grid."
        )


def load_price() -> pd.Series:
    """Load the hourly DE-LU day-ahead price target; require gap-free, NaN-free."""
    s = pd.read_parquet(RAW / "da_price.parquet")["da_price"].sort_index()
    _require_contiguous_hourly(s.index, "da_price")
    if s.isna().any():
        raise SystemExit(f"FATAL: da_price has {int(s.isna().sum())} NaN VALUES — target must be complete.")
    return s


def build_price_features(price: pd.Series) -> pd.DataFrame:
    """Price lags + 7-day rolling stats, all leakage-safe (window ends at D-1)."""
    feats = {name: price.shift(h) for name, h in PRICE_LAGS.items()}
    roll = price.shift(ROLL_SHIFT).rolling(ROLL_WINDOW, min_periods=ROLL_WINDOW)
    feats["price_roll7d_mean"] = roll.mean()
    feats["price_roll7d_std"] = roll.std()
    feats["price_roll7d_min"] = roll.min()
    feats["price_roll7d_max"] = roll.max()
    return pd.DataFrame(feats)


def build_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic calendar features in LOCAL (Europe/Brussels) time.

    Nationwide German holidays only (no subdivision).
    """
    local = index.tz_localize("UTC").tz_convert(LOCAL_TZ)
    de_hol = holidays.Germany(years=range(local.year.min(), local.year.max() + 1))
    out = pd.DataFrame(index=index)
    out["hour"] = local.hour
    out["dayofweek"] = local.dayofweek                 # Mon=0 .. Sun=6
    out["month"] = local.month
    out["is_weekend"] = (local.dayofweek >= 5).astype(int)
    out["is_holiday_de"] = np.fromiter((d in de_hol for d in local.date),
                                       dtype=int, count=len(local))
    out["sin_hour"] = np.sin(2 * np.pi * local.hour / 24)
    out["cos_hour"] = np.cos(2 * np.pi * local.hour / 24)
    out["sin_doy"] = np.sin(2 * np.pi * local.dayofyear / 365.25)
    out["cos_doy"] = np.cos(2 * np.pi * local.dayofyear / 365.25)
    return out


def load_load_forecast() -> pd.Series:
    """Day-ahead total load forecast (MW); pre-gate (<=10:00 CET D-1) -> used directly.
    Index must be contiguous; NaN VALUES are allowed here and handled downstream."""
    s = pd.read_parquet(RAW / "load_forecast_da.parquet")["Forecasted Load"].sort_index()
    _require_contiguous_hourly(s.index, "load_forecast_da")
    return s


def load_res_forecast() -> pd.DataFrame:
    """Day-ahead wind/solar forecast (MW). NOT gate-safe for day D (published
    <=18:00 D-1); the D-1 vintage is applied downstream. NaN VALUES handled there."""
    df = pd.read_parquet(RAW / "wind_solar_forecast_da.parquet").sort_index()
    _require_contiguous_hourly(df.index, "wind_solar_forecast_da")
    return df


def build_fundamental_features(load_fc: pd.Series, res_fc: pd.DataFrame) -> pd.DataFrame:
    """Load forecast (D, direct) + RES forecast (D-1 vintage) + residual load.

    residual_load = load_fc(D) - sum(RES D-1 vintage): leakage-safe but an
    APPROXIMATION of true day-D residual load (RES-for-D is not pre-gate).
    NaN in any component propagates (handled explicitly downstream).
    """
    res_d1 = res_fc.shift(24).rename(columns=RES_RENAME)   # (D,h) <- RES forecast for (D-1,h)
    out = pd.concat([load_fc.rename("load_fc"), res_d1], axis=1, sort=False)
    out["residual_load"] = out["load_fc"] - (
        out["solar_d1"] + out["wind_offshore_d1"] + out["wind_onshore_d1"]
    )
    return out.sort_index()


def build_dataset() -> pd.DataFrame:
    """Assemble target + all feature blocks on the STRUCTURAL common range.

    The range start/end are derived from warm-up (shift/rolling) and the
    non-overlapping series tails only — NOT from NaN values. The returned frame
    is a contiguous hourly grid; rows with genuinely-missing raw values still
    carry NaN and are dropped EXPLICITLY (with a manifest) by main().
    """
    price = load_price()
    price_feats = build_price_features(price)
    cal = build_calendar_features(price.index)
    load_fc = load_load_forecast()
    res = load_res_forecast()
    fund = build_fundamental_features(load_fc, res)

    full = pd.concat([price.rename("da_price"), price_feats, cal, fund], axis=1, sort=False)

    start = max(price.index[0] + MAX_LOOKBACK,          # price-feature warm-up
                load_fc.index[0],                       # load coverage start
                res.index[0] + pd.Timedelta("24h"))     # RES shift-24 warm-up
    end = min(price.index[-1], load_fc.index[-1], res.index[-1])
    grid = pd.date_range(start, end, freq="1h")
    return full.reindex(grid)


def assign_splits(index: pd.DatetimeIndex) -> pd.Series:
    """Temporal split with an embargo (>= MAX_LOOKBACK) at the start of val/test."""
    split = pd.Series("train", index=index, dtype=object)
    split[(index >= SPLIT_B1) & (index < SPLIT_B2)] = "val"
    split[index >= SPLIT_B2] = "test"
    emb = (((index >= SPLIT_B1) & (index < SPLIT_B1 + EMBARGO))
           | ((index >= SPLIT_B2) & (index < SPLIT_B2 + EMBARGO)))
    split[emb] = "embargo"
    return split


def forecast_origin_utc(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """12:00 CET on D-1 (the gate) for each delivery hour, as a UTC-naive stamp."""
    local = index.tz_localize("UTC").tz_convert(LOCAL_TZ)
    origin_local = local.normalize() - pd.Timedelta("1D") + pd.Timedelta("12h")
    return pd.DatetimeIndex(origin_local).tz_convert("UTC").tz_localize(None)


def write_leakage_audit(ds: pd.DataFrame, n_dropped: int) -> None:
    """Write docs/phase2_leakage_audit.md with per-feature timing + split info."""
    def span(name):
        sub = ds.index[ds["split"] == name]
        return f"{len(sub):>6d} rows  [{sub.min()} .. {sub.max()}]" if len(sub) else "(none)"
    lookback_h = int(MAX_LOOKBACK / pd.Timedelta("1h"))
    embargo_h = int(EMBARGO / pd.Timedelta("1h"))
    md = f"""# Phase 2 — leakage audit (DE-LU day-ahead price dataset)

**Forecast origin (leakage anchor):** 12:00 CET on day D-1 (EPEX day-ahead gate
closure). Every hour of delivery day D is predicted from information knowable at
or before that instant. Each row stores this as `forecast_origin` (UTC).

## Feature as-of timing

| Feature | Source | As-of / lookback | Admissible because |
| --- | --- | --- | --- |
| `da_price` (target) | ENTSO-E DA price | delivery hour t (label) | — |
| `price_lag_24h/48h/72h/168h` | ENTSO-E DA price | t-24/48/72/168h | >=24h multiples: each lagged hour cleared by D-2 noon, known at the D-1 gate |
| `price_roll7d_{{mean,std,min,max}}` | ENTSO-E DA price | window price[t-{lookback_h}h .. t-24h] | `shift(24).rolling(168)` ends at the D-1 equivalent hour; never sees day-D prices |
| `load_fc` | ENTSO-E DA load forecast [6.1.B] | delivery hour t (day D) | Reg. 543/2013 Art. 6(2)(b): published <=2h before gate (<=10:00 CET D-1) -> pre-gate |
| `solar_d1`, `wind_offshore_d1`, `wind_onshore_d1` | ENTSO-E DA wind/solar forecast [14.1.D] | forecast for (D-1,h) (RES shifted +24h) | Art. 14(2)(d): the day-D forecast is only guaranteed by 18:00 D-1 (post-gate), so the D-1 vintage (public <=18:00 D-2) is used instead |
| `residual_load` | engineered | load_fc(D) - sum(RES D-1 vintage) | see approximation note below |
| calendar (`hour`,`dayofweek`,`month`,`is_weekend`,`is_holiday_de`,`sin/cos_hour`,`sin/cos_doy`) | deterministic | delivery hour t (local time) | fully determined by the calendar; nationwide DE holidays; no data lookback |

### residual_load — vintage-mixing approximation (explicit)

`residual_load = load_fc(D) - (solar+wind_offshore+wind_onshore)(D-1 vintage)`.
It mixes a **day-D** load forecast with a **D-1-vintage** RES forecast, because
the day-D RES forecast is not guaranteed public before the noon gate. It is
therefore **leakage-safe by construction but an approximation** of true day-D
residual load, not the exact quantity. Documented deliberately.

## Row handling (warm-up vs genuinely-missing data)

- **Warm-up:** the first {lookback_h}h (168h rolling window + 24h shift) plus the
  24h RES shift cannot be computed; removed by the structural common-range trim.
- **Genuinely-missing raw values:** {n_dropped} rows inside the common range had
  NaN feature values from ENTSO-E forecasts that were never published (load:
  2018 startup + a 2022 outage + 2 singletons; RES: 3 DST fall-back hours).
  Confirmed unrecoverable by a targeted re-pull. These rows are dropped
  **explicitly** with a manifest at `data/processed/_dropped_nan_rows.csv`
  (timestamp + which columns were NaN). No interpolation, no fill.

## Temporal split + embargo

Chronological split (no shuffle — autocorrelation + non-stationarity make a
random split leak and misestimate deployment performance):

- **train**: {span('train')}
- **val**:   {span('val')}
- **test**:  {span('test')}

**Embargo = {embargo_h}h (8 days)** at the start of val and test. It must be
>= MAX_LOOKBACK = {lookback_h}h; a 7-day (168h) embargo would be 23h short
because the rolling stat reaches {lookback_h}h back. With it, no retained
val/test row's feature window reaches a retained row of the prior split.
"""
    AUDIT_DOC.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_DOC.write_text(md)


def main() -> None:
    ds = build_dataset()

    # contiguity guaranteed by reindex; handle genuinely-missing rows EXPLICITLY
    nan_mask = ds.isna().any(axis=1)
    n_dropped = int(nan_mask.sum())
    if n_dropped:
        nan_rows = ds[nan_mask]
        manifest = pd.DataFrame(
            {"nan_columns": nan_rows.isna().apply(lambda r: ",".join(r.index[r]), axis=1)}
        )
        PROCESSED.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(MANIFEST)
        print(f"[missing-data] dropping {n_dropped} rows with NaN feature values "
              f"(genuinely unpublished by ENTSO-E; manifest -> {MANIFEST}). By year:")
        print(nan_rows.index.to_series().dt.year.value_counts().sort_index().to_string())
        ds = ds.loc[~nan_mask]
    if int(ds.isna().sum().sum()) != 0:
        raise SystemExit("FATAL: NaN remain after explicit drop.")

    split = assign_splits(ds.index)
    ds = ds.loc[split != "embargo"].copy()
    ds["forecast_origin"] = forecast_origin_utc(ds.index)
    ds["split"] = split.loc[ds.index].astype(str)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(DATASET)
    write_leakage_audit(ds, n_dropped)

    print("saved:", DATASET, "shape", ds.shape)
    print("cols:", list(ds.columns))
    print(ds["split"].value_counts().reindex(["train", "val", "test"]).to_string())
    print("audit doc:", AUDIT_DOC)


if __name__ == "__main__":
    main()
