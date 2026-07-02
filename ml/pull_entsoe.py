#!/usr/bin/env python
"""
Phase 2 — pull RAW ENTSO-E series for German (DE-LU) day-ahead price forecasting.

This is the DATA-PULL layer ONLY. It fetches the real ENTSO-E Transparency
series, stores them verbatim on a common hourly UTC grid, and reports coverage
and gaps. It performs NO feature engineering, NO leakage shifting, NO
interpolation, and NO fabrication. Feature construction (lags, calendar,
residual-load, and the wind/solar D-1-vintage leakage fallback) happens in a
SEPARATE later script, only after this raw coverage has been reviewed.

INTEGRITY CONTRACT (Phase 1 discipline, non-negotiable):
  * Real ENTSO-E data only — never mocked/synthetic.
  * Never interpolate or fabricate to fill gaps. Gaps are DETECTED, counted,
    written to a manifest CSV, and flagged LOUDLY for a human decision.
  * Fail loudly (non-zero exit) if the API errors hard or a whole series comes
    back empty. ENTSOE_API_TOKEN is read from env/.env and never printed.

VERIFIED LEAKAGE TIMING (Commission Regulation (EU) 543/2013; DE-LU day-ahead
auction gate closure = 12:00 CET on D-1):
  * Day-ahead TOTAL LOAD forecast [6.1.B] — Art. 6(2)(b): published no later
    than TWO HOURS BEFORE day-ahead gate closure  =>  <= 10:00 CET on D-1.
    GUARANTEED PRE-GATE  ->  usable as a forecast-for-D feature. (clean)
  * Day-ahead WIND & SOLAR forecast [14.1.D] — Art. 14(2)(d): published no
    later than 18:00 Brussels time on D-1  =>  6h AFTER the noon gate.
    NOT guaranteed pre-gate  ->  the feature step must use the D-1 VINTAGE
    (forecast targeting D-1, published <= 18:00 on D-2, which IS pre-gate).
    This script pulls the raw series at its native target-day timestamps; the
    leakage-safe +1-day shift is applied DOWNSTREAM, not here.

Bidding zone: DE-LU only, and only from 2018-10-01 (the DE-AT-LU split date).
Earlier dates belong to a structurally different zone and are refused.

RESOLUTION: EPEX day-ahead moved from hourly to 15-minute settlement partway
through 2025. Aggregation to the hourly grid is PERIOD-AWARE (per hour bin),
not a single global mode, so hourly and 15-min segments are both handled
correctly and the transition is reported. See aggregate_to_hourly().

Requires (energy-ml env): entsoe-py  (declared in ml/environment.yml).
Reads ENTSOE_API_TOKEN from the environment or <repo>/.env.

This does a multi-minute network pull — run it yourself and report the summary:
    mamba activate energy-ml
    python ml/pull_entsoe.py --start 2018-10-01 --end 2026-07-01
    python ml/pull_entsoe.py --series da_price      # re-pull one series only

NOTE: entsoe-py method signatures vary across versions. If a query errors on an
unexpected keyword, check the installed version's EntsoePandasClient API.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "entsoe"
ZONE = "DE_LU"
ZONE_TZ = "Europe/Brussels"   # zone-local; used only to define calendar-year chunks
DA_PROCESS = "A01"            # entsoe-py process_type A01 = "Day ahead"
ZONE_START = pd.Timestamp("2018-10-01", tz=ZONE_TZ)  # DE-AT-LU -> DE-LU split
CHUNK_SPAN = pd.Timedelta("90D")    # sub-year chunk span (< 1yr; see period_chunks)
CHUNK_OVERLAP = pd.Timedelta("2D")  # overlap adjacent chunks; pull_series dedups it


def die(msg: str) -> None:
    sys.exit(f"FATAL: {msg}")


def load_token() -> str:
    """Read ENTSOE_API_TOKEN from env or <repo>/.env. Never logged."""
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
        die("ENTSOE_API_TOKEN not found in environment or <repo>/.env.")
    return token  # never logged


def period_chunks(start: pd.Timestamp, end: pd.Timestamp):
    """Yield overlapping [cs, ce) spans of at most ~CHUNK_SPAN (well under a year).

    WHY sub-year, not yearly: entsoe-py's query_day_ahead_prices is @year_limited
    and a ~1-year day-ahead PRICE query omits the hour 24h BEFORE the query end
    (verified empirically: a 1-year query drops end-24h; a 3-month query is clean;
    @year_limited reproduces the omission at each internal 1-year block boundary).
    Increasing overlap does NOT fix it — it just relocates the dropped hour (a
    2-day overlap moved the hole from 30 Dec to 28 Dec). Keeping every query well
    under a year avoids the omission entirely; CHUNK_OVERLAP + dedup in
    pull_series() cover the (now sub-year) seams.
    """
    cur = start
    while cur < end:
        nxt = min(cur + CHUNK_SPAN, end)
        cs = max(start, cur - CHUNK_OVERLAP)
        ce = min(end, nxt + CHUNK_OVERLAP)
        yield cs, ce
        cur = nxt


def pull_series(label, fn, start, end):
    """Pull one series in overlapping sub-year chunks with retry.

    `fn(chunk_start, chunk_end)` -> pd.Series | pd.DataFrame (tz-aware index).
    Returns (concatenated_obj_or_None, failed_spans). A NoMatchingDataError for
    a chunk means ENTSO-E published nothing for it: recorded and skipped, not
    fabricated. Other exceptions are retried, then recorded as a failed span.
    """
    try:
        from entsoe.exceptions import NoMatchingDataError
    except Exception:                       # pragma: no cover - version guard
        NoMatchingDataError = ()            # type: ignore

    parts, failed = [], []
    for cs, ce in period_chunks(start, end):
        last_err = None
        for attempt in range(3):
            try:
                obj = fn(cs, ce)
                n = 0 if obj is None else len(obj)
                if n:
                    parts.append(obj)
                print(f"  [{label}] {cs.date()}..{ce.date()}  rows={n}")
                last_err = None
                break
            except NoMatchingDataError:
                print(f"  [{label}] {cs.date()}..{ce.date()}  NO DATA (ENTSO-E published none)")
                failed.append((str(cs.date()), str(ce.date()), "NoMatchingData"))
                last_err = None
                break
            except Exception as e:          # transient / rate-limit
                last_err = e
                print(f"  [{label}] {cs.date()}..{ce.date()}  attempt {attempt+1}/3 "
                      f"failed: {type(e).__name__}: {e}")
                time.sleep(5)
        if last_err is not None:
            failed.append((str(cs.date()), str(ce.date()),
                           f"{type(last_err).__name__}: {last_err}"))
    if not parts:
        return None, failed
    out = pd.concat(parts)
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out, failed


def _resolution_audit(idx, label) -> None:
    """Print native spacing per calendar year + any 1h<->15min transitions.

    Period-aware, NOT a single global mode: this is what catches a mid-series
    resolution change (e.g. EPEX day-ahead moving to 15-min settlement in 2025)
    that a global mode hides behind the hourly majority.
    """
    ser = pd.Series(idx)
    print(f"\n--- resolution audit [{label}] (native, pre-aggregation) ---")
    years = pd.Index(idx).year
    for y in sorted(set(years)):
        sub = pd.Series(idx[years == y])
        if len(sub) < 2:
            continue
        vc = sub.diff().dropna().value_counts()
        print(f"  {y}: " + ", ".join(f"{td}x{c}" for td, c in vc.items()))
    reg = ser.diff().map(lambda td: "1h" if td == pd.Timedelta("1h")
                         else ("15min" if td == pd.Timedelta("15min") else "other"))
    prev = reg.shift()
    trans = [(idx[i], prev.iloc[i], reg.iloc[i]) for i in range(1, len(reg))
             if reg.iloc[i] in ("1h", "15min") and prev.iloc[i] in ("1h", "15min")
             and reg.iloc[i] != prev.iloc[i]]
    if trans:
        print(f"  RESOLUTION TRANSITION(S) [{label}]:")
        for ts, a, b in trans:
            print(f"    {ts} UTC:  {a} -> {b}")
    else:
        print(f"  single resolution regime [{label}] — no 1h<->15min transition")


def aggregate_to_hourly(obj, label):
    """Collapse a UTC-naive series/frame onto a clean hourly grid, PERIOD-AWARE.

    Each hour bin is averaged over whatever native points it holds: 1 for an
    hourly hour (pass-through), 4 for a 15-min hour (mean of the quarter-hours).
    Hourly segments therefore survive unchanged and 15-min segments collapse to
    their hourly mean, with NO global-mode assumption and NO mixing across the
    transition (each hour is classified by its own contents). Fully-empty hours
    are dropped (kept absent), matching the missing-hour convention used by the
    other series. Irregular hour-bins (native count not 1 or 4) are flagged.
    """
    obj = obj.sort_index()
    obj = obj[~obj.index.duplicated(keep="first")]
    _resolution_audit(obj.index, label)

    counts = pd.Series(1, index=obj.index).groupby(obj.index.floor("h")).size()
    weird = counts[~counts.isin([1, 4])]
    if len(weird):
        print(f"  !! [{label}] {len(weird)} hour-bin(s) with irregular native point "
              f"count (not 1 or 4) — partial/mixed hour, averaged over what exists:")
        print(weird.head(10).to_string())

    hourly = obj.resample("1h").mean()
    hourly = hourly.dropna(how="all") if isinstance(hourly, pd.DataFrame) else hourly.dropna()
    return hourly


def to_hourly_utc(obj, label):
    """tz-aware index -> UTC-naive, then period-aware hourly aggregation."""
    obj = obj.copy()
    obj.index = pd.to_datetime(obj.index).tz_convert("UTC").tz_localize(None)
    return aggregate_to_hourly(obj, label)


def coverage_report(obj, label) -> int:
    """Print coverage; write missing-hour manifest if any. Returns #missing."""
    idx = obj.index
    first, last = idx.min(), idx.max()
    expected = pd.date_range(first, last, freq="1h")
    missing = expected.difference(idx)
    dup = int(idx.duplicated().sum())
    print(f"\n--- coverage [{label}] ---")
    print(f"  span:      {first} .. {last}  (UTC)")
    print(f"  rows:      {len(idx)}   expected contiguous hourly: {len(expected)}")
    print(f"  missing:   {len(missing)} hours   duplicates: {dup}")
    if len(missing):
        gaps_file = OUT_DIR / f"_gaps_{label}.csv"
        pd.Series(missing, name="missing_utc_hour").to_csv(gaps_file, index=False)
        print(f"  !! GAPS written to {gaps_file} — NOT filled. "
              f"Decide handling before feature engineering.")
    return len(missing)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pull raw ENTSO-E DE-LU series (Phase 2).")
    ap.add_argument("--start", default="2018-10-01",
                    help="inclusive start date (DE-LU zone exists since 2018-10-01)")
    ap.add_argument("--end", default="2026-07-01", help="exclusive end date")
    ap.add_argument("--series", nargs="+", default=None,
                    help="subset of series to pull (default: all); e.g. --series da_price")
    args = ap.parse_args()

    start = pd.Timestamp(args.start, tz=ZONE_TZ)
    end = pd.Timestamp(args.end, tz=ZONE_TZ)
    if start < ZONE_START:
        die("start < 2018-10-01 would mix the pre-split DE-AT-LU zone — refusing.")
    if end <= start:
        die("end must be after start.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    token = load_token()
    from entsoe import EntsoePandasClient
    client = EntsoePandasClient(api_key=token)

    print(f"Pulling ENTSO-E {ZONE}  {start.date()} .. {end.date()}  ->  {OUT_DIR}")

    specs = {
        "da_price": lambda cs, ce: client.query_day_ahead_prices(ZONE, start=cs, end=ce),
        "load_forecast_da": lambda cs, ce: client.query_load_forecast(
            ZONE, start=cs, end=ce, process_type=DA_PROCESS),
        "wind_solar_forecast_da": lambda cs, ce: client.query_wind_and_solar_forecast(
            ZONE, start=cs, end=ce, process_type=DA_PROCESS),
    }

    if args.series:
        unknown = [s for s in args.series if s not in specs]
        if unknown:
            die(f"unknown --series {unknown}; valid: {list(specs)}")
        specs = {k: v for k, v in specs.items() if k in args.series}
        print(f"(restricted to series: {list(specs)})")

    results, all_failed = {}, {}
    for label, fn in specs.items():
        print(f"\n=== {label} ===")
        obj, failed = pull_series(label, fn, start, end)
        if obj is None:
            die(f"series '{label}' returned NO data across the entire range — cannot proceed.")
        results[label] = to_hourly_utc(obj, label)
        all_failed[label] = failed

    print("\n=== saving raw parquet ===")
    for label, obj in results.items():
        df = obj.to_frame(name=label) if isinstance(obj, pd.Series) else obj
        path = OUT_DIR / f"{label}.parquet"
        df.to_parquet(path)
        print(f"  wrote {path}  shape={df.shape}")

    print("\n================ COVERAGE / INTEGRITY SUMMARY ================")
    total_missing = sum(coverage_report(obj, label) for label, obj in results.items())

    print("\n--- failed / no-data chunks ---")
    any_failed = False
    for label, failed in all_failed.items():
        for cs, ce, why in failed:
            any_failed = True
            print(f"  [{label}] {cs}..{ce}: {why}")
    if not any_failed:
        print("  none")

    print("\nDONE. Raw series saved. NO interpolation/fabrication performed.")
    if total_missing or any_failed:
        print("!! Gaps and/or failed chunks detected above — REVIEW before feature engineering.")
    else:
        print("Clean contiguous hourly coverage across all series.")


if __name__ == "__main__":
    main()
