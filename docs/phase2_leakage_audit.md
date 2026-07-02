# Phase 2 — leakage audit (DE-LU day-ahead price dataset)

**Forecast origin (leakage anchor):** 12:00 CET on day D-1 (EPEX day-ahead gate
closure). Every hour of delivery day D is predicted from information knowable at
or before that instant. Each row stores this as `forecast_origin` (UTC).

## Feature as-of timing

| Feature | Source | As-of / lookback | Admissible because |
| --- | --- | --- | --- |
| `da_price` (target) | ENTSO-E DA price | delivery hour t (label) | — |
| `price_lag_24h/48h/72h/168h` | ENTSO-E DA price | t-24/48/72/168h | >=24h multiples: each lagged hour cleared by D-2 noon, known at the D-1 gate |
| `price_roll7d_{mean,std,min,max}` | ENTSO-E DA price | window price[t-191h .. t-24h] | `shift(24).rolling(168)` ends at the D-1 equivalent hour; never sees day-D prices |
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

- **Warm-up:** the first 191h (168h rolling window + 24h shift) plus the
  24h RES shift cannot be computed; removed by the structural common-range trim.
- **Genuinely-missing raw values:** 773 rows inside the common range had
  NaN feature values from ENTSO-E forecasts that were never published (load:
  2018 startup + a 2022 outage + 2 singletons; RES: 3 DST fall-back hours).
  Confirmed unrecoverable by a targeted re-pull. These rows are dropped
  **explicitly** with a manifest at `data/processed/_dropped_nan_rows.csv`
  (timestamp + which columns were NaN). No interpolation, no fill.

## Temporal split + embargo

Chronological split (no shuffle — autocorrelation + non-stationarity make a
random split leak and misestimate deployment performance):

- **train**:  53855 rows  [2018-10-08 21:00:00 .. 2024-12-31 23:00:00]
- **val**:     4152 rows  [2025-01-09 00:00:00 .. 2025-06-30 23:00:00]
- **test**:    8541 rows  [2025-07-09 00:00:00 .. 2026-06-29 21:00:00]

**Embargo = 192h (8 days)** at the start of val and test. It must be
>= MAX_LOOKBACK = 191h; a 7-day (168h) embargo would be 23h short
because the rolling stat reaches 191h back. With it, no retained
val/test row's feature window reaches a retained row of the prior split.
