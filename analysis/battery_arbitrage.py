#!/usr/bin/env python
"""
Phase 4 — battery day-ahead arbitrage under three price signals (approved
formulation). Quantifies the EUR/MW/yr value of forecast quality.

Battery (sourced; see Phase-4 formulation): P = 1 MW, E = 2 MWh usable (2h,
new-build German standard per Battery-Charts/RWTH), RTE = 85% (NREL ATB 2024)
split as eta_c = eta_d = sqrt(0.85); SoC in [0, E]; daily boundary
S_start = S_end = E/2. Degradation, grid fees, intraday recourse: excluded
(documented assumptions).

Per local delivery day D (Europe/Brussels; 23/24/25 hours, DST-aware), MILP:
    max  sum_t dt * price_dec[t] * (d_t - c_t)
    s.t. s_t = s_{t-1} + dt*(eta*c_t - d_t/eta),  s_{-1} := E/2,  s_last = E/2
         0 <= s_t <= E ; 0 <= c_t <= P*u_t ; 0 <= d_t <= P*(1-u_t) ; u_t binary
Binary u_t forbids simultaneous charge+discharge, which the LP would exploit
at negative prices (465 negative-price hours in the window; min -499 EUR/MWh).

Scenarios (identical structure; only the DECISION price differs):
    perfect      decision = realized da_price   (upper bound, this convention)
    persistence  decision = price_lag_24h       (naive floor, gate-safe)
    lstm_q50     decision = frozen quantile-LSTM median (the model)
Realized revenue always settles the committed schedule at da_price.

Fail-loudly: every daily MILP must end GRB.OPTIMAL or the run aborts; for the
perfect scenario, realized revenue must equal the objective to 1e-6.

Runs INSIDE the pypsa-eur pixi env (gurobipy 13):
    cd pypsa-eur && pixi run python ../analysis/battery_arbitrage.py
Reads  data/processed/test_forecasts.csv   (written by ml/export_test_forecasts.py)
Writes results/phase4/{arbitrage_summary.json, daily_revenues.csv,
                       hourly_schedules.csv, sample_week.csv, sample_week.png}
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "processed" / "test_forecasts.csv"
OUTDIR = ROOT / "results" / "phase4"

P_MW = 1.0
E_MWH = 2.0
RTE = 0.85
ETA = RTE ** 0.5          # symmetric split: eta_c = eta_d = sqrt(RTE)
S_REF = E_MWH / 2.0
DT = 1.0                  # hourly products
LOCAL_TZ = "Europe/Brussels"
EXPECTED_DAYS = 340       # measured in the formulation step; misalignment = abort
EXPECTED_HOURS = 8159
SCENARIOS = {"perfect": "da_price", "persistence": "persistence", "lstm_q50": "q50"}


def expected_hours(d) -> int:
    t0 = pd.Timestamp(d).tz_localize(LOCAL_TZ)
    t1 = (pd.Timestamp(d) + pd.Timedelta("1D")).tz_localize(LOCAL_TZ)
    return int((t1 - t0) / pd.Timedelta("1h"))


def solve_day(env: gp.Env, price_dec: np.ndarray):
    """Solve one delivery-day MILP; return (charge, discharge, soc, objective)."""
    n = len(price_dec)
    m = gp.Model(env=env)
    m.Params.MIPGap = 0.0
    c = m.addVars(n, lb=0.0, ub=P_MW)
    d = m.addVars(n, lb=0.0, ub=P_MW)
    s = m.addVars(n, lb=0.0, ub=E_MWH)
    u = m.addVars(n, vtype=GRB.BINARY)
    for t in range(n):
        prev = S_REF if t == 0 else s[t - 1]
        m.addConstr(s[t] == prev + DT * (ETA * c[t] - d[t] / ETA))
        m.addConstr(c[t] <= P_MW * u[t])
        m.addConstr(d[t] <= P_MW * (1 - u[t]))
    m.addConstr(s[n - 1] == S_REF)
    m.setObjective(gp.quicksum(DT * price_dec[t] * (d[t] - c[t]) for t in range(n)),
                   GRB.MAXIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise SystemExit(f"FATAL: daily MILP ended with status {m.Status} != OPTIMAL({GRB.OPTIMAL})")
    return (np.array([c[t].X for t in range(n)]),
            np.array([d[t].X for t in range(n)]),
            np.array([s[t].X for t in range(n)]),
            m.ObjVal)


def main() -> None:
    df = pd.read_csv(CSV, index_col=0, parse_dates=True)
    if df.isna().any().any():
        raise SystemExit("FATAL: NaN in test_forecasts.csv")
    local = df.index.tz_localize("UTC").tz_convert(LOCAL_TZ)
    df["local_date"] = local.date

    # complete local delivery days only (all 23/24/25 hours present)
    sizes = df.groupby("local_date").size()
    days = [d for d, n in sizes.items() if n == expected_hours(d)]
    n_hours = int(sizes.loc[days].sum())
    print(f"evaluable local days: {len(days)}  hours: {n_hours}")
    if len(days) != EXPECTED_DAYS or n_hours != EXPECTED_HOURS:
        raise SystemExit(f"FATAL: expected {EXPECTED_DAYS} days / {EXPECTED_HOURS} h, "
                         f"got {len(days)} / {n_hours} — input misaligned.")

    env = gp.Env(params={"OutputFlag": 0, "LogToConsole": 0})   # one license session
    daily_rows, sched_rows = [], []
    for scen, col in SCENARIOS.items():
        n_opt = 0
        for day in days:
            g = df[df["local_date"] == day]
            c, d, s, obj = solve_day(env, g[col].to_numpy())
            realized = float(np.sum(DT * g["da_price"].to_numpy() * (d - c)))
            if scen == "perfect":
                assert abs(realized - obj) <= 1e-6 * max(1.0, abs(obj)), \
                    f"FATAL: perfect-foresight realized != objective on {day}"
            n_opt += 1
            daily_rows.append({"date": str(day), "scenario": scen,
                               "expected_rev_eur": obj, "realized_rev_eur": realized,
                               "cycles": float(np.sum(d) * DT / E_MWH)})
            sched_rows.append(pd.DataFrame({
                "scenario": scen, "charge_mw": c, "discharge_mw": d, "soc_mwh": s,
                "price_real": g["da_price"].to_numpy(),
                "price_decision": g[col].to_numpy()}, index=g.index))
        print(f"scenario {scen:12s}: {n_opt}/{len(days)} daily MILPs solved to OPTIMAL")

    daily = pd.DataFrame(daily_rows)
    sched = pd.concat(sched_rows).rename_axis("utc_hour")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUTDIR / "daily_revenues.csv", index=False)
    sched.to_csv(OUTDIR / "hourly_schedules.csv")

    # ---- headline numbers ----
    ann = 8760.0 / n_hours
    summary = {"battery": {"P_MW": P_MW, "E_MWh": E_MWH, "duration_h": E_MWH / P_MW,
                           "RTE": RTE, "eta_each_leg": ETA,
                           "soc_boundary_MWh": S_REF, "products": "hourly DA"},
               "window": {"days": len(days), "hours": n_hours,
                          "first_day": str(days[0]), "last_day": str(days[-1])},
               "solve": {"status": f"all {len(days)}x{len(SCENARIOS)} daily MILPs OPTIMAL",
                         "mipgap": 0.0},
               "scenarios": {}}
    for scen in SCENARIOS:
        sub = daily[daily["scenario"] == scen]
        tot = float(sub["realized_rev_eur"].sum())
        exp = float(sub["expected_rev_eur"].sum())
        summary["scenarios"][scen] = {
            "realized_eur_total": tot,
            "realized_eur_per_mw_yr": tot / P_MW * ann,
            "expected_eur_total_at_decision_prices": exp,
            "avg_cycles_per_day": float(sub["cycles"].mean()),
        }
    perf = summary["scenarios"]["perfect"]["realized_eur_per_mw_yr"]
    pers = summary["scenarios"]["persistence"]["realized_eur_per_mw_yr"]
    lstm = summary["scenarios"]["lstm_q50"]["realized_eur_per_mw_yr"]
    summary["headline"] = {
        "value_of_forecast_quality_eur_per_mw_yr": lstm - pers,
        "gap_to_perfect_eur_per_mw_yr": perf - lstm,
        "pct_of_perfect": {"persistence": 100 * pers / perf, "lstm_q50": 100 * lstm / perf},
    }
    (OUTDIR / "arbitrage_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n================ PHASE 4 RESULTS (realized at actual DA prices) ================")
    for scen in SCENARIOS:
        s_ = summary["scenarios"][scen]
        print(f"  {scen:12s} total {s_['realized_eur_total']:>10.0f} EUR   "
              f"{s_['realized_eur_per_mw_yr']:>10.0f} EUR/MW/yr   "
              f"({100 * s_['realized_eur_per_mw_yr'] / perf:5.1f}% of perfect)   "
              f"avg {s_['avg_cycles_per_day']:.2f} cycles/day")
    print(f"\n  VALUE OF FORECAST QUALITY (lstm_q50 - persistence): "
          f"{summary['headline']['value_of_forecast_quality_eur_per_mw_yr']:+.0f} EUR/MW/yr")
    print(f"  GAP TO PERFECT (perfect - lstm_q50):                "
          f"{summary['headline']['gap_to_perfect_eur_per_mw_yr']:+.0f} EUR/MW/yr")

    # physical sanity: realized-price averages weighted by charge vs discharge
    print("\n---- physical sanity (charge low / discharge high, at REAL prices) ----")
    for scen in SCENARIOS:
        g = sched[sched["scenario"] == scen]
        pc = float(np.average(g["price_real"], weights=np.maximum(g["charge_mw"], 1e-12)))
        pd_ = float(np.average(g["price_real"], weights=np.maximum(g["discharge_mw"], 1e-12)))
        neg = int(((g["charge_mw"] > 1e-6) & (g["price_real"] < 0)).sum())
        print(f"  {scen:12s} charge-weighted avg price {pc:7.2f}  |  "
              f"discharge-weighted {pd_:7.2f} EUR/MWh  |  charging in {neg} negative-price hours")

    # ---- sample week: consecutive 7 evaluated days with max perfect revenue ----
    pday = (daily[daily["scenario"] == "perfect"]
            .set_index("date")["realized_rev_eur"])
    dts = [pd.Timestamp(x) for x in pday.index]
    best, best_i = -np.inf, 0
    for i in range(len(dts) - 6):
        if (dts[i + 6] - dts[i]).days == 6:
            w = pday.iloc[i:i + 7].sum()
            if w > best:
                best, best_i = w, i
    week_days = set(str(pd.Timestamp(x).date()) for x in dts[best_i:best_i + 7])
    wk = sched[[str(d) in week_days for d in
                sched.index.tz_localize("UTC").tz_convert(LOCAL_TZ).date]]
    wk.to_csv(OUTDIR / "sample_week.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    wp = wk[wk["scenario"] == "perfect"]
    ax[0].plot(wp.index, wp["price_real"], lw=1.2, color="black")
    ax[0].axhline(0, lw=0.5, color="grey")
    ax[0].set_ylabel("real DA price\n[EUR/MWh]")
    ax[0].set_title(f"Sample week {min(week_days)} .. {max(week_days)} "
                    f"(max perfect-foresight revenue week)")
    for a, scen in zip(ax[1:], SCENARIOS):
        g = wk[wk["scenario"] == scen]
        a.step(g.index, g["discharge_mw"] - g["charge_mw"], where="post", lw=1.0)
        a.axhline(0, lw=0.5, color="grey")
        a.set_ylabel(f"{scen}\nnet MW")
    ax[-1].set_xlabel("UTC hour")
    fig.tight_layout()
    fig.savefig(OUTDIR / "sample_week.png", dpi=120)
    print(f"\nwrote {OUTDIR}/arbitrage_summary.json, daily_revenues.csv, "
          f"hourly_schedules.csv, sample_week.csv, sample_week.png")


if __name__ == "__main__":
    main()
