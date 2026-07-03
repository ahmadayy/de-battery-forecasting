#!/usr/bin/env python
"""
Phase 4 — export frozen test-period forecasts for the battery arbitrage
optimizer. This is the FILE-BASED INTERFACE between the ml/ codebase (mamba,
this script) and the optimizer (pixi env, analysis/battery_arbitrage.py).

INFERENCE ONLY from frozen Phase-3 artifacts — no retraining, no tuning, no new
model decisions. Per LSTM-predictable test hour (UTC), writes
data/processed/test_forecasts.csv with:
    da_price     realized DA price (EUR/MWh)            [settlement prices]
    persistence  price_lag_24h (D-1 actuals, gate-safe) [scenario (b) signal]
    q10,q50,q90  frozen quantile-LSTM epoch-4 outputs   [scenario (c) signal]

Integrity contract: before writing, asserts the exported q50 and persistence
reproduce the COMMITTED Phase-3 test metrics (results/phase3/test_metrics.json)
to 1e-6, and that sequence targets align with the dataset prices. Fails loudly
on any mismatch.

Run: mamba run -n energy-ml python ml/export_test_forecasts.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_lstm as base
import train_lstm_quantile as Q

OUT_CSV = base.PROJECT_ROOT / "data" / "processed" / "test_forecasts.csv"


def main() -> None:
    ds = base.load_dataset()
    feats = base.feature_cols(ds)
    tr = ds[ds["split"] == "train"]
    te = ds[ds["split"] == "test"]

    fsc = StandardScaler().fit(tr[feats])           # train-only scalers, as in Phase 3
    tsc = StandardScaler().fit(tr[[base.TARGET]])
    te_s = te.copy()
    te_s[feats] = fsc.transform(te[feats])
    te_s[base.TARGET] = tsc.transform(te[[base.TARGET]])

    Xte, yte_s, end_times = base.build_sequences(te_s, feats, base.SEQ_LEN)
    common = pd.DatetimeIndex(end_times)

    model = Q.QuantileLSTM(len(feats), len(Q.QUANTILES))
    model.load_state_dict(torch.load(Q.MODEL_PATH))
    model.eval()
    with torch.no_grad():
        pv = model(torch.from_numpy(Xte)).numpy()
    pv_eur = np.column_stack([tsc.inverse_transform(pv[:, i:i + 1]).ravel() for i in range(3)])

    out = pd.DataFrame({
        "da_price": te.loc[common, base.TARGET].to_numpy(),
        "persistence": te.loc[common, "price_lag_24h"].to_numpy(),
        "q10": pv_eur[:, 0], "q50": pv_eur[:, 1], "q90": pv_eur[:, 2],
    }, index=common)
    out.index.name = "utc_hour"

    # ---- integrity asserts (fail loudly, write nothing on mismatch) ----
    assert np.allclose(tsc.inverse_transform(yte_s.reshape(-1, 1)).ravel(),
                       out["da_price"].to_numpy(), atol=1e-3), \
        "FATAL: sequence targets misaligned with dataset prices"
    ref = json.loads((base.OUT / "test_metrics.json").read_text())
    assert len(out) == ref["test_rows_scored"], \
        f"FATAL: exported {len(out)} rows != committed {ref['test_rows_scored']}"
    mae_q50 = float(np.mean(np.abs(out["da_price"] - out["q50"])))
    mae_pers = float(np.mean(np.abs(out["da_price"] - out["persistence"])))
    assert np.isclose(mae_q50, ref["quantile_lstm_median"]["mae"], atol=1e-6), \
        f"FATAL: exported q50 MAE {mae_q50} != committed {ref['quantile_lstm_median']['mae']}"
    assert np.isclose(mae_pers, ref["persistence"]["mae"], atol=1e-6), \
        f"FATAL: exported persistence MAE {mae_pers} != committed {ref['persistence']['mae']}"

    out.to_csv(OUT_CSV)
    print(f"wrote {OUT_CSV}  rows={len(out)}  span [{common.min()} .. {common.max()}] UTC")
    print(f"integrity: q50 MAE={mae_q50:.6f} and persistence MAE={mae_pers:.6f} "
          f"reproduce committed Phase-3 test metrics exactly ✓")


if __name__ == "__main__":
    main()
