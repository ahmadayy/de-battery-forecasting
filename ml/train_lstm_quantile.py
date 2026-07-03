#!/usr/bin/env python
"""
Phase 3, steps 2-4 — quantile LSTM (10th/50th/90th) for DE-LU day-ahead prices.

Minimal diff from ml/train_lstm.py: REUSES the same 168h leakage-safe per-split
batching, 2-layer / hidden-64 LSTM body, and train-only scalers. Only two things
change:
  * output head: Linear(64, 1) -> Linear(64, 3)   (one output per quantile)
  * loss: MSE -> pinball loss over [0.1, 0.5, 0.9]  (verified: 0 at exact match,
    asymmetric otherwise)
Median (q=0.5) is the point forecast. Reports val MAE/RMSE (median) vs the
baselines and the empirical coverage of the [q10, q90] band (calibration;
target ~80%). Monotonicity across quantiles is NOT enforced — the crossing rate
is reported instead, so any calibration issue is diagnosed honestly (step 5),
not hidden.

Model selection: the checkpoint with the BEST val pinball loss is saved and used
for all reported metrics (early-stopping checkpoint), NOT the final epoch. The
selected epoch is recorded in results/phase3/quantile_val_metrics.json.

Multi-minute on CPU — run manually:
    mamba run -n energy-ml python ml/train_lstm_quantile.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_lstm as base  # reuse constants + sequence/batching helpers

QUANTILES = [0.1, 0.5, 0.9]
MEDIAN_IDX = 1
MODEL_PATH = base.PROJECT_ROOT / "data" / "processed" / "lstm_quantile_best.pt"
METRICS_JSON = base.OUT / "quantile_val_metrics.json"


def pinball_loss(pred, target, quantiles):
    """pred: [B, Q], target: [B].  L_q = max(q*e, (q-1)*e), e = y - yhat."""
    e = target.unsqueeze(1) - pred
    q = torch.tensor(quantiles, dtype=pred.dtype)
    return torch.maximum(q * e, (q - 1.0) * e).mean()


class QuantileLSTM(nn.Module):
    def __init__(self, n_features: int, n_quantiles: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, base.HIDDEN, num_layers=base.LAYERS,
                            batch_first=True, dropout=base.DROPOUT)
        self.head = nn.Linear(base.HIDDEN, n_quantiles)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])  # [B, Q]


def main() -> None:
    torch.manual_seed(base.SEED)
    np.random.seed(base.SEED)
    ds = base.load_dataset()
    feats = base.feature_cols(ds)
    tr = ds[ds["split"] == "train"]
    va = ds[ds["split"] == "val"]

    fsc = StandardScaler().fit(tr[feats])           # scalers fit on TRAIN only
    tsc = StandardScaler().fit(tr[[base.TARGET]])

    def scaled(frame):
        g = frame.copy()
        g[feats] = fsc.transform(frame[feats])
        g[base.TARGET] = tsc.transform(frame[[base.TARGET]])
        return g

    Xtr, ytr, _ = base.build_sequences(scaled(tr), feats, base.SEQ_LEN)
    Xva, yva, _ = base.build_sequences(scaled(va), feats, base.SEQ_LEN)
    print(f"train seq={len(Xtr)}  val seq={len(Xva)}  quantiles={QUANTILES}")

    tr_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                           batch_size=base.BATCH, shuffle=True)
    va_loader = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
                           batch_size=base.BATCH)

    model = QuantileLSTM(len(feats), len(QUANTILES))
    opt = torch.optim.Adam(model.parameters(), lr=base.LR)

    base.OUT.mkdir(parents=True, exist_ok=True)
    hist, best, bad, best_epoch = [], np.inf, 0, -1
    for ep in range(base.EPOCHS):
        model.train()
        tl = n = 0
        for xb, yb in tr_loader:
            opt.zero_grad()
            loss = pinball_loss(model(xb), yb, QUANTILES)
            loss.backward()
            opt.step()
            tl += loss.item() * len(xb); n += len(xb)
        tl /= n
        model.eval()
        vl = m = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                vl += pinball_loss(model(xb), yb, QUANTILES).item() * len(xb); m += len(xb)
        vl /= m
        hist.append((ep, tl, vl))
        print(f"epoch {ep:02d}  train_pinball={tl:.4f}  val_pinball={vl:.4f}")
        if vl < best:
            best, bad, best_epoch = vl, 0, ep
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            bad += 1
            if bad >= base.PATIENCE:
                print(f"early stop at epoch {ep}")
                break

    h = pd.DataFrame(hist, columns=["epoch", "train_pinball", "val_pinball"])
    h.to_csv(base.OUT / "quantile_loss.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(h.epoch, h.train_pinball, label="train")
    plt.plot(h.epoch, h.val_pinball, label="val")
    plt.axvline(best_epoch, ls="--", c="grey", label=f"best (epoch {best_epoch})")
    plt.xlabel("epoch"); plt.ylabel("pinball loss (scaled)"); plt.legend(); plt.title("Quantile LSTM loss")
    plt.savefig(base.OUT / "quantile_loss.png", dpi=120)

    # ---- val evaluation from the BEST-val checkpoint (NOT the final epoch) ----
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    with torch.no_grad():
        pv = np.concatenate([model(xb).numpy() for xb, _ in va_loader])  # [N, Q] scaled
    pv_eur = np.column_stack([tsc.inverse_transform(pv[:, i:i + 1]).ravel() for i in range(len(QUANTILES))])
    yv_eur = tsc.inverse_transform(yva.reshape(-1, 1)).ravel()
    q10, q50, q90 = pv_eur[:, 0], pv_eur[:, 1], pv_eur[:, 2]

    mae_med = base.mae(yv_eur, q50)
    rmse_med = base.rmse(yv_eur, q50)
    coverage = float(np.mean((yv_eur >= q10) & (yv_eur <= q90))) * 100.0
    crossing = float(np.mean((q10 > q50) | (q50 > q90))) * 100.0

    b = json.loads(base.BASELINES_JSON.read_text())
    print(f"\nreported metrics are from the BEST-val checkpoint (epoch {best_epoch}), not the final epoch")
    print(f"quantile-LSTM val (median): MAE={mae_med:.3f}  RMSE={rmse_med:.3f}  EUR/MWh")
    print(f"baselines val:  persistence MAE={b['persistence']['mae']:.3f}  |  "
          f"LightGBM MAE={b['lightgbm']['mae']:.3f} RMSE={b['lightgbm']['rmse']:.3f}")
    print(f"80% interval [q10,q90] empirical coverage on val = {coverage:.1f}%  (target ~80%)")
    print(f"quantile crossing rate (q10>q50 or q50>q90) = {crossing:.2f}%  (monotonicity NOT enforced)")

    METRICS_JSON.write_text(json.dumps(
        {"eval_split": "val", "model": "quantile_lstm",
         "checkpoint_selection": "best val_pinball (early-stopping checkpoint), not final epoch",
         "checkpoint_epoch": int(best_epoch), "best_val_pinball": float(best), "epochs_run": len(hist),
         "quantiles": QUANTILES, "median_mae": mae_med, "median_rmse": rmse_med,
         "coverage_10_90_pct": coverage, "crossing_pct": crossing}, indent=2))
    print("saved ->", METRICS_JSON)


if __name__ == "__main__":
    main()
