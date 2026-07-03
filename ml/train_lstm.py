#!/usr/bin/env python
"""
Phase 3, steps 2-3 — LSTM forecaster for DE-LU day-ahead prices.

Seq-to-one LSTM: for target hour t, the input is the 22 engineered feature
vectors for the 168 consecutive hours ending at t; the output is the price at t.

Leakage-safe batching: sequences are built PER SPLIT from contiguous-hourly runs
only, so a 168h window never crosses a dropped-row hole, the embargo, or a split
boundary. Feature and target scalers are fit on TRAIN only.

Logs train/val loss per epoch (results/phase3/lstm_loss.csv + .png), saves the
best-val model, and reports val MAE/RMSE (EUR/MWh) against the baselines read
from results/phase3/baselines.json (never hardcoded).

Multi-minute on CPU — run manually:
    mamba run -n energy-ml python ml/train_lstm.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = PROJECT_ROOT / "data" / "processed" / "dataset.parquet"
OUT = PROJECT_ROOT / "results" / "phase3"
BASELINES_JSON = OUT / "baselines.json"
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "lstm_best.pt"
TARGET = "da_price"
NON_FEATURES = [TARGET, "forecast_origin", "split"]

SEQ_LEN = 168
HIDDEN = 64
LAYERS = 2
DROPOUT = 0.1
EPOCHS = 40
PATIENCE = 6
BATCH = 256
LR = 1e-3
SEED = 0


def load_dataset() -> pd.DataFrame:
    return pd.read_parquet(DATASET)


def feature_cols(ds: pd.DataFrame):
    return [c for c in ds.columns if c not in NON_FEATURES]


def _run_ids(index: pd.DatetimeIndex):
    """Integer run id per row; increments at any gap != 1h (hole/embargo/split)."""
    return (index.to_series().diff() != pd.Timedelta("1h")).cumsum().to_numpy()


def sequence_end_positions(index: pd.DatetimeIndex, seq_len: int):
    """Row positions that are valid sequence ends: the last of `seq_len`
    consecutive-hourly rows within a single contiguous run."""
    runs = _run_ids(index)
    ends = []
    for r in np.unique(runs):
        pos = np.where(runs == r)[0]
        if len(pos) >= seq_len:
            ends.extend(pos[seq_len - 1:])
    return np.array(sorted(ends), dtype=int)


def build_sequences(frame: pd.DataFrame, feat_cols, seq_len: int):
    """Return (X[N,seq_len,F] float32, y[N] float32, end_times) from contiguous runs."""
    from numpy.lib.stride_tricks import sliding_window_view
    frame = frame.sort_index()
    f = frame[feat_cols].to_numpy(np.float32)
    y = frame[TARGET].to_numpy(np.float32)
    runs = _run_ids(frame.index)
    Xs, ys, tt = [], [], []
    for r in np.unique(runs):
        pos = np.where(runs == r)[0]
        if len(pos) < seq_len:
            continue
        w = sliding_window_view(f[pos], seq_len, axis=0).transpose(0, 2, 1)  # (n-L+1, L, F)
        Xs.append(w)
        ys.append(y[pos][seq_len - 1:])
        tt.append(frame.index[pos][seq_len - 1:])
    X = np.concatenate(Xs).astype(np.float32)
    yv = np.concatenate(ys).astype(np.float32)
    times = tt[0]
    for extra in tt[1:]:
        times = times.append(extra)
    return X, yv, times


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, HIDDEN, num_layers=LAYERS,
                            batch_first=True, dropout=DROPOUT)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _print_baselines():
    if BASELINES_JSON.exists():
        b = json.loads(BASELINES_JSON.read_text())
        print(f"baselines ({b.get('eval_split','val')}):  "
              f"persistence MAE={b['persistence']['mae']:.3f} RMSE={b['persistence']['rmse']:.3f}  |  "
              f"LightGBM MAE={b['lightgbm']['mae']:.3f} RMSE={b['lightgbm']['rmse']:.3f}")
    else:
        print("baselines.json not found — run ml/train_baselines.py first for the comparison.")


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ds = load_dataset()
    feats = feature_cols(ds)
    tr = ds[ds["split"] == "train"]
    va = ds[ds["split"] == "val"]

    fsc = StandardScaler().fit(tr[feats])           # scalers fit on TRAIN only
    tsc = StandardScaler().fit(tr[[TARGET]])

    def scaled(frame):
        g = frame.copy()
        g[feats] = fsc.transform(frame[feats])
        g[TARGET] = tsc.transform(frame[[TARGET]])
        return g

    Xtr, ytr, _ = build_sequences(scaled(tr), feats, SEQ_LEN)
    Xva, yva, _ = build_sequences(scaled(va), feats, SEQ_LEN)
    print(f"train seq={len(Xtr)}  val seq={len(Xva)}  seq_shape={Xtr.shape[1:]}")

    tr_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                           batch_size=BATCH, shuffle=True)
    va_loader = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
                           batch_size=BATCH)

    model = LSTMRegressor(len(feats))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossfn = nn.MSELoss()

    OUT.mkdir(parents=True, exist_ok=True)
    hist, best, bad = [], np.inf, 0
    for ep in range(EPOCHS):
        model.train()
        tl = n = 0
        for xb, yb in tr_loader:
            opt.zero_grad()
            loss = lossfn(model(xb), yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(xb); n += len(xb)
        tl /= n
        model.eval()
        vl = m = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                vl += lossfn(model(xb), yb).item() * len(xb); m += len(xb)
        vl /= m
        hist.append((ep, tl, vl))
        print(f"epoch {ep:02d}  train_mse={tl:.4f}  val_mse={vl:.4f}")
        if vl < best:
            best, bad = vl, 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"early stop at epoch {ep}")
                break

    h = pd.DataFrame(hist, columns=["epoch", "train_mse", "val_mse"])
    h.to_csv(OUT / "lstm_loss.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(h.epoch, h.train_mse, label="train")
    plt.plot(h.epoch, h.val_mse, label="val")
    plt.xlabel("epoch"); plt.ylabel("MSE (scaled)"); plt.legend(); plt.title("LSTM loss")
    plt.savefig(OUT / "lstm_loss.png", dpi=120)

    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    with torch.no_grad():
        pv = np.concatenate([model(xb).numpy() for xb, _ in va_loader])
    pv_eur = tsc.inverse_transform(pv.reshape(-1, 1)).ravel()
    yv_eur = tsc.inverse_transform(yva.reshape(-1, 1)).ravel()
    print(f"\nLSTM val:   MAE={mae(yv_eur, pv_eur):.3f}  RMSE={rmse(yv_eur, pv_eur):.3f}  EUR/MWh")
    _print_baselines()


if __name__ == "__main__":
    main()
