#!/usr/bin/env python
"""
Phase 3, step 1 — baselines for DE-LU day-ahead price forecasting.

Two baselines the DL model must beat, trained on the train split and evaluated
on the val split (test is NOT touched here):
  (a) naive persistence: predict = price_lag_24h (same hour yesterday)
  (b) a simple LightGBM on all engineered features (fixed config, no tuning)

Honest reporting only: MAE + RMSE (EUR/MWh) on val, persisted to
results/phase3/baselines.json as the single source of truth (so downstream
models compare against real numbers, never hardcoded ones).

Run: mamba run -n energy-ml python ml/train_baselines.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = PROJECT_ROOT / "data" / "processed" / "dataset.parquet"
OUT = PROJECT_ROOT / "results" / "phase3"
BASELINES_JSON = OUT / "baselines.json"
TARGET = "da_price"
NON_FEATURES = [TARGET, "forecast_origin", "split"]


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def main() -> None:
    ds = pd.read_parquet(DATASET)
    tr = ds[ds["split"] == "train"]
    va = ds[ds["split"] == "val"]
    feats = [c for c in ds.columns if c not in NON_FEATURES]
    print(f"train rows={len(tr)}  val rows={len(va)}  n_features={len(feats)}")

    y = va[TARGET].to_numpy()

    p_pers = va["price_lag_24h"].to_numpy()
    print(f"\n(a) persistence (price_lag_24h):  MAE={mae(y, p_pers):.3f}  RMSE={rmse(y, p_pers):.3f}  EUR/MWh")

    model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=-1, verbosity=-1,
    )
    model.fit(tr[feats], tr[TARGET])
    p_lgb = model.predict(va[feats])
    print(f"(b) LightGBM (all features):      MAE={mae(y, p_lgb):.3f}  RMSE={rmse(y, p_lgb):.3f}  EUR/MWh")

    print("\ntop-10 LightGBM feature importances:")
    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    print(imp.head(10).to_string())

    OUT.mkdir(parents=True, exist_ok=True)
    metrics = {
        "eval_split": "val",
        "persistence": {"mae": mae(y, p_pers), "rmse": rmse(y, p_pers)},
        "lightgbm": {"mae": mae(y, p_lgb), "rmse": rmse(y, p_lgb)},
    }
    BASELINES_JSON.write_text(json.dumps(metrics, indent=2))
    print("\nsaved baseline metrics ->", BASELINES_JSON)


if __name__ == "__main__":
    main()
