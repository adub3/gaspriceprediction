from __future__ import annotations

import argparse
import pandas as pd
import numpy as np

from tests.evaluation.backtest import walk_forward
from tests.evaluation.metrics import brier_score, logloss, rocauc, calibration_table
from models.linearmodel import LinearRWEventModel


def main() -> None:
    # ----------------------------
    # 1. Parse CLI arguments
    # ----------------------------
    ap = argparse.ArgumentParser(description="Evaluate LinearRWEventModel on CSV data")
    ap.add_argument("--csv", required=True, help="Path to CSV with [date, price] columns")
    ap.add_argument("--threshold", type=float, required=True, help="Event threshold (e.g., 3.00)")
    ap.add_argument("--window", type=int, default=24, help="Rolling window size (months or periods)")
    ap.add_argument("--date_col", type=str, default=None, help="Optional name of date column")
    ap.add_argument("--price_col", type=str, default=None, help="Optional name of price column")
    args = ap.parse_args()

    # ----------------------------
    # 2. Load and prepare data
    # ----------------------------
    df = pd.read_csv(args.csv)
    date_col = args.date_col or df.columns[0]
    price_col = args.price_col or df.columns[1]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, price_col]).sort_values(date_col).reset_index(drop=True)
    df = df.set_index(date_col)
    df["price"] = df[price_col].astype(float)
    dates = df.index

    # ----------------------------
    # 3. Define adapter functions
    # ----------------------------
    def get_train(start: pd.Timestamp, end: pd.Timestamp):
        mask = (dates >= start) & (dates < end)
        X_train = df.loc[mask, ["price"]]
        y_train = (df.loc[mask, "price"].shift(-4) > args.threshold).astype(int)
        return X_train, y_train

    def get_forecast_row(t: pd.Timestamp):
        return pd.DataFrame({"price": [df.loc[t, "price"]]}, index=[t])

    def get_outcome(t: pd.Timestamp):
        return int(df.loc[t, "price"] > args.threshold)

    # ----------------------------
    # 4. Rolling backtest
    # ----------------------------
    window_months = args.window
    forecast_dates = dates[window_months:]  # leave warm-up
    model_ctor = lambda: LinearRWEventModel(threshold=args.threshold)

    res = walk_forward(
        forecast_dates=forecast_dates,
        get_train=get_train,
        get_forecast_row=get_forecast_row,
        get_outcome=get_outcome,
        model_ctor=model_ctor,
        cfg={"window_months": window_months, "step_months": 1},
    )

    # ----------------------------
    # 5. Compute metrics
    # ----------------------------
    brier = brier_score(res.p_hat, res.y)
    ll = logloss(res.p_hat, res.y)
    auc = rocauc(res.p_hat, res.y)
    calib, ece = calibration_table(res.p_hat, res.y, bins=5)

    print("\n===== LINEAR RANDOM-WALK EVENT MODEL =====")
    print(f"Samples evaluated: {len(res)}")
    print(f"Brier Score: {brier:.6f}")
    print(f"LogLoss:     {ll:.6f}")
    print(f"ROC-AUC:     {auc:.6f}")
    print(f"ECE:         {ece:.6f}")

    print("\nCalibration bins:")
    print(calib.round(3))

    print("\nFirst few results:")
    print(res.head())

    # ----------------------------
    # 6. Save results (optional)
    # ----------------------------
    res.to_csv("linear_rw_eval_results.csv")
    print("\nSaved detailed results to linear_rw_eval_results.csv")


if __name__ == "__main__":
    main()
