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
    ap.add_argument("--window", type=int, default=24, help="Rolling window size (rows/periods)")
    ap.add_argument("--date_col", type=str, default=None, help="Optional name of date column")
    ap.add_argument("--price_col", type=str, default=None, help="Optional name of price column")
    ap.add_argument(
        "--horizon",
        type=int,
        default=4,
        help="Forecast horizon in rows/periods for the event label (default: 4)",
    )
    args = ap.parse_args()

    # ----------------------------
    # 2. Load and prepare data
    # ----------------------------
    df = pd.read_csv(args.csv)
    date_col = args.date_col or df.columns[0]
    price_col = args.price_col or df.columns[1]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = (
        df.dropna(subset=[date_col, price_col])
        .sort_values(date_col)
        .reset_index(drop=True)
        .set_index(date_col)
    )
    df["price"] = df[price_col].astype(float)
    dates = df.index

    # ----------------------------
    # 3. Define adapter functions (H-step event; no leakage)
    # ----------------------------
    H = args.horizon

    def get_train(start: pd.Timestamp, end: pd.Timestamp):
        """Return (X_train, y_train) for [start, end) without look-ahead."""
        win = df.loc[(dates >= start) & (dates < end), ["price"]].copy()

        # Build y(t) = 1{ price(t+H) > threshold } within the window only.
        win["y_fwd"] = win["price"].shift(-H)

        # Drop the last H rows (they don't have t+H inside the window).
        if len(win) <= H:
            # Not enough rows to train — return empty frames to let the engine skip if needed
            return win.iloc[0:0][["price"]], win.iloc[0:0]["price"]

        X_train = win.iloc[:-H, :][["price"]]
        y_train = (win.iloc[:-H]["y_fwd"] > args.threshold).astype(int)

        # In case of any remaining NaNs (edge cases), align/drop safely
        mask = y_train.notna()
        return X_train.loc[mask], y_train.loc[mask]

    def get_forecast_row(t: pd.Timestamp):
        """Features for forecasting at time t."""
        return pd.DataFrame({"price": [df.loc[t, "price"]]}, index=[t])

    def get_outcome(t: pd.Timestamp):
        """Evaluate the same target used in training at t+H."""
        # t is guaranteed to be in dates (we construct forecast_dates from dates)
        pos = dates.get_indexer([t])[0]
        future_pos = pos + H
        # Should be valid because we restrict forecast_dates to exclude the last H rows
        future_t = dates[future_pos]
        return int(df.loc[future_t, "price"] > args.threshold)

    # ----------------------------
    # 4. Rolling backtest
    # ----------------------------
    window_rows = args.window

    # Only forecast dates that have an observable outcome at t+H
    # and leave a warm-up of `window_rows` rows.
    if len(dates) < window_rows + H:
        raise ValueError("Not enough data for the chosen --window and --horizon.")

    forecast_dates = dates[window_rows:-H]
    model_ctor = lambda: LinearRWEventModel(threshold=args.threshold)

    res = walk_forward(
        forecast_dates=forecast_dates,
        get_train=get_train,
        get_forecast_row=get_forecast_row,
        get_outcome=get_outcome,
        model_ctor=model_ctor,
        # Keep legacy keys expected by walk_forward, but pass row counts
        cfg={"window_months": window_rows, "step_months": 1},
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
