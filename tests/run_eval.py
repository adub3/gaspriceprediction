# tests/run_eval.py
from __future__ import annotations

import argparse
import os
from datetime import datetime
import pandas as pd

from tests.evaluation.backtest import walk_forward
from tests.evaluation.metrics import (
    evaluate_and_plot,  # new one-call wrapper
)
from models.linearmodel import LinearRWEventModel


def main() -> None:
    # ----------------------------
    # CLI
    # ----------------------------
    ap = argparse.ArgumentParser(description="Evaluate LinearRWEventModel on CSV data")
    ap.add_argument("--csv", required=True, help="Path to CSV with [date, price] columns")
    ap.add_argument("--threshold", type=float, required=True, help="Event threshold (e.g., 3.00)")
    ap.add_argument("--window", type=int, default=24, help="Rolling window size (rows/periods)")
    ap.add_argument("--date_col", type=str, default=None, help="Optional name of date column")
    ap.add_argument("--price_col", type=str, default=None, help="Optional name of price column")
    ap.add_argument("--horizon", type=int, default=4, help="Forecast horizon in rows/periods")

    # evaluation controls
    ap.add_argument("--bin_method", type=str, default="quantile",
                    choices=["quantile", "uniform"], help="Calibration binning")
    ap.add_argument("--bins", type=int, default=20, help="Target number of bins")
    ap.add_argument("--min_bin_n", type=int, default=50, help="Minimum samples per bin")
    ap.add_argument("--bootstrap", type=int, default=500, help="Bootstrap iterations for CIs (0=off)")
    ap.add_argument("--outdir", type=str, default=None, help="Output folder (default: ./eval_YYYYMMDD_HHMMSS)")
    args = ap.parse_args()

    # ----------------------------
    # Load data
    # ----------------------------
    df = pd.read_csv(args.csv)
    date_col = args.date_col or df.columns[0]
    price_col = args.price_col or df.columns[1]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = (df.dropna(subset=[date_col, price_col])
            .sort_values(date_col)
            .reset_index(drop=True)
            .set_index(date_col))
    df["price"] = df[price_col].astype(float)
    dates = df.index

    # ----------------------------
    # Label adapters
    # ----------------------------
    H = args.horizon

    def get_train(start: pd.Timestamp, end: pd.Timestamp):
        win = df.loc[(dates >= start) & (dates < end), ["price"]].copy()
        win["y_fwd"] = win["price"].shift(-H)
        if len(win) <= H:
            return win.iloc[0:0][["price"]], win.iloc[0:0]["price"]
        X_train = win.iloc[:-H, :][["price"]]
        y_train = (win.iloc[:-H]["y_fwd"] > args.threshold).astype(int)
        mask = y_train.notna()
        return X_train.loc[mask], y_train.loc[mask]

    def get_forecast_row(t: pd.Timestamp):
        return pd.DataFrame({"price": [df.loc[t, "price"]]}, index=[t])

    def get_outcome(t: pd.Timestamp):
        pos = dates.get_indexer([t])[0]
        future_pos = pos + H
        future_t = dates[future_pos]
        return int(df.loc[future_t, "price"] > args.threshold)

    # ----------------------------
    # Walk-forward backtest
    # ----------------------------
    window_rows = args.window
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
        cfg={"window_months": window_rows, "step_months": 1},
    )

    # ----------------------------
    # Evaluate & Plot (one call)
    # ----------------------------
    outdir = args.outdir or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(outdir, exist_ok=True)
    # Save raw predictions too
    res.to_csv(os.path.join(outdir, "linear_rw_eval_results.csv"), index=False)

    summary, calib_tbl = evaluate_and_plot(
        p_hat=res.p_hat.to_numpy(),
        y=res.y.to_numpy().astype(int),
        dates=res.index,  # used for rolling ECE if datetime-like
        outdir=outdir,
        bin_method=args.bin_method,
        n_bins=args.bins,
        min_bin_n=args.min_bin_n,
        bootstrap=args.bootstrap,
    )

    # ----------------------------
    # Print summary
    # ----------------------------
    print("\n===== LINEAR RANDOM-WALK EVENT MODEL =====")
    print(f"Samples evaluated: {len(res)}")
    for k in ("Brier","LogLoss","ROC_AUC","ECE","BSS"):
        if k in summary:
            print(f"{k}: {summary[k]:.6f}")
    if "ECE_CI_low" in summary:
        print(f"ECE 95% CI: ({summary['ECE_CI_low']:.6f}, {summary['ECE_CI_high']:.6f})")
    print("\nCalibration bins (head):")
    print(calib_tbl.head(10))
    print(f"\nSaved CSVs and figures to: {outdir}")


if __name__ == "__main__":
    main()
