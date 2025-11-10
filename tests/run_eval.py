# tests/run_eval.py
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
import pandas as pd

from tests.evaluation.backtest import walk_forward
from tests.evaluation.metrics import evaluate_and_plot  # one-call wrapper
from models.linearmodel import LinearRWEventModel


# ----------------------------
# Helpers
# ----------------------------
DEFAULT_ARIMA_BASELINE = {
    # From your table snippet
    "Name": "ARIMA–GARCH",
    "Bias": 0.0049,
    "MAE": 0.1166,
    "RMSE": 0.1753,
    "Brier": 0.0271,
    "LogLoss": 0.2527,
    "BSS": 0.8672,   # 86.72%
    "AvgVol": 0.1101,
    "CI_Coverage": 0.5315,  # 53.15%
    # leave AUC/ECE blank if not applicable to baseline
    "ROC_AUC": None,
    "ECE": None,
}

def pct_or_na(x):
    if x is None:
        return "--"
    try:
        return f"{100.0 * float(x):.2f}\\%"
    except Exception:
        return "--"

def val_or_dash(x, digits=4):
    if x is None:
        return "--"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "--"

def with_ci(val, lo, hi, digits=4):
    """Format value with 95% CI if available: 0.0568 (0.0487–0.0658)"""
    v = val_or_dash(val, digits)
    if lo is None or hi is None:
        return v
    try:
        return f"{float(val):.{digits}f} ({float(lo):.{digits}f}–{float(hi):.{digits}f})"
    except Exception:
        return v

def write_calib_bins(calib_tbl: pd.DataFrame, outdir: str):
    path = os.path.join(outdir, "calibration_bins.csv")
    calib_tbl.to_csv(path, index=False)
    return path

def write_summary_json(summary: dict, outdir: str):
    path = os.path.join(outdir, "metrics_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return path

def load_baseline(path_json: str | None) -> dict:
    if path_json is None:
        return DEFAULT_ARIMA_BASELINE
    with open(path_json, "r", encoding="utf-8") as f:
        return json.load(f)

def latex_compare_table(baseline: dict, model_summary: dict, model_name: str = "LinearRW") -> str:
    """
    Build a compact LaTeX table comparing ARIMA–GARCH baseline vs New Model using available fields.
    Shows CIs where present in model_summary (ECE, Brier, LogLoss, AUC).
    """
    # Pull model values
    brier = model_summary.get("Brier")
    brier_lo = model_summary.get("Brier_CI_low")
    brier_hi = model_summary.get("Brier_CI_high")

    logloss = model_summary.get("LogLoss")
    logloss_lo = model_summary.get("LogLoss_CI_low")
    logloss_hi = model_summary.get("LogLoss_CI_high")

    auc = model_summary.get("ROC_AUC")
    auc_lo = model_summary.get("ROC_AUC_CI_low")
    auc_hi = model_summary.get("ROC_AUC_CI_high")

    ece = model_summary.get("ECE")
    ece_lo = model_summary.get("ECE_CI_low")
    ece_hi = model_summary.get("ECE_CI_high")

    bss = model_summary.get("BSS")  # fraction (e.g., 0.7127)

    # Build LaTeX
    lines = []
    lines.append(r"% --- Table: Forecast Performance Comparison ---")
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Forecast Performance Comparison: ARIMA--GARCH vs. " + model_name + r"}")
    lines.append(r"\label{tab:forecast-performance}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Metric} & \textbf{ARIMA--GARCH} & \textbf{" + model_name + r"} \\")
    lines.append(r"\midrule")
    lines.append(rf"Mean Error (Bias) & {val_or_dash(baseline.get('Bias'))} & -- \\")
    lines.append(rf"Mean Absolute Error (MAE) & {val_or_dash(baseline.get('MAE'))} & -- \\")
    lines.append(rf"Root Mean Squared Error (RMSE) & {val_or_dash(baseline.get('RMSE'))} & -- \\")
    lines.append(rf"Brier Score & {val_or_dash(baseline.get('Brier'))} & {with_ci(brier, brier_lo, brier_hi)} \\")
    lines.append(rf"Log Loss & {val_or_dash(baseline.get('LogLoss'))} & {with_ci(logloss, logloss_lo, logloss_hi)} \\")
    lines.append(rf"ROC--AUC & {val_or_dash(baseline.get('ROC_AUC'))} & {with_ci(auc, auc_lo, auc_hi)} \\")
    lines.append(rf"Expected Calibration Error (ECE) & {val_or_dash(baseline.get('ECE'))} & {with_ci(ece, ece_lo, ece_hi)} \\")
    lines.append(rf"Brier Skill Score (RVE) & {pct_or_na(baseline.get('BSS'))} & {pct_or_na(bss)} \\")
    lines.append(rf"Average Forecasted Volatility & {val_or_dash(baseline.get('AvgVol'))} & -- \\")
    lines.append(rf"95\% CI Coverage & {pct_or_na(baseline.get('CI_Coverage'))} & -- \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

def write_latex_table(tex_str: str, outdir: str, fname: str = "forecast_performance_comparison.tex"):
    path = os.path.join(outdir, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex_str)
    return path


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

    # outputs
    ap.add_argument("--outdir", type=str, default=None, help="Output folder (default: ./eval_YYYYMMDD_HHMMSS)")
    ap.add_argument("--emit_tex", action="store_true", help="Write LaTeX comparison table vs ARIMA–GARCH")
    ap.add_argument("--baseline_json", type=str, default=None,
                    help="Optional JSON file with baseline metrics (overrides built-in ARIMA–GARCH defaults)")
    ap.add_argument("--model_name", type=str, default="LinearRW", help="Name label for the evaluated model")
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

    # Save raw predictions
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

    # Persist artifacts
    bins_path = write_calib_bins(calib_tbl, outdir)
    summary_path = write_summary_json(summary, outdir)

    # ----------------------------
    # Print summary
    # ----------------------------
    print("\n===== LINEAR RANDOM-WALK EVENT MODEL =====")
    print(f"Samples evaluated: {len(res)}")

    # Core metrics
    if "Brier" in summary:   print(f"Brier:   {summary['Brier']:.6f}")
    if "LogLoss" in summary: print(f"LogLoss: {summary['LogLoss']:.6f}")
    if "ROC_AUC" in summary: print(f"ROC-AUC: {summary['ROC_AUC']:.6f}")
    if "ECE" in summary:     print(f"ECE:     {summary['ECE']:.6f}")

    # Brier Skill Score
    if "BSS" in summary:
        try:
            print(f"Brier Skill Score: {summary['BSS']:.6f} ({100.0*summary['BSS']:.2f}%)")
        except Exception:
            print(f"Brier Skill Score: {summary['BSS']}")

    # Optional CIs (shown if bootstrap > 0 and keys exist)
    def maybe_ci(field):
        lo = summary.get(f"{field}_CI_low")
        hi = summary.get(f"{field}_CI_high")
        if lo is not None and hi is not None and field in summary:
            print(f"{field} 95% CI: ({lo:.6f}, {hi:.6f})")

    for fld in ["ECE", "Brier", "LogLoss", "ROC_AUC"]:
        maybe_ci(fld)

    print("\nCalibration bins (head):")
    print(calib_tbl.head(10))

    print(f"\nSaved raw predictions: {os.path.join(outdir, 'linear_rw_eval_results.csv')}")
    print(f"Saved calibration bins: {bins_path}")
    print(f"Saved metrics summary:  {summary_path}")

    # ----------------------------
    # LaTeX comparison (optional)
    # ----------------------------
    if args.emit_tex:
        baseline = load_baseline(args.baseline_json)
        tex_str = latex_compare_table(baseline, summary, model_name=args.model_name)
        tex_path = write_latex_table(tex_str, outdir)
        print(f"Saved LaTeX comparison table: {tex_path}")


if __name__ == "__main__":
    main()
