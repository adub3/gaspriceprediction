
from __future__ import annotations

import pandas as pd

from tests.evaluation.backtest import walk_forward
from tests.evaluation.metrics import brier_score, logloss, rocauc, calibration_table
from models.gradient_boosting import GradientBoostingModel, create_features


def main() -> None:
    # ----------------------------
    # 1. Load and prepare data
    # ----------------------------
    df = pd.read_csv("/Users/kabirgrewal/kalshiproject/gaspriceprediction/data/GASREGW.csv")
    df = df.rename(columns={"observation_date": "date", "GASREGW": "price"})
    df["date"] = pd.to_datetime(df["date"])

    # ----------------------------
    # 2. Feature Engineering
    # ----------------------------
    features_df = create_features(df)
    X = features_df.drop(columns=["price", "is_over_3"])
    y = features_df["is_over_3"]
    dates = features_df.index

    # ----------------------------
    # 3. Define adapter functions
    # ----------------------------
    def get_train(start: pd.Timestamp, end: pd.Timestamp):
        mask = (X.index >= start) & (X.index < end)
        X_train = X.loc[mask]
        y_train = y.loc[mask]
        return X_train, y_train

    def get_forecast_row(t: pd.Timestamp):
        return X.loc[[t]]

    def get_outcome(t: pd.Timestamp):
        return y.loc[t]

    # ----------------------------
    # 4. Rolling backtest
    # ----------------------------
    window_months = 120  # 10 years
    forecast_dates = X.index[window_months:]  # leave warm-up
    model_ctor = lambda: GradientBoostingModel()

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

    print("\n===== GRADIENT BOOSTING EVENT MODEL =====")
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
    res.to_csv("gb_eval_results.csv")
    print("\nSaved detailed results to gb_eval_results.csv")


if __name__ == "__main__":
    main()
