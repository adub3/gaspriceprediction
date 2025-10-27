from __future__ import annotations

import pandas as pd

from tests.evaluation.backtest import walk_forward
from tests.evaluation.metrics import brier_score, logloss, rocauc, calibration_table
from models.arima_garch import ArimaGarchModel


def main() -> None:
    # ----------------------------
    # 1. Load and prepare data
    # ----------------------------
    df = pd.read_csv("/Users/kabirgrewal/kalshiproject/gaspriceprediction/data/GASREGW.csv")
    df = df.rename(columns={"observation_date": "date", "GASREGW": "price"})
    df["date"] = pd.to_datetime(df["date"])
    df['price'] = df['price'].interpolate()
    df = df.set_index('date')

    # Target variable for evaluation
    y = (df['price'].shift(-1) > 3.0).astype(int).dropna()
    X = df[[]].loc[y.index] # Empty dataframe with correct index

    # ----------------------------
    # 3. Define adapter functions
    # ----------------------------
    def get_train(start: pd.Timestamp, end: pd.Timestamp):
        mask = (df.index >= start) & (df.index < end)
        # ARIMA-GARCH works with the price series directly
        return df.loc[mask, ['price']], df.loc[mask, 'price']

    def get_forecast_row(t: pd.Timestamp):
        # The model doesn't need features, just the timestamp for the index
        return pd.DataFrame(index=[t])

    def get_outcome(t: pd.Timestamp):
        return y.loc[t]

    # ----------------------------
    # 4. Rolling backtest
    # ----------------------------
    window_months = 120  # 10 years
    forecast_dates = y.index[window_months:]  # leave warm-up
    model_ctor = lambda: ArimaGarchModel(threshold=3.0)

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

    print("===== ARIMA-GARCH EVENT MODEL =====")
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
    res.to_csv("arima_garch_eval_results.csv")
    print("\nSaved detailed results to arima_garch_eval_results.csv")


if __name__ == "__main__":
    main()
