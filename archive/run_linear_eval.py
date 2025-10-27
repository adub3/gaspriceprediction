from __future__ import annotations

import pandas as pd

from tests.evaluation.backtest import walk_forward
from tests.evaluation.metrics import brier_score, logloss, rocauc, calibration_table
from models.linearmodel import LinearRWEventModel


def main() -> None:
    """Evaluate the LinearRWEventModel."""
    # ----------------------------
    # 1. Load and prepare data
    # ----------------------------
    df = pd.read_csv("/Users/kabirgrewal/kalshiproject/gaspriceprediction/data/GASREGW.csv")
    df = df.rename(columns={"observation_date": "date", "GASREGW": "price"})
    df["date"] = pd.to_datetime(df["date"])
    df['price'] = df['price'].interpolate()
    df = df.set_index('date')

    threshold = 3.0
    y = (df['price'].shift(-1) > threshold).astype(int).dropna()

    # ----------------------------
    # 2. Define adapter functions
    # ----------------------------
    def get_train(start: pd.Timestamp, end: pd.Timestamp):
        mask = (df.index >= start) & (df.index < end)
        # The linear model expects a DataFrame with a 'price' column
        X_train = df.loc[mask, ["price"]]
        # The target is whether the price *in 4 weeks* will be over the threshold
        y_train = (df.loc[mask, "price"].shift(-4) > threshold).astype(int)
        return X_train, y_train

    def get_forecast_row(t: pd.Timestamp):
        return pd.DataFrame({"price": [df.loc[t, "price"]]}, index=[t])

    def get_outcome(t: pd.Timestamp):
        return y.loc[t]

    # ----------------------------
    # 3. Rolling backtest
    # ----------------------------
    window_months = 120  # 10 years
    # Align forecast dates with the outcome series `y`
    forecast_dates = y.index.intersection(df.index[window_months:])

    model_ctor = lambda: LinearRWEventModel(threshold=threshold)

    res = walk_forward(
        forecast_dates=forecast_dates,
        get_train=get_train,
        get_forecast_row=get_forecast_row,
        get_outcome=get_outcome,
        model_ctor=model_ctor,
        cfg={"window_months": window_months, "step_months": 1},
    )

    # ----------------------------
    # 4. Save results
    # ----------------------------
    res.to_csv("linear_rw_eval_results.csv")
    print("\nSaved detailed results for Linear Model to linear_rw_eval_results.csv")


if __name__ == "__main__":
    main()
