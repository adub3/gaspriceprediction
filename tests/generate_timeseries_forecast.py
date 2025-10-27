from __future__ import annotations

import pandas as pd

from tests.evaluation.backtest import walk_forward
from models.arima_garch_simple import ArimaGarchSimpleModel


def main() -> None:
    # ----------------------------
    # 1. Load and prepare data
    # ----------------------------
    df = pd.read_csv("/Users/kabirgrewal/kalshiproject/gaspriceprediction/data/GASREGW.csv")
    df = df.rename(columns={"observation_date": "date", "GASREGW": "price"})
    df["date"] = pd.to_datetime(df["date"])
    df['price'] = df['price'].interpolate()
    df = df.set_index('date')

    # We need the actual price for the forecast period for plotting
    y = df['price'].shift(-4).dropna() # Shift by 4 weeks for the one-month-out forecast
    X = df[[]].loc[y.index]

    # ----------------------------
    # 3. Define adapter functions
    # ----------------------------
    def get_train(start: pd.Timestamp, end: pd.Timestamp):
        mask = (df.index >= start) & (df.index < end)
        return df.loc[mask, ['price']], df.loc[mask, 'price']

    def get_forecast_row(t: pd.Timestamp):
        return pd.DataFrame(index=[t])

    def get_outcome(t: pd.Timestamp):
        return y.loc[t]

    # ----------------------------
    # 4. Rolling backtest
    # ----------------------------
    window_months = 120  # 10 years
    forecast_dates = y.index[window_months:]  # leave warm-up
    model_ctor = lambda: ArimaGarchSimpleModel()

    def predictor(model, X_future):
        return model.predict_full_distribution(X_future)

    res = walk_forward(
        forecast_dates=forecast_dates,
        get_train=get_train,
        get_forecast_row=get_forecast_row,
        get_outcome=get_outcome,
        model_ctor=model_ctor,
        cfg={"window_months": window_months, "step_months": 1},
        predictor=predictor
    )

    # The results from walk_forward will be a series of dicts in the p_hat column
    res['mean_forecast'] = res['p_hat'].apply(lambda x: x['mean'])
    res['vol_forecast'] = res['p_hat'].apply(lambda x: x['vol'])
    res = res.rename(columns={'y': 'actual_price'})
    res = res.drop(columns=['p_hat'])

    # ----------------------------
    # 6. Save results
    # ----------------------------
    res.to_csv("arima_garch_forecast_results.csv")
    print("\nSaved detailed forecast results to arima_garch_forecast_results.csv")


if __name__ == "__main__":
    main()