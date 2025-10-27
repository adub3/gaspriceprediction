from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from models.arima_garch import ArimaGarchModel


def main() -> None:
    """Run a backtest to generate ARIMA-GARCH forecast data."""
    # ----------------------------
    # 1. Load and prepare data
    # ----------------------------
    df = pd.read_csv("/Users/kabirgrewal/kalshiproject/gaspriceprediction/data/GASREGW.csv")
    df = df.rename(columns={"observation_date": "date", "GASREGW": "price"})
    df["date"] = pd.to_datetime(df["date"])
    df['price'] = df['price'].interpolate()
    df = df.set_index('date')

    # ----------------------------
    # 2. Run rolling forecast
    # ----------------------------
    window_months = 120  # 10 years
    forecast_dates = df.index[window_months:]
    
    results = []
    model = ArimaGarchModel(threshold=3.0)

    # Use tqdm for a progress bar, as this is a long process
    for t_end in tqdm(forecast_dates, desc="Generating Forecast Data"):
        t_start = (t_end - pd.offsets.DateOffset(months=window_months)).normalize()
        
        # Define training data for this window
        train_mask = (df.index >= t_start) & (df.index < t_end)
        train_series = df.loc[train_mask, 'price']

        # Fit the model
        model.fit(None, train_series) # X is not used, y is the price series

        # Get the full distribution forecast
        forecast = model.predict_full_distribution(pd.DataFrame(index=[t_end]))

        results.append({
            "date": t_end,
            "actual_price": df.loc[t_end, "price"],
            "mean_forecast": forecast['mean'],
            "vol_forecast": forecast['vol'],
        })

    # ----------------------------
    # 3. Save results
    # ----------------------------
    results_df = pd.DataFrame(results).set_index("date")
    results_df.to_csv("arima_garch_forecast_data.csv")
    print("\nSaved detailed forecast data to arima_garch_forecast_data.csv")


if __name__ == "__main__":
    main()
