from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    # ----------------------------
    # 1. Load forecast data
    # ----------------------------
    res = pd.read_csv("arima_garch_forecast_results.csv", index_col='date', parse_dates=True)

    # ----------------------------
    # 2. Calculate confidence intervals
    # ----------------------------
    # Calculate the 95% confidence interval
    res['lower_ci'] = res['mean_forecast'] - 1.96 * res['vol_forecast']
    res['upper_ci'] = res['mean_forecast'] + 1.96 * res['vol_forecast']

    # ----------------------------
    # 3. Create the plot
    # ----------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(res.index, res['actual_price'], label='Actual Price')
    plt.plot(res.index, res['mean_forecast'], label='Forecasted Price', linestyle='--')
    plt.fill_between(res.index, res['lower_ci'], res['upper_ci'], color='gray', alpha=0.2, label='95% Confidence Interval')

    plt.title('ARIMA-GARCH Forecast vs. Actual Gas Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 5)

    # ----------------------------
    # 4. Save the plot
    # ----------------------------
    plt.savefig("arima_garch_forecast_plot.png")
    print("\nSaved forecast plot to arima_garch_forecast_plot.png")


if __name__ == "__main__":
    main()