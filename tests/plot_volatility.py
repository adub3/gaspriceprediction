from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    # ----------------------------
    # 1. Load forecast data
    # ----------------------------
    res = pd.read_csv("arima_garch_forecast_results.csv", index_col='date', parse_dates=True)

    # ----------------------------
    # 2. Create the plot
    # ----------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(res.index, res['vol_forecast'], label='Forecasted Volatility')

    plt.title('Forecasted Volatility Over Time')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)

    # ----------------------------
    # 3. Save the plot
    # ----------------------------
    plt.savefig("volatility_over_time.png")
    print("\nSaved volatility plot to volatility_over_time.png")


if __name__ == "__main__":
    main()

