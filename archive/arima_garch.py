
from __future__ import annotations

import pandas as pd
import pmdarima as pm
from arch import arch_model
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA


class ArimaGarchModel:
    """An ARIMA-GARCH model for gas price prediction."""
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.arima_order = None
        self.arima_model = None
        self.garch_model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ArimaGarchModel":
        """Fit the ARIMA-GARCH model."""
        # Use auto_arima to find the best ARIMA order
        auto_arima_model = pm.auto_arima(
            y, 
            start_p=1, start_q=1,
            test='adf',       # use adftest to find optimal 'd'
            max_p=3, max_q=3, # maximum p and q
            m=1,              # frequency of series
            d=None,           # let model determine 'd'
            seasonal=False,   # No seasonality
            start_P=0, 
            D=0, 
            trace=False,
            error_action='ignore',  
            suppress_warnings=True, 
            stepwise=True
        )
        self.arima_order = auto_arima_model.order

        # Fit the ARIMA model
        self.arima_model = ARIMA(y, order=self.arima_order).fit()

        # Fit a GARCH(1,1) model on the residuals
        self.garch_model = arch_model(self.arima_model.resid, p=1, q=1).fit(disp='off')

        return self

    def predict_proba(self, X_future: pd.DataFrame) -> pd.Series:
        """Predict the probability of exceeding the threshold."""
        # Forecast mean from ARIMA
        arima_forecast = self.arima_model.get_forecast(steps=1)
        mean_forecast = arima_forecast.predicted_mean.iloc[0]

        # Forecast volatility from GARCH
        garch_forecast = self.garch_model.forecast(horizon=1)
        vol_forecast = garch_forecast.variance.iloc[-1, 0] ** 0.5

        # Calculate probability
        probability_over_threshold = 1 - norm.cdf(self.threshold, loc=mean_forecast, scale=vol_forecast)

        return pd.Series([probability_over_threshold], index=X_future.index)

    def predict_full_distribution(self, X_future: pd.DataFrame) -> dict:
        """Predict the full forecast distribution (mean and volatility)."""
        # Forecast mean from ARIMA
        arima_forecast = self.arima_model.get_forecast(steps=1)
        mean_forecast = arima_forecast.predicted_mean.iloc[0]

        # Forecast volatility from GARCH
        garch_forecast = self.garch_model.forecast(horizon=1)
        vol_forecast = garch_forecast.variance.iloc[-1, 0] ** 0.5

        return {'mean': mean_forecast, 'vol': vol_forecast}
