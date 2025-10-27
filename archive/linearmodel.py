import numpy as np
import pandas as pd
from math import erf, sqrt


class LinearRWEventModel:
    """
    Linear Regression + Random Walk Event Model
    -------------------------------------------
    • Fits a simple linear trend: price_t = a + b*t + error_t
    • Uses residual std (sigma) as random-walk uncertainty
    • Predicts P(next_price > threshold) under N(mu_next, sigma^2)

    Expected interface:
        .fit(X, y) -> self
        .predict_proba(X_future) -> pd.Series  # aligned to X_future.index

    Notes:
      - `X` should be a DataFrame with one column: 'price'
      - `y` is unused (kept for compatibility)
      - Index of X must be time-ordered
    """

    def __init__(self, threshold: float = 3.00):
        self.threshold = threshold
        self.intercept_ = None
        self.slope_ = None
        self.sigma_ = None
        self.n_train_ = None

    # ==========================================================
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearRWEventModel":
        """Fit linear regression to price over time and compute residual variance."""
        if X.shape[0] < 3:
            raise ValueError("Need at least 3 rows to fit regression.")

        # Treat index order as time steps
        t = np.arange(len(X), dtype=float)
        y_data = X.iloc[:, 0].astype(float).to_numpy()

        # Fit y = intercept + slope*t
        slope, intercept = np.polyfit(t, y_data, 1)
        y_hat = intercept + slope * t
        resid = y_data - y_hat

        self.intercept_ = float(intercept)
        self.slope_ = float(slope)
        self.sigma_ = float(np.std(resid, ddof=1))
        self.n_train_ = len(X)

        return self

    # ==========================================================
    def predict_proba(self, X_future: pd.DataFrame) -> pd.Series:
        """Return probability that the *next* price exceeds threshold."""
        if any(v is None for v in [self.intercept_, self.slope_, self.sigma_]):
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        # For each row in X_future, compute mu_next (trend-based mean)
        n_future = len(X_future)
        t_future = self.n_train_ + np.arange(n_future, dtype=float)
        mu_next = self.intercept_ + self.slope_ * t_future

        sigma = self.sigma_
        threshold = self.threshold

        if sigma <= 0:
            probs = np.where(mu_next > threshold, 1.0, 0.0)
        else:
            z = (mu_next - threshold) / sigma
            probs = 0.5 * (1.0 + erf(z / sqrt(2.0)))

        return pd.Series(probs, index=X_future.index, name="p_event")
