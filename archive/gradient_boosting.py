
from __future__ import annotations

import lightgbm as lgb
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for the gas price prediction model."""
    df_copy = df.copy().copy() # Create a mutable copy
    df_copy['price'] = df_copy['price'].interpolate()

    # Target variable
    df_copy['is_over_3'] = (df_copy['price'].shift(-4) > 3.0).astype(int)

    # Lag features
    for lag in [1, 2, 4, 8, 52]:
        df_copy[f'price_lag_{lag}'] = df_copy['price'].shift(lag)

    # Rolling window features
    for window in [4, 12, 52]:
        df_copy[f'rolling_mean_{window}'] = df_copy['price'].shift(1).rolling(window).mean()
        df_copy[f'rolling_std_{window}'] = df_copy['price'].shift(1).rolling(window).std()

    # Time-based features
    df_copy['month'] = df_copy['date'].dt.month
    df_copy['week_of_year'] = df_copy['date'].dt.isocalendar().week

    df_copy = df_copy.set_index('date')
    df_copy = df_copy.dropna()

    return df_copy


class GradientBoostingModel:
    """A gradient boosting model for gas price prediction."""
    def __init__(self):
        self.model = lgb.LGBMClassifier(objective='binary', n_estimators=100, random_state=42)
        self.feature_names = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GradientBoostingModel":
        """Fit the model."""
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X_future: pd.DataFrame) -> pd.Series:
        """Predict probabilities."""
        # Ensure X_future has the same columns as the training data
        X_future_aligned = X_future[self.feature_names]
        return pd.Series(self.model.predict_proba(X_future_aligned)[:, 1], index=X_future.index)

