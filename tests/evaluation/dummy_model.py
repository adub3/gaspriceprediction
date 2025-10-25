
"""A tiny dummy model for testing the evaluation pipeline."""
from __future__ import annotations
import numpy as np
import pandas as pd

class DummyEventModel:
    """Minimal model implementing .fit() and .predict_proba() for tests."""
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.x_mean = None
        self.x_std = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DummyEventModel":
        """Fit: compute mean/std of first column for normalization."""
        x = np.asarray(X.iloc[:, 0], dtype=float)
        self.x_mean = np.mean(x)
        self.x_std = np.std(x) if np.std(x) > 0 else 1.0
        return self

    def predict_proba(self, X_future: pd.DataFrame) -> pd.Series:
        """Return a logistic probability for event (feature > threshold)."""
        x = np.asarray(X_future.iloc[:, 0], dtype=float)
        z = (x - self.threshold) / (self.x_std + 1e-8)
        p = 1.0 / (1.0 + np.exp(-z))
        return pd.Series(p, index=X_future.index, name="p_event")
