
"""Shared metrics for evaluating binary event forecasts.

All functions here are *data-agnostic*: they accept arrays/Series of predictions
and outcomes and return scalar scores or summary tables.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss as sk_log_loss, roc_auc_score


def brier_score(p: np.ndarray | pd.Series, y: np.ndarray | pd.Series) -> float:
    """Mean squared error between predicted probabilities and binary outcomes.

    Lower is better. Perfect calibration and sharpness → 0.0
    """
    p_arr = np.asarray(p, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    return float(np.mean((p_arr - y_arr) ** 2))


def logloss(p: np.ndarray | pd.Series, y: np.ndarray | pd.Series, eps: float = 1e-12) -> float:
    """Cross-entropy / negative log-likelihood for Bernoulli outcomes.

    Lower is better. Harshly penalizes confident wrong calls.
    """
    p_arr = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    y_arr = np.asarray(y, dtype=float)

    return float(sk_log_loss(y_arr, p_arr, labels=[0, 1]))


def rocauc(p: np.ndarray | pd.Series, y: np.ndarray | pd.Series) -> float:
    """Area under ROC curve (discrimination).

    If y has only one class, returns NaN.
    """
    y_arr = np.asarray(y, dtype=float)
    unique = np.unique(y_arr)

    if unique.size < 2:
        return float('nan')

    return float(roc_auc_score(y_arr, np.asarray(p, dtype=float)))


def calibration_table(
    p: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    bins: int = 10,
) -> Tuple[pd.DataFrame, float]:
    """Reliability table + Expected Calibration Error (ECE).

    We bin by quantiles of predicted probability so that bins are populated.
    """
    df = pd.DataFrame({
        "p": np.asarray(p, dtype=float),
        "y": np.asarray(y, dtype=float),
    })

    df["bin"] = pd.qcut(df["p"], q=bins, duplicates="drop")

    table = (
        df.groupby("bin")
          .agg(p_mean=("p", "mean"), y_freq=("y", "mean"), n=("y", "size"))
          .reset_index(drop=True)
    )

    table["abs_gap"] = (table["y_freq"] - table["p_mean"]).abs()

    ece = float((table["abs_gap"] * table["n"] / table["n"].sum()).sum())

    return table, ece


def interval_coverage(lower: np.ndarray, upper: np.ndarray, y_cont: np.ndarray) -> float:
    """Share of times y_cont lies inside [lower, upper]."""
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    y = np.asarray(y_cont, dtype=float)

    inside = (y >= lo) & (y <= hi)

    return float(np.mean(inside))


def pinball_loss(q_pred: np.ndarray, y: np.ndarray, tau: float) -> float:
    """Quantile (pinball) loss at level tau in (0, 1)."""
    q = np.asarray(q_pred, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    diff = y_arr - q

    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))
