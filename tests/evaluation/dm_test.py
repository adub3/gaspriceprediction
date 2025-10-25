
"""Diebold–Mariano test with a Newey–West variance estimator.

This tests whether two forecast models have the same expected loss.
Input is the *per-period* losses from each model (same shape).
"""
from __future__ import annotations

from typing import Literal, Tuple

import numpy as np


def _bartlett_weights(max_lag: int) -> np.ndarray:
    """Return Bartlett (triangular) weights w[0]=1, w[l]=1 - l/(L+1)."""
    if max_lag < 0:
        raise ValueError("max_lag must be >= 0")

    w = np.ones(max_lag + 1, dtype=float)

    if max_lag > 0:
        w[1:] = 1.0 - np.arange(1, max_lag + 1, dtype=float) / (max_lag + 1)

    return w


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    L: int | None = None,
    alternative: Literal["two_sided", "greater", "less"] = "two_sided",
) -> Tuple[float, float]:
    """Test equal predictive accuracy between model A and model B.

    Parameters
    ----------
    loss_a, loss_b : arrays of losses for the same periods
    L : max lag for Newey–West variance; defaults to floor(T**(1/3))
    alternative :
        - "two_sided": A and B have different mean loss
        - "greater"  : A has *greater* mean loss than B (A worse)
        - "less"     : A has *smaller* mean loss than B (A better)

    Returns
    -------
    (statistic, p_value)
    """
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("loss_a and loss_b must have the same shape")

    d = a - b  # positive → A worse than B
    T = d.size
    mean_d = float(np.mean(d))

    if L is None:
        L = int(np.floor(T ** (1.0 / 3.0)))

    # Newey–West variance of mean(d)
    weights = _bartlett_weights(L)

    # gamma_0 (variance)
    nw_var = np.var(d, ddof=1)

    # add 2 * sum w_l * gamma_l
    for lag in range(1, L + 1):
        cov = np.cov(d[:-lag], d[lag:], ddof=1)[0, 1]
        nw_var += 2.0 * weights[lag] * cov

    nw_var = nw_var / T

    if nw_var > 0.0:
        stat = mean_d / np.sqrt(nw_var)
    else:
        stat = np.inf

    # Normal CDF via erf (avoid SciPy dependency)
    from math import erf, sqrt

    def std_norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    if alternative == "two_sided":
        p_value = 2.0 * (1.0 - std_norm_cdf(abs(stat)))
    elif alternative == "greater":
        p_value = 1.0 - std_norm_cdf(stat)
    elif alternative == "less":
        p_value = std_norm_cdf(stat)
    else:
        raise ValueError("alternative must be 'two_sided', 'greater', or 'less'")

    return float(stat), float(p_value)
