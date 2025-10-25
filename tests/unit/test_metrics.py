
"""Unit tests for metrics: quick sanity checks."""
from __future__ import annotations

import numpy as np

from tests.evaluation.metrics import (
    brier_score,
    logloss,
    rocauc,
    calibration_table,
)


def test_brier_basic() -> None:
    p = np.array([0.1, 0.9, 0.8, 0.2], dtype=float)
    y = np.array([0.0, 1.0, 1.0, 0.0], dtype=float)

    # Perfect probs → Brier = 0
    assert abs(brier_score(y, y) - 0.0) < 1e-12

    val = brier_score(p, y)

    assert 0.0 < val < 0.05


def test_logloss_sanity() -> None:
    p = np.array([0.01, 0.99, 0.8, 0.2], dtype=float)
    y = np.array([0.0, 1.0, 1.0, 0.0], dtype=float)

    ll = logloss(p, y)
    bad = logloss(1.0 - p, y)

    assert ll > 0.0
    assert bad > ll  # overconfident wrong gets punished


def test_rocauc_perfect() -> None:
    y = np.array([0, 0, 1, 1], dtype=float)
    p = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)

    assert abs(rocauc(p, y) - 1.0) < 1e-12


def test_calibration_table_and_ece() -> None:
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], dtype=float)
    y = np.array([0,   0,   0,   1,   1,   1], dtype=float)

    table, ece = calibration_table(p, y, bins=3)

    assert len(table) == 3
    assert 0.0 <= ece <= 1.0
