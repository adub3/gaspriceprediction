
"""Unit tests for splitters: window math and leakage guard."""
from __future__ import annotations

import pandas as pd
import pytest

from tests.evaluation.splitters import RollingWindowConfig, rolling_windows, leakage_guard


def test_rolling_windows_length() -> None:
    dates = pd.date_range("2020-01-01", periods=3, freq="MS")
    cfg = RollingWindowConfig(window_months=2)

    windows = list(rolling_windows(dates, cfg))

    assert len(windows) == 3

    start0, end0 = windows[0]

    assert end0.month == 1
    assert start0.month == 11  # two months back from January


def test_leakage_guard_raises() -> None:
    idx = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])
    cutoff = pd.Timestamp("2020-02-01")  # anything after this is leakage

    with pytest.raises(ValueError):
        leakage_guard(idx, cutoff)
