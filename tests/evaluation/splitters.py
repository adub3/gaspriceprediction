
"""Time-aware utilities for rolling backtests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import pandas as pd


@dataclass
class RollingWindowConfig:
    """Configuration for rolling windows in months."""
    window_months: int = 120
    step_months: int = 1


def add_months(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    """Return a timestamp shifted by a given number of calendar months."""
    return (ts + pd.offsets.DateOffset(months=months)).normalize()


def rolling_windows(
    dates: pd.DatetimeIndex,
    cfg: RollingWindowConfig,
) -> Iterator[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Yield (start, end) training windows for each end date in `dates`."""
    for end in dates:
        start = add_months(end, -cfg.window_months)
        yield (start, end)


def leakage_guard(feature_dates: pd.DatetimeIndex, cutoff: pd.Timestamp) -> None:
    """Raise if any feature timestamp is after (>) the forecast cutoff."""
    if feature_dates.max() > cutoff:
        raise ValueError(
            f"Data leakage detected: feature timestamp beyond cutoff {cutoff.date()}"
        )
