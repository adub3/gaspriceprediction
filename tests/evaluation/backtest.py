
"""Generic walk-forward (rolling) backtest engine.

Models must implement:
    .fit(X, y) -> self
    .predict_proba(X_future) -> pd.Series  # one probability aligned to X_future index
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import pandas as pd

from .splitters import RollingWindowConfig, leakage_guard


def walk_forward(
    forecast_dates: pd.DatetimeIndex,
    get_train: Callable[[pd.Timestamp, pd.Timestamp], tuple[pd.DataFrame, pd.Series]],
    get_forecast_row: Callable[[pd.Timestamp], pd.DataFrame],
    get_outcome: Callable[[pd.Timestamp], int],
    model_ctor: Callable[[], Any],
    cfg: Dict[str, Any],
    predictor: Callable | None = None,
) -> pd.DataFrame:
    """Run a rolling backtest and return a tidy DataFrame of results.

    Returns a DataFrame indexed by date with columns:
        - p_hat : model probability of the event at that origin
        - y     : realized 0/1 outcome
    """
    window_months = int(cfg.get("window_months", 120))
    step_months = int(cfg.get("step_months", 1))

    # We step by months through the provided forecast_dates.
    results: list[dict] = []

    for t_end in forecast_dates:
        t_start = (t_end - pd.offsets.DateOffset(months=window_months)).normalize()

        # Training slice (vintage-safe)
        X_train, y_train = get_train(t_start, t_end)

        if hasattr(X_train, "index"):
            leakage_guard(pd.DatetimeIndex(X_train.index), t_end)

        model = model_ctor()
        model = model.fit(X_train, y_train)

        # One-row forecast (vintage-safe)
        X_future = get_forecast_row(t_end)

        if hasattr(X_future, "index"):
            leakage_guard(pd.DatetimeIndex(X_future.index), t_end)

        if predictor:
            p_hat = predictor(model, X_future)
        else:
            p_series = model.predict_proba(X_future)
            p_hat = float(p_series.iloc[-1])

        # Realized outcome (0/1)
        y_val = get_outcome(t_end)
        if isinstance(y_val, (int, float)):
            y_val = y_val

        results.append({
            "date": t_end,
            "p_hat": p_hat,
            "y": y_val,
        })

    df = pd.DataFrame(results).set_index("date").sort_index()

    return df
