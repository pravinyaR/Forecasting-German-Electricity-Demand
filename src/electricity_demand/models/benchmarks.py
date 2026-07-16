# src/electricity_demand/models/benchmarks.py
"""
Benchmark forecasters from Lecture 1 (Part 2).

Each function takes the training series and a horizon and returns a forecast as
a pandas Series aligned to ``index`` (the test index). These are the models
every more complex approach must beat -- seasonal-naive in particular, because
weekly electricity demand has a strong annual cycle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_index(horizon: int, index=None):
    return index if index is not None else pd.RangeIndex(horizon)


def mean_forecast(train: pd.Series, horizon: int, index=None) -> pd.Series:
    """Every future value = mean of the training series."""
    value = float(np.asarray(train, dtype=float).mean())
    idx = _as_index(horizon, index)
    return pd.Series(np.full(horizon, value), index=idx, name="mean")


def naive_forecast(train: pd.Series, horizon: int, index=None) -> pd.Series:
    """Every future value = the last observed value."""
    last = float(np.asarray(train, dtype=float)[-1])
    idx = _as_index(horizon, index)
    return pd.Series(np.full(horizon, last), index=idx, name="naive")


def seasonal_naive_forecast(train: pd.Series, horizon: int,
                            seasonality: int = 52, index=None) -> pd.Series:
    """
    Value at horizon step h = value one full season earlier. For weekly data
    with seasonality 52 this is 'same week last year'.
    """
    vals = np.asarray(train, dtype=float)
    if len(vals) < seasonality:
        raise ValueError("Training series shorter than one seasonal period.")
    last_season = vals[-seasonality:]
    preds = np.array([last_season[i % seasonality] for i in range(horizon)])
    idx = _as_index(horizon, index)
    return pd.Series(preds, index=idx, name="seasonal_naive")


def drift_forecast(train: pd.Series, horizon: int, index=None) -> pd.Series:
    """
    Straight line through the first and last training points, extrapolated.
    Slope = (y_last - y_first) / (n - 1).
    """
    vals = np.asarray(train, dtype=float)
    n = len(vals)
    slope = (vals[-1] - vals[0]) / (n - 1)
    steps = np.arange(1, horizon + 1)
    preds = vals[-1] + slope * steps
    idx = _as_index(horizon, index)
    return pd.Series(preds, index=idx, name="drift")


def all_benchmarks(train: pd.Series, horizon: int, index=None,
                   seasonality: int = 52) -> dict[str, pd.Series]:
    """Convenience: return all four benchmark forecasts in a dict."""
    return {
        "mean": mean_forecast(train, horizon, index),
        "naive": naive_forecast(train, horizon, index),
        "seasonal_naive": seasonal_naive_forecast(train, horizon, seasonality, index),
        "drift": drift_forecast(train, horizon, index),
    }
