import numpy as np
import pandas as pd

from electricity_demand.models.benchmarks import (
    mean_forecast, naive_forecast, seasonal_naive_forecast, drift_forecast,
)


def _series(n=120):
    idx = pd.date_range("2015-01-04", periods=n, freq="W")
    return pd.Series(np.arange(n, dtype=float) + 10.0, index=idx)


def test_forecast_lengths_match_horizon():
    train = _series()
    h = 12
    idx = pd.date_range(train.index[-1] + pd.Timedelta(weeks=1), periods=h, freq="W")
    for fn in (mean_forecast, naive_forecast, drift_forecast):
        assert len(fn(train, h, index=idx)) == h
    assert len(seasonal_naive_forecast(train, h, seasonality=52, index=idx)) == h


def test_naive_is_last_value():
    train = _series()
    fc = naive_forecast(train, 5)
    assert np.allclose(fc.values, train.iloc[-1])


def test_seasonal_naive_repeats_last_season():
    train = _series(110)
    fc = seasonal_naive_forecast(train, 52, seasonality=52)
    assert np.allclose(fc.values, train.iloc[-52:].values)


def test_drift_is_linear():
    train = _series()  # perfectly linear
    fc = drift_forecast(train, 3)
    # slope is 1.0 per step for arange series
    assert np.allclose(np.diff(fc.values), 1.0)
