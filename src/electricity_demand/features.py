# src/electricity_demand/features.py
"""
Feature engineering and time-series diagnostics.

Covers:
* Part 1 diagnostics -- STL decomposition, ADF + KPSS stationarity tests.
* Part 4 -- Berlin temperature covariate and weekly temperature features.
* Part 5 -- calendar, holiday, lag and rolling-window features, assembled into
  a supervised-learning table.

Leakage policy
--------------
Every lag/rolling feature is built with ``shift`` so a row at time ``t`` only
ever sees information available strictly before ``t``. Rolling statistics are
computed on the shifted series, never on the current value. Scalers/models are
fitted on the training portion only (that happens in the model modules, not
here).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from electricity_demand.config import (
    BERLIN_LAT,
    BERLIN_LON,
    OPEN_METEO_URL,
    HEATING_BASE_C,
    COOLING_BASE_C,
    WEEKLY_SEASONALITY,
)


# ==========================================================================
# Part 1: stationarity diagnostics
# ==========================================================================
def adf_test(series: pd.Series, **kwargs) -> pd.Series:
    """
    Augmented Dickey-Fuller test.

    H0: a unit root is present (series is non-stationary).
    A small p-value (< 0.05) lets us reject H0 in favour of stationarity.
    """
    from statsmodels.tsa.stattools import adfuller

    stat, pvalue, usedlag, nobs, crit, icbest = adfuller(
        series.dropna(), **kwargs
    )
    out = {
        "test": "ADF",
        "statistic": stat,
        "pvalue": pvalue,
        "used_lag": usedlag,
        "n_obs": nobs,
    }
    for k, v in crit.items():
        out[f"crit_{k}"] = v
    out["stationary_at_5pct"] = bool(pvalue < 0.05)
    return pd.Series(out)


def kpss_test(series: pd.Series, regression: str = "c", **kwargs) -> pd.Series:
    """
    KPSS test (complements ADF).

    H0: the series is (trend-)stationary.
    Here a *small* p-value (< 0.05) indicates NON-stationarity -- the opposite
    interpretation to ADF, which is why the two are reported together.
    """
    from statsmodels.tsa.stattools import kpss

    stat, pvalue, nlags, crit = kpss(
        series.dropna(), regression=regression, nlags="auto", **kwargs
    )
    out = {
        "test": f"KPSS ({regression})",
        "statistic": stat,
        "pvalue": pvalue,
        "n_lags": nlags,
    }
    for k, v in crit.items():
        out[f"crit_{k}"] = v
    out["stationary_at_5pct"] = bool(pvalue >= 0.05)
    return pd.Series(out)


def stationarity_report(series: pd.Series) -> pd.DataFrame:
    """Run ADF and KPSS and return a tidy comparison table."""
    return pd.concat(
        [adf_test(series), kpss_test(series)], axis=1
    ).T.reset_index(drop=True)


def decompose(series: pd.Series, period: int = WEEKLY_SEASONALITY,
              robust: bool = True):
    """STL decomposition into trend / seasonal / residual components."""
    from statsmodels.tsa.seasonal import STL

    stl = STL(series, period=period, robust=robust)
    return stl.fit()


# ==========================================================================
# Part 4: temperature covariate
# ==========================================================================
def get_open_meteo_temperature(
    start_date: str,
    end_date: str,
    latitude: float = BERLIN_LAT,
    longitude: float = BERLIN_LON,
) -> pd.DataFrame:
    """
    Download daily mean 2 m temperature for Berlin from the Open-Meteo archive.

    Requires network access to ``archive-api.open-meteo.com``. Returns a frame
    indexed by date with a single column ``temperature_2m_mean``.
    """
    import requests

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean",
        "timezone": "Europe/Berlin",
    }
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=120)
    resp.raise_for_status()
    daily = resp.json()["daily"]

    temp = pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"]),
            "temperature_2m_mean": daily["temperature_2m_mean"],
        }
    ).set_index("date")
    return temp


def weekly_temperature_features(temp_daily: pd.DataFrame,
                                weekly_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Turn daily Berlin temperature into weekly features aligned to ``weekly_index``:
    mean/min/max and heating/cooling degree sums.
    """
    t = temp_daily["temperature_2m_mean"]

    feats = pd.DataFrame(index=weekly_index)
    feats["temp_mean"] = t.resample("W").mean()
    feats["temp_min"] = t.resample("W").min()
    feats["temp_max"] = t.resample("W").max()

    heating = np.maximum(HEATING_BASE_C - t, 0.0).resample("W").sum()
    cooling = np.maximum(t - COOLING_BASE_C, 0.0).resample("W").sum()
    feats["heating_degree_days"] = heating
    feats["cooling_degree_days"] = cooling

    return feats.reindex(weekly_index)


# ==========================================================================
# Part 5: calendar / holiday / lag / rolling features
# ==========================================================================
def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic calendar features, all known at any forecast origin."""
    idx = pd.DatetimeIndex(index)
    woy = idx.isocalendar().week.astype(int).to_numpy()
    df = pd.DataFrame(index=idx)
    df["month"] = idx.month
    df["week_of_year"] = woy
    df["quarter"] = idx.quarter
    # Fourier terms for the annual cycle -- smooth seasonal encoding.
    df["sin_year"] = np.sin(2 * np.pi * woy / WEEKLY_SEASONALITY)
    df["cos_year"] = np.cos(2 * np.pi * woy / WEEKLY_SEASONALITY)
    return df


def holiday_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    German public-holiday features per week. Known in advance, so valid future
    covariates. Falls back to zeros if the optional ``holidays`` package is
    absent.
    """
    idx = pd.DatetimeIndex(index)
    df = pd.DataFrame(index=idx)
    try:
        import holidays as _holidays

        years = range(idx.min().year, idx.max().year + 1)
        de = _holidays.Germany(years=years)
        # Count holidays falling in each (weekly) bin: reindex daily then group.
        daily = pd.date_range(idx.min() - pd.Timedelta(days=7), idx.max(), freq="D")
        is_hol = pd.Series([d in de for d in daily], index=daily, dtype=float)
        weekly_counts = is_hol.resample("W").sum().reindex(idx).fillna(0.0)
        df["holiday_days"] = weekly_counts
        df["has_holiday"] = (weekly_counts > 0).astype(int)
    except Exception:  # pragma: no cover - optional dependency
        df["holiday_days"] = 0.0
        df["has_holiday"] = 0
    return df


def add_lag_features(df: pd.DataFrame, target: str = "load_gw",
                     lags=(1, 2, 3, 4, 52)) -> pd.DataFrame:
    """Add strictly-past lags of the target (leakage-safe via shift)."""
    out = df.copy()
    for L in lags:
        out[f"{target}_lag_{L}"] = out[target].shift(L)
    return out


def add_rolling_features(df: pd.DataFrame, target: str = "load_gw",
                         windows=(4, 8, 52)) -> pd.DataFrame:
    """
    Rolling mean/std over strictly-past values. We shift by one BEFORE rolling
    so the window for row t ends at t-1 and never includes the current target.
    """
    out = df.copy()
    past = out[target].shift(1)
    for w in windows:
        out[f"{target}_rollmean_{w}"] = past.rolling(w).mean()
        out[f"{target}_rollstd_{w}"] = past.rolling(w).std()
    return out


def make_feature_table(
    weekly_load: pd.Series,
    temp_weekly: pd.DataFrame | None = None,
    target: str = "load_gw",
) -> pd.DataFrame:
    """
    Assemble the full weekly modelling table: target + temperature (optional) +
    calendar + holiday + lag + rolling features. Rows with NaNs introduced by
    lagging are dropped at the end.
    """
    df = weekly_load.to_frame(target).copy()

    if temp_weekly is not None:
        df = df.join(temp_weekly)

    df = df.join(calendar_features(df.index))
    df = df.join(holiday_features(df.index))
    df = add_lag_features(df, target=target)
    df = add_rolling_features(df, target=target)

    # Interpolate any covariate gaps in time, then drop rows still missing
    # (the initial lag burn-in).
    df = df.interpolate("time").dropna()
    return df
