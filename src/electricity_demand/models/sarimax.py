# src/electricity_demand/models/sarimax.py
"""
SARIMA / SARIMAX model (Parts 3 and 4).

* ``grid_search_sarima`` loops over the required (p, d, q) grid -- p in [0,6],
  d in [0,2], q in [0,6] -- plus a small seasonal grid, and selects the model
  with the lowest AIC.
* ``fit_sarimax`` fits a chosen order, optionally with exogenous regressors
  (temperature -> the 'X' in SARIMAX, Part 4).
* ``forecast_sarimax`` returns the mean forecast and a prediction interval.
"""

from __future__ import annotations

import warnings
import itertools

import numpy as np
import pandas as pd

from electricity_demand.config import (
    P_RANGE, D_RANGE, Q_RANGE,
    SEASONAL_P_RANGE, SEASONAL_D_RANGE, SEASONAL_Q_RANGE,
    WEEKLY_SEASONALITY,
    DEFAULT_ORDER, DEFAULT_SEASONAL_ORDER,
)


def _fit_one(y, order, seasonal_order, exog=None):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        y,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        trend="c" if order[1] == 0 else "n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def grid_search_sarima(
    y_train: pd.Series,
    exog_train=None,
    seasonal: bool = True,
    seasonal_period: int = WEEKLY_SEASONALITY,
    p_range=P_RANGE, d_range=D_RANGE, q_range=Q_RANGE,
    sp_range=SEASONAL_P_RANGE, sd_range=SEASONAL_D_RANGE, sq_range=SEASONAL_Q_RANGE,
    max_models: int | None = None,
    verbose: bool = True,
):
    """
    Exhaustive AIC search. Returns ``(best_fit, results_df)`` where
    ``results_df`` is every attempted order sorted by AIC.

    Failing fits (non-convergence, singular matrices) are skipped and recorded
    with AIC = NaN so the search is robust. Set ``max_models`` to cap runtime
    while developing.
    """
    if seasonal:
        seasonal_orders = [
            (P, D, Q, seasonal_period)
            for P, D, Q in itertools.product(sp_range, sd_range, sq_range)
        ]
    else:
        seasonal_orders = [(0, 0, 0, 0)]

    orders = list(itertools.product(p_range, d_range, q_range))
    combos = list(itertools.product(orders, seasonal_orders))
    if max_models:
        combos = combos[:max_models]

    records = []
    best = {"aic": np.inf, "fit": None, "order": None, "seasonal_order": None}

    for i, (order, sorder) in enumerate(combos, 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = _fit_one(y_train, order, sorder, exog=exog_train)
            aic = float(fit.aic)
            records.append({"order": order, "seasonal_order": sorder,
                            "aic": aic, "bic": float(fit.bic)})
            if aic < best["aic"]:
                best.update(aic=aic, fit=fit, order=order, seasonal_order=sorder)
                if verbose:
                    print(f"[{i}/{len(combos)}] new best AIC={aic:.1f} "
                          f"order={order} seasonal={sorder}")
        except Exception as exc:  # noqa: BLE001
            records.append({"order": order, "seasonal_order": sorder,
                            "aic": np.nan, "bic": np.nan})
            if verbose and i % 25 == 0:
                print(f"[{i}/{len(combos)}] ... (last failure: {type(exc).__name__})")

    results_df = (
        pd.DataFrame(records).sort_values("aic").reset_index(drop=True)
    )
    if best["fit"] is None:
        raise RuntimeError("No SARIMA model converged over the search grid.")

    if verbose:
        print(f"\nBest model: order={best['order']} "
              f"seasonal_order={best['seasonal_order']} AIC={best['aic']:.1f}")
    return best["fit"], results_df


def fit_sarimax(
    y_train: pd.Series,
    X_train=None,
    order=DEFAULT_ORDER,
    seasonal_order=DEFAULT_SEASONAL_ORDER,
):
    """Fit a single SARIMAX with a given order (optionally with exog)."""
    return _fit_one(y_train, order, seasonal_order, exog=X_train)


def forecast_sarimax(model_fit, horizon: int, X_test=None, index=None,
                     alpha: float = 0.2):
    """
    Forecast ``horizon`` steps ahead.

    Returns just the mean Series (for pipeline compatibility). Use
    ``forecast_sarimax_intervals`` when you also want the interval.
    """
    fc = model_fit.get_forecast(steps=horizon, exog=X_test)
    mean = fc.predicted_mean
    if index is not None:
        mean.index = index
    return mean.rename("sarimax")


def forecast_sarimax_intervals(model_fit, horizon: int, X_test=None,
                               index=None, alpha: float = 0.2):
    """
    Forecast with a (1-alpha) prediction interval. ``alpha=0.2`` gives an 80%
    interval. Returns ``(mean, lower, upper)`` as pandas Series.
    """
    fc = model_fit.get_forecast(steps=horizon, exog=X_test)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=alpha)
    lower = ci.iloc[:, 0]
    upper = ci.iloc[:, 1]
    if index is not None:
        mean.index = index
        lower.index = index
        upper.index = index
    return (mean.rename("mean"),
            lower.rename("lower"),
            upper.rename("upper"))


def get_residuals(model_fit) -> pd.Series:
    """In-sample residuals for diagnostic plots (Part 3)."""
    return pd.Series(model_fit.resid).rename("residuals")
