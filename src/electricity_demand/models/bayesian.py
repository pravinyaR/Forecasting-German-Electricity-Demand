# src/electricity_demand/models/bayesian.py
"""
Optional Bayesian regression forecaster.

A lightweight, dependency-free (sklearn-only) Bayesian linear model over the
same feature table used by the feature-based model. It gives posterior
predictive intervals for free, which is the point of including it: uncertainty
quantification with interpretable coefficients.

For a fuller treatment (priors, MCMC, posterior diagnostics) this is where a
PyMC/NumPyro model would go; the interface below is deliberately the same as
the other model modules so it slots into the pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_bayesian_ridge(X_train, y_train, **kwargs):
    from sklearn.linear_model import BayesianRidge

    model = BayesianRidge(**kwargs)
    model.fit(np.asarray(X_train, dtype=float), np.asarray(y_train, dtype=float))
    return model


def predict_bayesian(model, X_test, index=None, alpha: float = 0.2):
    """
    Posterior predictive mean and (1-alpha) interval.

    ``BayesianRidge.predict(return_std=True)`` gives the predictive std, from
    which we build a Gaussian interval.
    """
    from scipy.stats import norm

    X = np.asarray(X_test, dtype=float)
    mean, std = model.predict(X, return_std=True)
    z = norm.ppf(1 - alpha / 2)
    lower = mean - z * std
    upper = mean + z * std

    idx = index if index is not None else getattr(X_test, "index", None)
    return (
        pd.Series(mean, index=idx, name="bayesian"),
        pd.Series(lower, index=idx, name="lower"),
        pd.Series(upper, index=idx, name="upper"),
    )
