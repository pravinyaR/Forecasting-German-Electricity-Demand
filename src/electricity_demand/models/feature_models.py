# src/electricity_demand/models/feature_models.py
"""
Feature-based machine-learning forecasters (Part 5).

The model predicts weekly load from lag, rolling, calendar, holiday and
temperature features. Because the features are all built with ``shift`` (see
``features.py``), a direct (non-recursive) multi-step forecast over the test
window uses only information available before each target week.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from electricity_demand.config import RANDOM_STATE
from electricity_demand import features as feat


def make_ml_table(data: pd.DataFrame, target: str = "load_gw") -> pd.DataFrame:
    """
    Build / return the supervised-learning table.

    If ``data`` already contains lag features (produced by
    ``features.make_feature_table``) it is returned unchanged; otherwise the
    features are constructed here from the target column plus any temperature
    columns present.
    """
    has_lags = any(c.startswith(f"{target}_lag_") for c in data.columns)
    if has_lags:
        return data.dropna().copy()

    temp_cols = [c for c in data.columns
                 if c in ("temp_mean", "temp_min", "temp_max",
                          "heating_degree_days", "cooling_degree_days")]
    temp_weekly = data[temp_cols] if temp_cols else None
    return feat.make_feature_table(data[target], temp_weekly=temp_weekly,
                                   target=target)


def fit_random_forest(X_train, y_train, **kwargs):
    from sklearn.ensemble import RandomForestRegressor

    params = dict(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    params.update(kwargs)
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model


def fit_gradient_boosting(X_train, y_train, **kwargs):
    from sklearn.ensemble import HistGradientBoostingRegressor

    params = dict(random_state=RANDOM_STATE)
    params.update(kwargs)
    model = HistGradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    return model


def predict_feature_model(model, X_test, index=None) -> pd.Series:
    preds = np.asarray(model.predict(X_test), dtype=float)
    idx = index if index is not None else getattr(X_test, "index", None)
    return pd.Series(preds, index=idx, name="feature_model")


def feature_importance(model, feature_names) -> pd.Series:
    """Return a sorted importance Series when the model exposes one."""
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_names)
        return imp.sort_values(ascending=False)
    return pd.Series(dtype=float)
