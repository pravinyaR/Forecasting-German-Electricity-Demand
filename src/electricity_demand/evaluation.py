# src/electricity_demand/evaluation.py
"""
Forecast evaluation metrics (used across Parts 2-6).

MASE is scaled by the in-sample seasonal-naive error, which makes it directly
comparable across series and is the headline metric the brief asks us to rank
models by. A MASE < 1 means the model beats a seasonal-naive forecast on the
training scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from electricity_demand.config import WEEKLY_SEASONALITY


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def bias(y_true, y_pred) -> float:
    """Mean signed error (positive => forecasts run high)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(y_pred - y_true))


def mase(y_true, y_pred, y_train, seasonality: int = WEEKLY_SEASONALITY) -> float:
    """
    Mean Absolute Scaled Error.

    Denominator is the mean absolute seasonal-naive error on the TRAINING set,
    so the scale is fixed independently of the test period.
    """
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= seasonality:
        raise ValueError(
            f"Training series too short ({len(y_train)}) for seasonality "
            f"{seasonality}."
        )
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = naive_errors.mean()
    if scale == 0:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float)
                                - np.asarray(y_pred, dtype=float))) / scale)


def coverage(y_true, lower, upper) -> float:
    """Empirical coverage: fraction of actuals inside [lower, upper]."""
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def evaluate_forecast(name, y_true, y_pred, y_train,
                      seasonality: int = WEEKLY_SEASONALITY) -> dict:
    """Return a one-row metrics dict for a single model's forecast."""
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(np.asarray(y_pred, dtype=float), index=y_true.index)
    return {
        "model": name,
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MASE": mase(y_true, y_pred, y_train, seasonality=seasonality),
        "Bias": bias(y_true, y_pred),
    }


def comparison_table(records: list[dict], sort_by: str = "MASE") -> pd.DataFrame:
    """Stack per-model dicts into a sorted comparison table."""
    df = pd.DataFrame(records)
    if sort_by in df.columns:
        df = df.sort_values(sort_by).reset_index(drop=True)
    return df
