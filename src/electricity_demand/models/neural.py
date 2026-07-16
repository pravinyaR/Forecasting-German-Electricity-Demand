# src/electricity_demand/models/neural.py
"""
LSTM forecaster for the hourly series (Part 6).

TensorFlow/Keras is imported lazily so the rest of the package works without a
deep-learning stack installed.

Honest note on the horizon
--------------------------
The brief asks for a two-year forecast. Two years of hourly data is ~17,500
steps. A single-step LSTM rolled out recursively that far will accumulate error
and drift badly -- this is a genuine limitation worth discussing in the report,
not a bug. Three sensible framings are provided:

* ``recursive_forecast``  : classic seq-to-one rolled forward (use for short
  horizons / illustration of drift).
* ``direct_multistep``    : predict a block of ``H`` future hours at once
  (seq-to-seq head), which avoids compounding error.
* aggregate the hourly forecast to weekly with ``to_weekly`` so it is
  comparable to the other models on the common metric table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from electricity_demand.config import RANDOM_STATE


# --------------------------------------------------------------------------
# Windowing (leakage-safe: scaler is fitted on train only, see build_dataset)
# --------------------------------------------------------------------------
def make_windows(values: np.ndarray, lookback: int, horizon: int = 1,
                 stride: int = 1):
    """
    Turn a 1-D array into (X, y) supervised windows.

    ``stride`` subsamples the window start points. With hourly data the windows
    overlap almost completely, so a stride of e.g. 4-24 cuts training cost
    dramatically with little information loss.
    """
    X, y = [], []
    last = len(values) - lookback - horizon + 1
    for i in range(0, last, stride):
        X.append(values[i: i + lookback])
        y.append(values[i + lookback: i + lookback + horizon])
    X = np.asarray(X)
    y = np.asarray(y)
    return X[..., None], y  # X shape (n, lookback, 1)


def build_dataset(train: pd.Series, lookback: int = 168, horizon: int = 1,
                  stride: int = 1):
    """
    Scale on the TRAIN series only, then window it. Returns scaled windows and
    the fitted scaler (needed to invert forecasts).

    Fitting the scaler here -- on the training series alone -- is what keeps the
    test period out of the preprocessing, i.e. no leakage.
    """
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_scaled = scaler.fit_transform(
        np.asarray(train, dtype=float).reshape(-1, 1)
    ).ravel()
    X, y = make_windows(train_scaled, lookback, horizon, stride=stride)
    return X, y, scaler


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_lstm(lookback: int, horizon: int = 1, units: int = 64,
               n_layers: int = 1, dropout: float = 0.0, lr: float = 1e-3):
    """Build (but do not train) a Keras LSTM. Lazy TF import."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    tf.random.set_seed(RANDOM_STATE)
    model = Sequential()
    for layer in range(n_layers):
        return_seq = layer < n_layers - 1
        kwargs = dict(units=units, return_sequences=return_seq)
        if layer == 0:
            kwargs["input_shape"] = (lookback, 1)
        model.add(LSTM(**kwargs))
        if dropout:
            model.add(Dropout(dropout))
    model.add(Dense(horizon))
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr))
    return model


def train_lstm(model, X, y, epochs: int = 20, batch_size: int = 64,
               validation_split: float = 0.1, verbose: int = 1):
    y2 = y.reshape(y.shape[0], -1)
    history = model.fit(
        X, y2, epochs=epochs, batch_size=batch_size,
        validation_split=validation_split, shuffle=False, verbose=verbose,
    )
    return history


# --------------------------------------------------------------------------
# Forecasting
# --------------------------------------------------------------------------
def recursive_forecast(model, train: pd.Series, scaler, lookback: int,
                       horizon: int, index=None) -> pd.Series:
    """One-step model rolled forward ``horizon`` times (feeds its own outputs)."""
    scaled = scaler.transform(
        np.asarray(train, dtype=float).reshape(-1, 1)
    ).ravel()
    window = list(scaled[-lookback:])
    preds_scaled = []
    for _ in range(horizon):
        x = np.asarray(window[-lookback:]).reshape(1, lookback, 1)
        yhat = float(model.predict(x, verbose=0).ravel()[0])
        preds_scaled.append(yhat)
        window.append(yhat)
    preds = scaler.inverse_transform(
        np.asarray(preds_scaled).reshape(-1, 1)
    ).ravel()
    return pd.Series(preds, index=index, name="lstm")


def direct_multistep(model, train: pd.Series, scaler, lookback: int,
                     index=None) -> pd.Series:
    """Predict the whole horizon block in one shot (model has H outputs)."""
    scaled = scaler.transform(
        np.asarray(train, dtype=float).reshape(-1, 1)
    ).ravel()
    x = scaled[-lookback:].reshape(1, lookback, 1)
    preds_scaled = model.predict(x, verbose=0).ravel()
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    return pd.Series(preds, index=index, name="lstm")


def rolling_block_forecast(model, train: pd.Series, scaler, lookback: int,
                           block: int, horizon: int, index=None) -> pd.Series:
    """
    Long-horizon forecast for a seq-to-seq model with ``block`` outputs.

    The model predicts a whole block (e.g. 168 hours = 1 week) at once; the
    block is appended to the context and the model is called again, until
    ``horizon`` steps are covered. This needs horizon/block predict calls
    instead of ``horizon`` of them, and error compounds once per block rather
    than once per hour -- far better behaved than a step-by-step rollout,
    though drift over a two-year horizon is still unavoidable.
    """
    scaled = scaler.transform(
        np.asarray(train, dtype=float).reshape(-1, 1)
    ).ravel()
    context = list(scaled[-lookback:])
    preds_scaled: list[float] = []

    n_blocks = int(np.ceil(horizon / block))
    for _ in range(n_blocks):
        x = np.asarray(context[-lookback:]).reshape(1, lookback, 1)
        yhat = model.predict(x, verbose=0).ravel()[:block]
        preds_scaled.extend(yhat.tolist())
        context.extend(yhat.tolist())

    preds = scaler.inverse_transform(
        np.asarray(preds_scaled[:horizon]).reshape(-1, 1)
    ).ravel()
    return pd.Series(preds, index=index, name="lstm")


def to_weekly(hourly_forecast: pd.Series) -> pd.Series:
    """Aggregate an hourly LSTM forecast to weekly means for comparison."""
    return hourly_forecast.resample("W").mean().rename("lstm_weekly")
