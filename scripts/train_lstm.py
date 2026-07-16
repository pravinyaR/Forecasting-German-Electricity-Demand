# scripts/train_lstm.py
"""
Part 6: train an LSTM on the hourly German load series and forecast the final
two years.

Design choices (all discussed in the report):
* Direct multi-step: the network predicts a 168-hour (one week) block in one
  shot, rather than a single hour. Rolling that block forward 104 times covers
  the two-year horizon while compounding error once per week instead of once
  per hour.
* lookback = 336 hours (two weeks) so the network sees both the daily cycle and
  the weekday/weekend pattern.
* Windows are strided to keep training tractable on CPU.
* The MinMax scaler is fitted on the TRAINING hours only -- no leakage.
"""

from __future__ import annotations

import json
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from electricity_demand.data import load_processed_hourly
from electricity_demand.models import neural as nn
from electricity_demand.models import benchmarks as bm
from electricity_demand import evaluation as ev
from electricity_demand.config import (
    FIGURE_DIR, METRICS_DIR, FORECAST_DIR, MODEL_OBJECT_DIR, TEST_WEEKS,
)

LOOKBACK = 336        # 2 weeks of history (selected by scripts/tune_lstm.py)
BLOCK = 168           # predict 1 week at a time
STRIDE = 6            # subsample window starts
UNITS = 64            # tuned
N_LAYERS = 2          # tuned
DROPOUT = 0.0         # tuned
EPOCHS = 10           # tuned config, longer final fit
BATCH = 128


def main():
    t_start = time.time()

    # ---- data ----------------------------------------------------------
    hourly = load_processed_hourly()
    test_hours = TEST_WEEKS * 7 * 24
    train_h, test_h = hourly.iloc[:-test_hours], hourly.iloc[-test_hours:]
    print(f"train {len(train_h)} h | test {len(test_h)} h", flush=True)

    # ---- windows (scaler fitted on train only) -------------------------
    X, y, scaler = nn.build_dataset(train_h, lookback=LOOKBACK,
                                    horizon=BLOCK, stride=STRIDE)
    print(f"windows: X={X.shape} y={y.shape}", flush=True)

    # ---- model ---------------------------------------------------------
    model = nn.build_lstm(lookback=LOOKBACK, horizon=BLOCK, units=UNITS,
                          n_layers=N_LAYERS, dropout=DROPOUT, lr=1e-3)
    model.summary()

    hist = nn.train_lstm(model, X, y, epochs=EPOCHS, batch_size=BATCH,
                         validation_split=0.1, verbose=2)
    print(f"trained in {time.time()-t_start:.0f}s", flush=True)

    # ---- forecast the full two years -----------------------------------
    fc_hourly = nn.rolling_block_forecast(
        model, train_h, scaler, lookback=LOOKBACK, block=BLOCK,
        horizon=len(test_h), index=test_h.index,
    )
    print("hourly forecast done", flush=True)

    # ---- evaluate on hourly and on weekly means ------------------------
    rows = [ev.evaluate_forecast("lstm (hourly)", test_h, fc_hourly, train_h,
                                 seasonality=24 * 7)]

    fc_weekly = nn.to_weekly(fc_hourly)
    actual_weekly = test_h.resample("W").mean()
    train_weekly = train_h.resample("W").mean().dropna()
    common = fc_weekly.index.intersection(actual_weekly.index)
    fc_weekly, actual_weekly = fc_weekly.loc[common], actual_weekly.loc[common]

    rows.append(ev.evaluate_forecast("lstm (weekly agg)", actual_weekly,
                                     fc_weekly, train_weekly, seasonality=52))
    sn = bm.seasonal_naive_forecast(train_weekly, len(actual_weekly), 52,
                                    actual_weekly.index)
    rows.append(ev.evaluate_forecast("seasonal_naive (weekly)", actual_weekly,
                                     sn, train_weekly, seasonality=52))

    res = pd.DataFrame(rows).round(4)
    print(res.to_string(index=False), flush=True)

    # ---- save ----------------------------------------------------------
    for d in (FIGURE_DIR, METRICS_DIR, FORECAST_DIR, MODEL_OBJECT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    res.to_csv(METRICS_DIR / "lstm_metrics.csv", index=False)
    fc_hourly.to_frame("lstm_hourly").to_csv(FORECAST_DIR / "lstm_hourly_forecast.csv")
    fc_weekly.to_frame("lstm_weekly").to_csv(FORECAST_DIR / "lstm_weekly_forecast.csv")
    model.save(MODEL_OBJECT_DIR / "lstm_model.keras")

    with open(METRICS_DIR / "lstm_config.json", "w") as fh:
        json.dump({"lookback": LOOKBACK, "block": BLOCK, "stride": STRIDE,
                   "units": UNITS, "n_layers": N_LAYERS, "dropout": DROPOUT,
                   "epochs": EPOCHS, "batch": BATCH,
                   "final_loss": float(hist.history["loss"][-1]),
                   "final_val_loss": float(hist.history["val_loss"][-1]),
                   "runtime_s": round(time.time() - t_start, 1)}, fh, indent=2)

    # ---- figures -------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(hist.history["loss"], label="train")
    ax.plot(hist.history["val_loss"], label="val")
    ax.set_title("LSTM training curve"); ax.set_xlabel("epoch"); ax.set_ylabel("MSE")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIGURE_DIR / "lstm_training_curve.png", dpi=150, bbox_inches="tight")

    # first two weeks of the horizon: does it capture daily/weekly shape?
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(test_h.index[:336], test_h.values[:336], label="Actual", lw=1.2)
    ax.plot(fc_hourly.index[:336], fc_hourly.values[:336], label="LSTM", lw=1.2)
    ax.set_title("LSTM hourly forecast: first two weeks of the horizon")
    ax.set_xlabel("Date"); ax.set_ylabel("Load (GW)"); ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "lstm_first_two_weeks.png", dpi=150, bbox_inches="tight")

    # weekly aggregate over the whole two years
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train_weekly.index[-104:], train_weekly.values[-104:], color="black",
            lw=1.0, label="Train (weekly)")
    ax.plot(actual_weekly.index, actual_weekly.values, lw=1.5, label="Actual")
    ax.plot(fc_weekly.index, fc_weekly.values, lw=1.3, label="LSTM (weekly agg)")
    ax.plot(sn.index, sn.values, lw=1.0, ls="--", label="Seasonal naive")
    ax.axvline(train_weekly.index[-1], color="grey", ls=":")
    ax.set_title("LSTM two-year forecast aggregated to weekly means")
    ax.set_xlabel("Date"); ax.set_ylabel("Load (GW)"); ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "lstm_weekly_forecast.png", dpi=150, bbox_inches="tight")

    print(f"TOTAL {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
