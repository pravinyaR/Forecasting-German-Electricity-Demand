# scripts/tune_lstm.py
"""
Part 6: LSTM hyperparameter and layer-design search.

Searches over units, number of layers, lookback and dropout. Selection is on
VALIDATION loss (the last 10% of the training windows, chronologically) -- the
test period is never touched during tuning, which is what keeps the final
evaluation honest.

Each configuration is trained for a small number of epochs; this is a coarse
search sized for CPU. The winning configuration is then retrained for longer in
``train_lstm.py``.
"""

from __future__ import annotations

import json
import time
import warnings
import itertools

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from electricity_demand.data import load_processed_hourly
from electricity_demand.models import neural as nn
from electricity_demand.config import METRICS_DIR, TEST_WEEKS

BLOCK = 168        # predict one week at a time (fixed)
STRIDE = 12        # coarser stride during search, for speed
EPOCHS = 3         # short budget per configuration
BATCH = 128

# The search space (layer design + capacity + context length + regularisation)
GRID = {
    "units": [32, 64],
    "n_layers": [1, 2],
    "lookback": [168, 336],
    "dropout": [0.0, 0.2],
}


def main(max_configs: int | None = None):
    hourly = load_processed_hourly()
    test_hours = TEST_WEEKS * 7 * 24
    train_h = hourly.iloc[:-test_hours]

    keys = list(GRID)
    combos = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]
    if max_configs:
        combos = combos[:max_configs]
    print(f"searching {len(combos)} configurations", flush=True)

    rows = []
    for i, cfg in enumerate(combos, 1):
        t0 = time.time()
        try:
            X, y, _ = nn.build_dataset(train_h, lookback=cfg["lookback"],
                                       horizon=BLOCK, stride=STRIDE)
            model = nn.build_lstm(lookback=cfg["lookback"], horizon=BLOCK,
                                  units=cfg["units"], n_layers=cfg["n_layers"],
                                  dropout=cfg["dropout"], lr=1e-3)
            hist = nn.train_lstm(model, X, y, epochs=EPOCHS, batch_size=BATCH,
                                 validation_split=0.1, verbose=0)
            row = dict(cfg)
            row["n_params"] = int(model.count_params())
            row["train_loss"] = round(float(hist.history["loss"][-1]), 5)
            row["val_loss"] = round(float(hist.history["val_loss"][-1]), 5)
            row["sec"] = round(time.time() - t0, 1)
            rows.append(row)
            print(f"[{i}/{len(combos)}] {cfg} -> val_loss={row['val_loss']:.5f} "
                  f"({row['sec']:.0f}s)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(combos)}] {cfg} FAILED {type(exc).__name__}", flush=True)

    res = pd.DataFrame(rows).sort_values("val_loss").reset_index(drop=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(METRICS_DIR / "lstm_tuning.csv", index=False)

    print("\n=== LSTM tuning results (ranked by validation loss) ===")
    print(res.to_string(index=False))

    best = res.iloc[0].to_dict()
    with open(METRICS_DIR / "lstm_best_config.json", "w") as fh:
        json.dump(best, fh, indent=2, default=str)
    print("\nBest configuration:", {k: best[k] for k in keys})
    return res


if __name__ == "__main__":
    main()
