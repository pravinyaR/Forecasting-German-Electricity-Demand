# scripts/run_sarima_grid.py
"""
Part 3: the full AIC grid search over p in [0,6], d in [0,2], q in [0,6].

A seasonal SARIMA fit on 145 weekly observations with s=52 takes several
seconds, so the full grid (7 x 3 x 7 = 147 non-seasonal orders, multiplied by
the seasonal grid) is expensive. This script is therefore RESUMABLE: every
result is appended to ``outputs/metrics/sarima_aic_full.csv`` and already-tried
orders are skipped on a restart. Run it repeatedly (or with a time budget)
until it reports that the grid is complete.

Usage
-----
    python scripts/run_sarima_grid.py                 # run until done
    python scripts/run_sarima_grid.py --budget 240    # run for 240 seconds
"""

from __future__ import annotations

import argparse
import itertools
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from electricity_demand.data import load_processed_weekly
from electricity_demand.models import sarimax as sx
from electricity_demand.config import (
    METRICS_DIR, TEST_WEEKS, WEEKLY_SEASONALITY, P_RANGE, D_RANGE, Q_RANGE,
)

OUT = METRICS_DIR / "sarima_aic_full.csv"

# The seasonal part is fixed to (1,1,1,52): the EDA shows one clear annual
# cycle, and a seasonal difference is what makes the series stationary. Fixing
# it keeps the (p,d,q) search -- which the brief specifies -- tractable.
SEASONAL_ORDER = (1, 1, 1, WEEKLY_SEASONALITY)


def load_done() -> set:
    if not OUT.exists():
        return set()
    df = pd.read_csv(OUT)
    return set(zip(df.p, df.d, df.q))


def main(budget: float | None = None):
    t0 = time.time()
    y = load_processed_weekly()
    train = y.iloc[:-TEST_WEEKS]

    done = load_done()
    combos = [c for c in itertools.product(P_RANGE, D_RANGE, Q_RANGE)
              if c not in done]
    print(f"{len(done)} already done, {len(combos)} remaining", flush=True)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    header = not OUT.exists()

    for (p, d, q) in combos:
        if budget and (time.time() - t0) > budget:
            print("budget reached; rerun to continue", flush=True)
            break
        t1 = time.time()
        try:
            fit = sx.fit_sarimax(train, order=(p, d, q),
                                 seasonal_order=SEASONAL_ORDER)
            row = {"p": p, "d": d, "q": q, "aic": round(float(fit.aic), 3),
                   "bic": round(float(fit.bic), 3), "sec": round(time.time() - t1, 1),
                   "converged": True}
        except Exception as exc:  # noqa: BLE001
            row = {"p": p, "d": d, "q": q, "aic": np.nan, "bic": np.nan,
                   "sec": round(time.time() - t1, 1), "converged": False}
        pd.DataFrame([row]).to_csv(OUT, mode="a", header=header, index=False)
        header = False
        print(f"({p},{d},{q}) AIC={row['aic']} [{row['sec']}s]", flush=True)

    # report
    if OUT.exists():
        df = pd.read_csv(OUT).sort_values("aic")
        total = len(list(itertools.product(P_RANGE, D_RANGE, Q_RANGE)))
        print(f"\n{len(df)}/{total} orders evaluated "
              f"({int(df.converged.sum())} converged)")
        print("\nTop 10 by AIC:")
        print(df.head(10).to_string(index=False))
        if len(df) == total:
            print("\nGRID COMPLETE")
    return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=None,
                    help="seconds to run before stopping (resumable)")
    args = ap.parse_args()
    main(budget=args.budget)
