# src/electricity_demand/data.py
"""
Data acquisition and preparation (Part 1).

Responsibilities
----------------
1. Download the raw 60-minute OPSD file (or read a local copy).
2. Extract and clean the German actual-load series, convert MW -> GW.
3. Clip to the study window (2015-01-01 -> end of file, Oct 2020).
4. Aggregate to daily and weekly average load.
5. Persist processed series and reload them for modelling.

Design notes
------------
* Timestamps in OPSD are UTC. We convert to Europe/Berlin local time before
  resampling so that weekly/daily bins line up with the German calendar (and
  with the temperature covariate, which is fetched on Berlin local dates).
* The load column name has changed across OPSD releases, so we detect it.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from electricity_demand.config import (
    OPSD_URL,
    RAW_DATA_DIR,
    RAW_FILENAME,
    PROCESSED_DATA_DIR,
    DE_LOAD_COLUMN_CANDIDATES,
    TIMESTAMP_COLUMN_CANDIDATES,
    START_DATE,
    END_DATE,
    MW_TO_GW,
)


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def download_raw(url: str = OPSD_URL, dest: Path | None = None,
                 overwrite: bool = False) -> Path:
    """
    Download the raw OPSD 60-minute CSV to ``data/raw/``.

    Requires network access to ``data.open-power-system-data.org``. If the file
    already exists it is reused unless ``overwrite=True``.
    """
    dest = dest or (RAW_DATA_DIR / RAW_FILENAME)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not overwrite:
        print(f"Raw file already present: {dest}")
        return dest

    import requests  # local import so the package imports without requests

    print(f"Downloading {url}\n  -> {dest}")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    print("Download complete.")
    return dest


# --------------------------------------------------------------------------
# Column detection helpers
# --------------------------------------------------------------------------
def _find_column(columns, candidates, kind: str) -> str:
    cols = list(columns)
    # exact match first
    for cand in candidates:
        if cand in cols:
            return cand
    # then a substring / prefix match
    for cand in candidates:
        for col in cols:
            if col.startswith(cand) or cand in col:
                return col
    raise KeyError(
        f"Could not locate the {kind} column. Tried {candidates!r}. "
        f"Available columns (first 40): {cols[:40]}"
    )


# --------------------------------------------------------------------------
# Load + clean the hourly series
# --------------------------------------------------------------------------
def load_raw_60min(path: Path | None = None) -> pd.DataFrame:
    """Read the raw OPSD CSV and return it with a UTC DatetimeIndex."""
    path = path or (RAW_DATA_DIR / RAW_FILENAME)
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Raw file not found at {path}. Run `download_data.py` first, "
            "or place the OPSD 60-minute CSV there manually."
        )

    ts_col = None
    header = pd.read_csv(path, nrows=0).columns
    ts_col = _find_column(header, TIMESTAMP_COLUMN_CANDIDATES, "timestamp")

    df = pd.read_csv(path, parse_dates=[ts_col])
    df = df.set_index(ts_col).sort_index()

    # Ensure tz-aware UTC index.
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def clean_hourly_load(
    df_raw: pd.DataFrame | None = None,
    start_date: str = START_DATE,
    end_date: str | None = END_DATE,
) -> pd.Series:
    """
    Extract the German actual-load series in GW on a clean hourly grid.

    Steps: pick the DE load column, convert to Berlin local time, clip to the
    study window, reindex onto a regular hourly grid, interpolate short gaps.
    Returns a Series named ``load_gw``.
    """
    if df_raw is None:
        df_raw = load_raw_60min()

    load_col = _find_column(df_raw.columns, DE_LOAD_COLUMN_CANDIDATES, "DE load")
    load = df_raw[load_col].astype(float) * MW_TO_GW

    # UTC -> Berlin local, then drop tz so calendar aggregation is intuitive.
    load = load.tz_convert("Europe/Berlin").tz_localize(None)
    load = load.rename("load_gw")

    # Clip to the study window.
    load = load.loc[start_date:] if start_date else load
    load = load.loc[:end_date] if end_date else load

    # The autumn DST fall-back repeats one local hour each year, producing
    # duplicate local timestamps. Collapse duplicates (mean) before reindexing.
    load = load.groupby(level=0).mean().sort_index()

    # Regular hourly grid + gap handling.
    full_index = pd.date_range(load.index.min(), load.index.max(), freq="h")
    load = load.reindex(full_index)
    n_missing = int(load.isna().sum())
    if n_missing:
        warnings.warn(f"Interpolating {n_missing} missing hourly load values.")
        load = load.interpolate(method="time").ffill().bfill()

    load.index.name = "timestamp"
    return load.rename("load_gw")


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def aggregate_to_daily(hourly: pd.Series) -> pd.Series:
    """Daily *average* load in GW (weekly seasonality shows up here)."""
    daily = hourly.resample("D").mean().rename("load_gw")
    daily.index.name = "date"
    return daily


def aggregate_to_weekly(hourly: pd.Series) -> pd.Series:
    """Weekly *average* load in GW (annual seasonality shows up here)."""
    weekly = hourly.resample("W").mean().rename("load_gw")
    weekly.index.name = "date"
    # Drop partial weeks at the ends that can bias the series.
    return weekly.dropna()


# --------------------------------------------------------------------------
# Build + persist the processed datasets
# --------------------------------------------------------------------------
def build_processed(
    path: Path | None = None,
    save: bool = True,
    start_date: str = START_DATE,
    end_date: str | None = END_DATE,
) -> dict[str, pd.Series]:
    """
    Full Part-1 preparation: raw -> hourly/daily/weekly, optionally saved to
    ``data/processed/``. Returns a dict of the three series.
    """
    hourly = clean_hourly_load(load_raw_60min(path), start_date, end_date)
    daily = aggregate_to_daily(hourly)
    weekly = aggregate_to_weekly(hourly)

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        hourly.to_frame().to_csv(PROCESSED_DATA_DIR / "hourly.csv")
        daily.to_frame().to_csv(PROCESSED_DATA_DIR / "daily.csv")
        weekly.to_frame().to_csv(PROCESSED_DATA_DIR / "weekly.csv")
        print(f"Saved processed series to {PROCESSED_DATA_DIR}")

    return {"hourly": hourly, "daily": daily, "weekly": weekly}


def _read_series(path: Path, value_col: str = "load_gw") -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=[0])
    return df[value_col]


def load_processed_hourly() -> pd.Series:
    return _read_series(PROCESSED_DATA_DIR / "hourly.csv")


def load_processed_daily() -> pd.Series:
    return _read_series(PROCESSED_DATA_DIR / "daily.csv")


def load_processed_weekly() -> pd.Series:
    return _read_series(PROCESSED_DATA_DIR / "weekly.csv")


def load_processed_data() -> pd.DataFrame:
    """
    Return the weekly modelling table used by the pipeline.

    If a feature-enriched table (``weekly_features.csv``, produced by
    ``make_features.py``) exists it is used; otherwise the bare weekly load
    series is returned as a one-column DataFrame. Either way the target column
    is ``load_gw``.
    """
    feat_path = PROCESSED_DATA_DIR / "weekly_features.csv"
    if feat_path.exists():
        df = pd.read_csv(feat_path, index_col=0, parse_dates=[0])
        if "load_gw" not in df.columns:
            raise KeyError("weekly_features.csv is missing the 'load_gw' column.")
        return df

    weekly_path = PROCESSED_DATA_DIR / "weekly.csv"
    if not weekly_path.exists():
        raise FileNotFoundError(
            "No processed weekly data found. Run `python scripts/make_features.py` "
            "(or electricity_demand.data.build_processed) first."
        )
    return load_processed_weekly().to_frame("load_gw")
