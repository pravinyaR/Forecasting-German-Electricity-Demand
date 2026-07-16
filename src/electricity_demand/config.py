# src/electricity_demand/config.py
"""
Central configuration for the electricity-demand forecasting project.

Everything that another module might need to agree on (paths, the train/test
split, the seasonal period, grid-search ranges, random seeds) lives here so
there is a single source of truth.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# PROJECT_ROOT resolves to the repository root regardless of where a script
# or notebook is launched from (this file lives at
# <root>/src/electricity_demand/config.py, so go up three parents).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUTS_DIR / "figures"
FORECAST_DIR = OUTPUTS_DIR / "forecasts"
METRICS_DIR = OUTPUTS_DIR / "metrics"
MODEL_OBJECT_DIR = OUTPUTS_DIR / "model_objects"

REPORTS_DIR = PROJECT_ROOT / "reports"

# --------------------------------------------------------------------------
# Raw-data source (Open Power System Data)
# --------------------------------------------------------------------------
OPSD_URL = (
    "https://data.open-power-system-data.org/time_series/"
    "2020-10-06/time_series_60min_singleindex.csv"
)
RAW_FILENAME = "time_series_60min_singleindex.csv"

# Candidate names for the German actual-load column, in order of preference.
# OPSD occasionally renames columns between releases, so we try a few.
DE_LOAD_COLUMN_CANDIDATES = (
    "DE_load_actual_entsoe_transparency",
    "DE_load_actual_entsoe_power_statistics",
    "DE_load_actual",
    "DE_load_",
)
TIMESTAMP_COLUMN_CANDIDATES = ("utc_timestamp", "cet_cest_timestamp")

# --------------------------------------------------------------------------
# Time-window and split
# --------------------------------------------------------------------------
START_DATE = "2015-01-01"          # keep data from here...
END_DATE = None                    # ...to the end of the file (Oct 2020)

TEST_WEEKS = 104                   # two-year hold-out horizon (Part 2/3)
WEEKLY_SEASONALITY = 52            # annual cycle in weekly data
DAILY_SEASONALITY = 7              # weekly cycle in daily data
HOURLY_DAY = 24
HOURLY_WEEK = 24 * 7

# Load is published in MW; we report in GW.
MW_TO_GW = 1.0 / 1000.0

# --------------------------------------------------------------------------
# SARIMA grid search (Part 3)
# --------------------------------------------------------------------------
# The brief requires looping over p in [0, 6], d in [0, 2], q in [0, 6].
P_RANGE = range(0, 7)
D_RANGE = range(0, 3)
Q_RANGE = range(0, 7)

# Seasonal orders are far more expensive, so we search a smaller grid.
SEASONAL_P_RANGE = range(0, 2)
SEASONAL_D_RANGE = range(0, 2)
SEASONAL_Q_RANGE = range(0, 2)

# A sensible default order (README starting point) used by the pipeline when
# a full grid search is not requested.
DEFAULT_ORDER = (1, 1, 1)
DEFAULT_SEASONAL_ORDER = (1, 1, 1, WEEKLY_SEASONALITY)

# --------------------------------------------------------------------------
# Temperature covariate (Part 4) -- Berlin as a representative location
# --------------------------------------------------------------------------
BERLIN_LAT = 52.52
BERLIN_LON = 13.41
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
HEATING_BASE_C = 15.5   # heating-degree base temperature
COOLING_BASE_C = 22.0   # cooling-degree base temperature

# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------
RANDOM_STATE = 0
