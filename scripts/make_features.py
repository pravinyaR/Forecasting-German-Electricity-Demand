# scripts/make_features.py
"""
Build the processed modelling datasets (Part 1) and the weekly feature table
(Parts 4-5). Temperature is fetched from Open-Meteo if network access allows;
if it fails, the weekly feature table is still built without temperature.
"""
import pandas as pd

from electricity_demand.config import PROCESSED_DATA_DIR
from electricity_demand.data import build_processed
from electricity_demand import features as feat


def main(with_temperature: bool = True):
    series = build_processed(save=True)
    weekly = series["weekly"]

    temp_weekly = None
    if with_temperature:
        try:
            temp_daily = feat.get_open_meteo_temperature(
                start_date=str(weekly.index.min().date()),
                end_date=str(weekly.index.max().date()),
            )
            temp_weekly = feat.weekly_temperature_features(temp_daily, weekly.index)
            print("Fetched Berlin temperature features.")
        except Exception as exc:  # noqa: BLE001
            print(f"Temperature fetch failed ({exc}); continuing without it.")

    table = feat.make_feature_table(weekly, temp_weekly=temp_weekly)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DATA_DIR / "weekly_features.csv"
    table.to_csv(out)
    print(f"Saved feature table ({table.shape[0]} rows, {table.shape[1]} cols) -> {out}")


if __name__ == "__main__":
    main()
