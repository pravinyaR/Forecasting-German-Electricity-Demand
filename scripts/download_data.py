# scripts/download_data.py
"""Download the raw OPSD 60-minute file into data/raw/."""
from electricity_demand.data import download_raw


def main():
    path = download_raw()
    print(f"Raw data ready at: {path}")


if __name__ == "__main__":
    main()
