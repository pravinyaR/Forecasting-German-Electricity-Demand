# scripts/evaluate_models.py
"""
Regenerate diagnostic figures from saved forecasts: the comparison plot already
comes from the pipeline; here we add per-model error diagnostics.
"""
import pandas as pd
import matplotlib.pyplot as plt

from electricity_demand.config import FORECAST_DIR, FIGURE_DIR


def main():
    fc = pd.read_csv(FORECAST_DIR / "all_forecasts.csv", index_col=0, parse_dates=[0])
    actual = fc["actual"]
    models = [c for c in fc.columns if c != "actual"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for m in models:
        ax.plot(fc.index, fc[m] - actual, label=m, lw=1.0)
    ax.axhline(0, color="grey", ls=":")
    ax.set_title("Forecast errors over the test period")
    ax.set_xlabel("Date")
    ax.set_ylabel("Error (GW)")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "error_diagnostics.png", dpi=300, bbox_inches="tight")
    print(f"Saved {FIGURE_DIR / 'error_diagnostics.png'}")


if __name__ == "__main__":
    main()
