# src/electricity_demand/plotting.py
"""
Reusable plotting functions. Every function returns a Matplotlib Figure so the
caller decides whether to show, save, or embed it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_series(series: pd.Series, title: str = "", ylabel: str = "Load (GW)"):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(series.index, series.values, lw=1.0)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def plot_decomposition(stl_result):
    """Plot an STL decomposition result (statsmodels DecomposeResult)."""
    fig = stl_result.plot()
    fig.set_size_inches(11, 8)
    fig.tight_layout()
    return fig


def plot_acf_pacf(series: pd.Series, lags: int = 60):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], method="ywm")
    axes[0].set_title("Autocorrelation (ACF)")
    axes[1].set_title("Partial autocorrelation (PACF)")
    fig.tight_layout()
    return fig


def plot_forecasts(train: pd.Series, test: pd.Series,
                   forecasts: dict[str, pd.Series],
                   title: str = "Forecast comparison",
                   context_periods: int | None = 156):
    """
    Overlay every model's forecast on the actuals. ``context_periods`` limits
    how much history is shown before the forecast origin (keeps the plot legible).
    """
    fig, ax = plt.subplots(figsize=(12, 5.5))

    hist = train.iloc[-context_periods:] if context_periods else train
    ax.plot(hist.index, hist.values, color="black", lw=1.2, label="Train")
    ax.plot(test.index, test.values, color="tab:blue", lw=1.6, label="Actual (test)")

    for name, pred in forecasts.items():
        ax.plot(pred.index, pred.values, lw=1.2, alpha=0.9, label=name)

    ax.axvline(train.index[-1], color="grey", ls=":", label="Forecast origin")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Load (GW)")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    return fig


def plot_prediction_interval(train, test, mean, lower, upper,
                             name: str = "SARIMA",
                             context_periods: int | None = 156):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    hist = train.iloc[-context_periods:] if context_periods else train
    ax.plot(hist.index, hist.values, color="black", lw=1.2, label="Train")
    ax.plot(test.index, test.values, color="tab:blue", lw=1.6, label="Actual")
    ax.plot(mean.index, mean.values, color="tab:red", lw=1.4, label=f"{name} mean")
    ax.fill_between(mean.index, lower.values, upper.values,
                    color="tab:red", alpha=0.2, label=f"{name} interval")
    ax.axvline(train.index[-1], color="grey", ls=":")
    ax.set_title(f"{name} forecast with prediction interval")
    ax.set_xlabel("Date")
    ax.set_ylabel("Load (GW)")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    return fig


def plot_residual_diagnostics(residuals: pd.Series, lags: int = 60):
    """
    Four-panel residual check for Part 3: residuals over time, histogram,
    ACF, and a normal Q-Q plot.
    """
    import scipy.stats as stats
    from statsmodels.graphics.tsaplots import plot_acf

    resid = pd.Series(residuals).dropna()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(resid.index, resid.values, lw=0.9)
    axes[0, 0].axhline(0, color="grey", ls=":")
    axes[0, 0].set_title("Residuals over time")

    axes[0, 1].hist(resid.values, bins=30, density=True, alpha=0.8)
    axes[0, 1].set_title("Residual distribution")

    plot_acf(resid, lags=lags, ax=axes[1, 0])
    axes[1, 0].set_title("Residual ACF")

    stats.probplot(resid.values, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Normal Q-Q plot")

    fig.tight_layout()
    return fig
