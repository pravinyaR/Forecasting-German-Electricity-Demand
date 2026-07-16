# src/electricity_demand/pipeline.py
"""
End-to-end weekly forecasting workflow.

    metrics_df, forecast_df = run_pipeline()

Steps: load processed weekly data -> train/test split -> benchmark models ->
optional SARIMAX / feature / Bayesian / neural models -> evaluate -> save
forecasts, metrics and a comparison figure.
"""

from __future__ import annotations

import pandas as pd

from electricity_demand.config import (
    FORECAST_DIR, METRICS_DIR, FIGURE_DIR, MODEL_OBJECT_DIR,
    TEST_WEEKS, WEEKLY_SEASONALITY,
)
from electricity_demand.data import load_processed_data
from electricity_demand.evaluation import evaluate_forecast
from electricity_demand.plotting import plot_forecasts

from electricity_demand.models.benchmarks import (
    mean_forecast, naive_forecast, seasonal_naive_forecast, drift_forecast,
)
from electricity_demand.models.sarimax import fit_sarimax, forecast_sarimax
from electricity_demand.models.feature_models import (
    make_ml_table, fit_gradient_boosting, predict_feature_model,
)

EXOG_CANDIDATES = [
    "temp_mean", "heating_degree_days", "cooling_degree_days",
    "holiday_days", "has_holiday",
]


def run_pipeline(
    test_weeks: int = TEST_WEEKS,
    include_sarimax: bool = True,
    include_feature_model: bool = True,
    include_bayesian: bool = False,
    include_neural: bool = False,
    sarimax_order=None,
    sarimax_seasonal_order=None,
):
    """Run the workflow and return ``(metrics_df, forecast_df)``."""
    # 1. Load + split -----------------------------------------------------
    data = load_processed_data()
    y = data["load_gw"]
    train, test = y.iloc[:-test_weeks], y.iloc[-test_weeks:]
    horizon = len(test)

    forecasts: dict[str, pd.Series] = {}

    # 2. Benchmarks -------------------------------------------------------
    forecasts["mean"] = mean_forecast(train, horizon, index=test.index)
    forecasts["naive"] = naive_forecast(train, horizon, index=test.index)
    forecasts["seasonal_naive"] = seasonal_naive_forecast(
        train, horizon, seasonality=WEEKLY_SEASONALITY, index=test.index)
    forecasts["drift"] = drift_forecast(train, horizon, index=test.index)

    # 3. SARIMAX ----------------------------------------------------------
    if include_sarimax:
        exog_cols = [c for c in EXOG_CANDIDATES if c in data.columns]
        if exog_cols:
            X = data[exog_cols]
            X_train, X_test = X.iloc[:-test_weeks], X.iloc[-test_weeks:]
        else:
            X_train = X_test = None

        kwargs = {}
        if sarimax_order:
            kwargs["order"] = sarimax_order
        if sarimax_seasonal_order:
            kwargs["seasonal_order"] = sarimax_seasonal_order

        sarimax_fit = fit_sarimax(y_train=train, X_train=X_train, **kwargs)
        forecasts["sarimax"] = forecast_sarimax(
            model_fit=sarimax_fit, horizon=horizon,
            X_test=X_test, index=test.index)

    # 4. Feature-based ML -------------------------------------------------
    if include_feature_model:
        ml = make_ml_table(data)
        ml_train, ml_test = ml.iloc[:-test_weeks], ml.iloc[-test_weeks:]
        target = "load_gw"
        feat_cols = [c for c in ml.columns if c != target]

        model = fit_gradient_boosting(ml_train[feat_cols], ml_train[target])
        forecasts["feature_model"] = predict_feature_model(
            model, ml_test[feat_cols], index=ml_test.index)

    # 5. Optional Bayesian / neural --------------------------------------
    if include_bayesian:
        from electricity_demand.models.bayesian import (
            fit_bayesian_ridge, predict_bayesian)
        ml = make_ml_table(data)
        ml_train, ml_test = ml.iloc[:-test_weeks], ml.iloc[-test_weeks:]
        feat_cols = [c for c in ml.columns if c != "load_gw"]
        bmodel = fit_bayesian_ridge(ml_train[feat_cols], ml_train["load_gw"])
        bmean, _, _ = predict_bayesian(bmodel, ml_test[feat_cols],
                                       index=ml_test.index)
        forecasts["bayesian"] = bmean

    if include_neural:
        raise NotImplementedError(
            "Neural forecasting runs on the HOURLY series and is heavy; drive "
            "it from scripts/ or a notebook using electricity_demand.models.neural."
        )

    # 6. Evaluate ---------------------------------------------------------
    metrics = [
        evaluate_forecast(name=name, y_true=test,
                          y_pred=pred.reindex(test.index), y_train=train)
        for name, pred in forecasts.items()
    ]
    metrics_df = pd.DataFrame(metrics).sort_values("MASE").reset_index(drop=True)

    # 7. Save -------------------------------------------------------------
    for d in (FORECAST_DIR, METRICS_DIR, FIGURE_DIR, MODEL_OBJECT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    forecast_df = pd.DataFrame({"actual": test})
    for name, pred in forecasts.items():
        forecast_df[name] = pred.reindex(test.index)

    forecast_df.to_csv(FORECAST_DIR / "all_forecasts.csv")
    metrics_df.to_csv(METRICS_DIR / "model_comparison.csv", index=False)

    fig = plot_forecasts(train=train, test=test, forecasts=forecasts)
    fig.savefig(FIGURE_DIR / "forecast_comparison.png", dpi=300,
                bbox_inches="tight")

    return metrics_df, forecast_df
