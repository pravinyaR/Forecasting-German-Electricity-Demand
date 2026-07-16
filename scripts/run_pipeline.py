# scripts/run_pipeline.py
from electricity_demand.pipeline import run_pipeline


def main():
    metrics_df, forecast_df = run_pipeline(
        include_sarimax=True,
        include_feature_model=True,
        include_bayesian=False,
        include_neural=False,
    )
    print("\nModel comparison (sorted by MASE)")
    print(metrics_df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
