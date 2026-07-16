import numpy as np
import pandas as pd

from electricity_demand.features import (
    add_lag_features, add_rolling_features, make_feature_table,
)


def _weekly(n=160):
    idx = pd.date_range("2015-01-04", periods=n, freq="W")
    return pd.Series(np.arange(n, dtype=float) + 100.0, index=idx, name="load_gw")


def test_lag_features_do_not_use_future():
    df = _weekly().to_frame()
    out = add_lag_features(df, lags=(1,))
    # lag_1 at row t must equal target at row t-1
    assert out["load_gw_lag_1"].iloc[5] == out["load_gw"].iloc[4]
    # and must never equal the current value (strictly past)
    assert out["load_gw_lag_1"].iloc[5] != out["load_gw"].iloc[5]


def test_rolling_uses_only_past():
    df = _weekly().to_frame()
    out = add_rolling_features(df, windows=(4,))
    # rolling mean at t is over values strictly before t (shifted by 1)
    manual = df["load_gw"].shift(1).rolling(4).mean().iloc[10]
    assert abs(out["load_gw_rollmean_4"].iloc[10] - manual) < 1e-9


def test_feature_table_has_no_missing_target():
    table = make_feature_table(_weekly())
    assert table["load_gw"].isna().sum() == 0
    assert len(table) > 0
