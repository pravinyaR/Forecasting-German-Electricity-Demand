import numpy as np
import pandas as pd

from electricity_demand.evaluation import mase, rmse, mae, bias, coverage


def test_mase_zero_for_perfect_forecast():
    train = pd.Series(np.sin(np.arange(200) * 0.1) * 5 + 50)
    y_true = pd.Series(np.arange(10, dtype=float))
    assert mase(y_true, y_true, train, seasonality=52) == 0.0


def test_rmse_mae_zero_for_perfect():
    y = np.arange(10, dtype=float)
    assert rmse(y, y) == 0.0
    assert mae(y, y) == 0.0
    assert bias(y, y) == 0.0


def test_coverage_bounds():
    y = np.array([1.0, 2.0, 3.0])
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([2.0, 2.0, 2.0])
    # 1 inside, 2 on boundary (inside), 3 outside -> 2/3
    assert abs(coverage(y, lower, upper) - 2 / 3) < 1e-9
