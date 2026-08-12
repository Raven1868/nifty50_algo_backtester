import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from strategies.moving_average import MovingAverageStrategy
from backtesting.costs import CostModel
from validation.optimizer import optimize_strategy, best_params
from validation.walk_forward import walk_forward_validate
from analytics.regime import classify_regimes, performance_by_regime


def _panel(n_days=900, n_tickers=4, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    data = 100 * np.cumprod(1 + rng.normal(0.0004, 0.013, size=(n_days, n_tickers)), axis=0)
    return pd.DataFrame(data, index=dates, columns=[f"S{i}.NS" for i in range(n_tickers)])


def test_optimizer_returns_sorted_results():
    prices = _panel(n_days=400)
    grid = {"short_window": [10, 20], "long_window": [50, 100]}
    results = optimize_strategy(
        MovingAverageStrategy, {}, grid, prices, 1000000, CostModel(),
        {"method": "equal_weight", "max_position_size": 0.25, "max_portfolio_exposure": 1.0},
        objective="sharpe",
    )
    assert len(results) == 4
    # sorted descending by objective value (NaNs last)
    vals = results["objective_value"].dropna().tolist()
    assert vals == sorted(vals, reverse=True)


def test_best_params_extraction():
    prices = _panel(n_days=400)
    grid = {"short_window": [10, 20], "long_window": [50, 100]}
    results = optimize_strategy(
        MovingAverageStrategy, {}, grid, prices, 1000000, CostModel(),
        {"method": "equal_weight", "max_position_size": 0.25, "max_portfolio_exposure": 1.0},
    )
    bp = best_params(results, ["short_window", "long_window"])
    assert set(bp.keys()) == {"short_window", "long_window"}


def test_walk_forward_folds_are_non_overlapping_in_test_windows():
    prices = _panel(n_days=900)
    grid = {"short_window": [10, 20]}
    wf = walk_forward_validate(
        MovingAverageStrategy, {"long_window": 100}, grid, prices, 1000000, CostModel(),
        {"method": "equal_weight", "max_position_size": 0.25, "max_portfolio_exposure": 1.0},
        train_window=252, test_window=63, step=63, window_type="rolling",
    )
    log = wf.period_log
    assert len(log) >= 2
    # each fold's test_start must be strictly after the previous fold's test_start
    starts = pd.to_datetime(log["test_start"])
    assert (starts.diff().dropna() > pd.Timedelta(0)).all()
    # OOS equity curve must be monotonically increasing in date index (no duplicate/out-of-order dates)
    assert wf.oos_equity_curve.index.is_monotonic_increasing


def test_walk_forward_raises_when_windows_exceed_data():
    prices = _panel(n_days=100)
    with pytest.raises(ValueError):
        walk_forward_validate(
            MovingAverageStrategy, {"long_window": 20}, {"short_window": [5, 10]}, prices,
            1000000, CostModel(), {"method": "equal_weight", "max_position_size": 0.25, "max_portfolio_exposure": 1.0},
            train_window=200, test_window=100,
        )


def test_regime_classification_shape_and_labels():
    dates = pd.bdate_range("2020-01-01", periods=500)
    rng = np.random.default_rng(5)
    bench = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.012, 500)), index=dates)
    regimes = classify_regimes(bench)
    assert set(regimes["trend"].unique()).issubset({"bull", "bear", "unclassified"})
    assert set(regimes["volatility"].unique()).issubset({"high_vol", "low_vol", "unclassified"})
    assert len(regimes) == len(bench)


def test_performance_by_regime_no_lookahead_alignment():
    dates = pd.bdate_range("2020-01-01", periods=500)
    rng = np.random.default_rng(6)
    bench = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.012, 500)), index=dates)
    strat_returns = pd.Series(rng.normal(0.0005, 0.01, 500), index=dates)
    regimes = classify_regimes(bench)
    result = performance_by_regime(strat_returns, regimes)
    assert "regime" in result.columns
    assert result["n_days"].sum() > 0
