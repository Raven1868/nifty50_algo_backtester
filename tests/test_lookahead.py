"""
Lookahead-bias regression test.

Method: run each strategy on the original price panel, then run it again on
a modified panel where prices AFTER a cutoff date have been altered
(replaced with random noise). If the strategy has no lookahead bias, its
signals BEFORE the cutoff date must be IDENTICAL in both runs — a signal at
date t can only depend on prices up to and including t, so changing prices
after t must not change the signal at t.

This is an effective sanity check because it does not require understanding
the strategy's internals — it directly tests the causal/informational
boundary that defines lookahead bias.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from strategies.mean_reversion import MeanReversionStrategy
from strategies.moving_average import MovingAverageStrategy
from strategies.momentum import MomentumStrategy
from backtesting.engine import BacktestEngine
from backtesting.costs import CostModel


def make_price_panel(n_days=300, n_tickers=5, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    tickers = [f"STK{i}.NS" for i in range(n_tickers)]
    prices = 100 * np.cumprod(1 + rng.normal(0.0004, 0.015, size=(n_days, n_tickers)), axis=0)
    return pd.DataFrame(prices, index=dates, columns=tickers)


@pytest.mark.parametrize("strategy_cls,params", [
    (MeanReversionStrategy, {"lookback": 20, "entry_z": -1.5, "exit_z": -0.3}),
    (MovingAverageStrategy, {"short_window": 10, "long_window": 40}),
])
def test_no_lookahead_in_signal_generation(strategy_cls, params):
    prices = make_price_panel()
    cutoff_idx = 200
    cutoff_date = prices.index[cutoff_idx]

    strategy = strategy_cls(params)
    signal_original = strategy.generate_signals(prices)

    prices_altered = prices.copy()
    rng = np.random.default_rng(999)
    n_after = len(prices) - cutoff_idx
    prices_altered.iloc[cutoff_idx:] = 100 * np.cumprod(
        1 + rng.normal(0.02, 0.05, size=(n_after, prices.shape[1])), axis=0
    )

    signal_altered = strategy_cls(params).generate_signals(prices_altered)

    before = signal_original.loc[:cutoff_date].iloc[:-1]  # strictly before cutoff
    before_altered = signal_altered.loc[:cutoff_date].iloc[:-1]

    pd.testing.assert_frame_equal(before, before_altered,
        obj=f"{strategy_cls.__name__} signal before cutoff changed after future prices were altered — LOOKAHEAD BIAS DETECTED")


def test_momentum_no_lookahead():
    prices = make_price_panel(n_days=400)
    cutoff_idx = 300
    cutoff_date = prices.index[cutoff_idx]
    params = {"lookback": 60, "top_n": 2, "skip_recent_days": 5}

    strategy = MomentumStrategy(params, rebalance_freq="W")
    signal_original = strategy.generate_signals(prices)

    prices_altered = prices.copy()
    rng = np.random.default_rng(123)
    n_after = len(prices) - cutoff_idx
    prices_altered.iloc[cutoff_idx:] = 100 * np.cumprod(
        1 + rng.normal(0.02, 0.05, size=(n_after, prices.shape[1])), axis=0
    )
    signal_altered = MomentumStrategy(params, rebalance_freq="W").generate_signals(prices_altered)

    before = signal_original.loc[:cutoff_date].iloc[:-1]
    before_altered = signal_altered.loc[:cutoff_date].iloc[:-1]
    pd.testing.assert_frame_equal(before, before_altered,
        obj="Momentum signal before cutoff changed after future prices altered — LOOKAHEAD BIAS DETECTED")


def test_engine_shifts_signal_before_execution():
    """Verify the engine's execution weight on day t equals the target
    weight computed from information available at day t-1, i.e. the engine
    itself introduces exactly a one-day (not zero-day) shift."""
    prices = make_price_panel(n_days=60, n_tickers=3)
    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    signal.iloc[10:, 0] = 1.0  # go long ticker 0 from day 10 onward

    engine = BacktestEngine(initial_capital=100000, cost_model=CostModel(),
                             position_sizing_cfg={"method": "equal_weight", "max_position_size": 1.0,
                                                   "max_portfolio_exposure": 1.0})
    result = engine.run(prices, signal)

    # Weight should NOT be active on day 10 itself (signal day), only from day 11
    assert result.weights.iloc[10][prices.columns[0]] == 0.0
    assert result.weights.iloc[11][prices.columns[0]] != 0.0


def test_volume_aware_engine_run_produces_higher_costs_for_thin_volume_stock():
    """Integration test: running the engine with volume_aware_slippage=True
    and a genuinely thin volume series should produce higher total costs
    than running the same signal/prices with volume-awareness disabled."""
    prices = make_price_panel(n_days=120, n_tickers=2)
    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    # Enter on day 30 (well past the rolling ADV window's min_periods) and
    # exit on day 60, so both trades occur once the ADV estimate is populated.
    signal.iloc[30:60, 0] = 1.0
    signal.iloc[30:60, 1] = -1.0

    # Very thin, constant volume -> a full-capital trade will breach participation limit
    volume = pd.DataFrame(500, index=prices.index, columns=prices.columns)

    sizing_cfg = {"method": "equal_weight", "max_position_size": 1.0, "max_portfolio_exposure": 1.0}

    flat_engine = BacktestEngine(1_000_000, CostModel(volume_aware_slippage=False), sizing_cfg)
    flat_result = flat_engine.run(prices, signal)

    va_engine = BacktestEngine(1_000_000, CostModel(volume_aware_slippage=True, max_participation_rate=0.05), sizing_cfg)
    va_result = va_engine.run(prices, signal, volume=volume)

    assert va_result.costs.sum() > flat_result.costs.sum()
