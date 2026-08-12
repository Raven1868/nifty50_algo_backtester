import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from analytics.performance import cagr, max_drawdown, sharpe_ratio, annualized_vol
from backtesting.costs import CostModel


def test_cagr_known_case():
    dates = pd.bdate_range("2020-01-01", periods=253)  # ~1 year
    equity = pd.Series(np.linspace(100000, 110000, 253), index=dates)
    result = cagr(equity)
    assert 0.09 < result < 0.11  # ~10% growth over ~1 year


def test_max_drawdown_known_case():
    equity = pd.Series([100, 110, 90, 95, 120], index=pd.bdate_range("2023-01-02", periods=5))
    dd = max_drawdown(equity)
    assert abs(dd - (90 / 110 - 1)) < 1e-9


def test_sharpe_zero_vol_returns_nan():
    returns = pd.Series([0.0] * 30)
    assert np.isnan(sharpe_ratio(returns))


def test_annualized_vol_positive():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.01, 252))
    assert annualized_vol(returns) > 0


def test_cost_model_fraction_positive_and_reasonable():
    cm = CostModel()
    frac = cm.cost_fraction()
    assert 0 < frac < 0.01  # sanity: well under 1% one-way for these bps-scale defaults


def test_cost_model_apply_scales_with_notional():
    cm = CostModel()
    assert cm.apply(200000) == 2 * cm.apply(100000)


def test_volume_aware_slippage_disabled_matches_flat_model():
    cm = CostModel(volume_aware_slippage=False)
    assert cm.apply_volume_aware(50000, avg_dollar_volume=1000000) == cm.apply(50000)


def test_volume_aware_slippage_below_participation_threshold_matches_flat():
    cm = CostModel(volume_aware_slippage=True, max_participation_rate=0.10)
    # trade is only 1% of avg daily volume -> well under 10% threshold -> flat cost
    cost = cm.apply_volume_aware(traded_notional=10000, avg_dollar_volume=1000000)
    assert abs(cost - cm.apply(10000)) < 1e-9


def test_volume_aware_slippage_scales_up_above_threshold():
    cm = CostModel(volume_aware_slippage=True, max_participation_rate=0.10, slippage_bps=5.0)
    small_trade_cost = cm.apply_volume_aware(traded_notional=50000, avg_dollar_volume=1000000)   # 5% participation
    large_trade_cost = cm.apply_volume_aware(traded_notional=500000, avg_dollar_volume=1000000)  # 50% participation
    # large trade's cost-per-rupee-traded must be higher due to slippage scaling
    assert (large_trade_cost / 500000) > (small_trade_cost / 50000)


def test_volume_aware_slippage_handles_zero_or_none_adv_gracefully():
    cm = CostModel(volume_aware_slippage=True)
    assert cm.apply_volume_aware(10000, avg_dollar_volume=0) == cm.apply(10000)
    assert cm.apply_volume_aware(10000, avg_dollar_volume=None) == cm.apply(10000)
