import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from strategies.moving_average import MovingAverageStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy


def _panel():
    dates = pd.bdate_range("2023-01-02", periods=250)
    rng = np.random.default_rng(1)
    data = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, size=(250, 4)), axis=0)
    return pd.DataFrame(data, index=dates, columns=["A.NS", "B.NS", "C.NS", "D.NS"])


def test_ma_crossover_signal_shape_and_values():
    prices = _panel()
    sig = MovingAverageStrategy({"short_window": 5, "long_window": 20}).generate_signals(prices)
    assert sig.shape == prices.shape
    assert set(np.unique(sig.values)).issubset({0.0, 1.0})


def test_ma_crossover_warmup_is_flat():
    prices = _panel()
    sig = MovingAverageStrategy({"short_window": 5, "long_window": 20}).generate_signals(prices)
    assert (sig.iloc[:19] == 0.0).all().all()  # long_window=20 -> warm-up for first 19 rows


def test_mean_reversion_output_bounds():
    prices = _panel()
    sig = MeanReversionStrategy({"lookback": 20, "entry_z": -1.5, "exit_z": -0.3, "allow_short": True}).generate_signals(prices)
    assert sig.values.min() >= -1.0
    assert sig.values.max() <= 1.0


def test_momentum_selects_at_most_top_n():
    prices = _panel()
    sig = MomentumStrategy({"lookback": 60, "top_n": 2, "skip_recent_days": 5}, rebalance_freq="W").generate_signals(prices)
    active_per_day = (sig != 0).sum(axis=1)
    assert active_per_day.max() <= 2
