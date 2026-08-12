"""
Moving Average Crossover Strategy.

Long when Short MA > Long MA, flat otherwise (long-only by default, matches
project spec's stated entry/exit rule). Supports SMA or EMA.
"""
from __future__ import annotations
import pandas as pd
from strategies.base import BaseStrategy


class MovingAverageStrategy(BaseStrategy):
    name = "moving_average"

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        short_w = int(p.get("short_window", 20))
        long_w = int(p.get("long_window", 100))
        ma_type = p.get("ma_type", "SMA").upper()

        if ma_type == "EMA":
            short_ma = prices.ewm(span=short_w, min_periods=short_w).mean()
            long_ma = prices.ewm(span=long_w, min_periods=long_w).mean()
        else:
            short_ma = prices.rolling(short_w, min_periods=short_w).mean()
            long_ma = prices.rolling(long_w, min_periods=long_w).mean()

        signal = (short_ma > long_ma).astype(float)
        # NaN during warm-up (insufficient history) -> flat, not long
        warmup_mask = short_ma.isna() | long_ma.isna()
        signal[warmup_mask] = 0.0
        return signal
