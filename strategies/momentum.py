"""
Cross-Sectional Momentum Strategy.

Ranks all tickers in the universe by trailing momentum on each rebalance
date and selects the top N. This is CROSS-SECTIONAL momentum (rank stocks
against each other), as distinct from TIME-SERIES momentum (each stock
against its own history — e.g. "is 12m return positive?"). This module
implements the cross-sectional variant per the project spec.

Momentum is measured as trailing total return over `lookback` sessions,
EXCLUDING the most recent `skip_recent_days` sessions — a standard academic
convention (Jegadeesh & Titman) that avoids the well-documented short-term
reversal effect contaminating the momentum signal.

Selection happens only on rebalance dates (config: backtest.rebalance_freq);
the resulting weight is held constant between rebalances. Selection uses
prices only up to and including the rebalance date itself.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    name = "momentum"

    def __init__(self, params: dict, rebalance_freq: str = "W"):
        super().__init__(params)
        self.rebalance_freq = rebalance_freq

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        lookback = int(p.get("lookback", 126))
        top_n = int(p.get("top_n", 10))
        skip = int(p.get("skip_recent_days", 5))

        # momentum score at date t = return from (t - lookback) to (t - skip)
        ref_price = prices.shift(skip)
        past_price = prices.shift(lookback + skip)
        momentum_score = (ref_price / past_price) - 1.0

        # Rebalance dates: last available trading date in each period
        rebalance_dates = prices.resample(self.rebalance_freq).apply(
            lambda x: x.index.max() if len(x) else None  # type: ignore[arg-type,return-value]
        )
        rebalance_dates = pd.Series(rebalance_dates.values.ravel()).dropna().unique()  # type: ignore[assignment]
        rebalance_dates = pd.DatetimeIndex(sorted(rebalance_dates))  # type: ignore[assignment]
        rebalance_dates = rebalance_dates.intersection(prices.index)  # type: ignore[operator]

        signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        for dt in rebalance_dates:
            row = momentum_score.loc[dt]
            row = row.dropna()
            if row.empty:
                continue
            top = row.sort_values(ascending=False).head(top_n).index
            signal.loc[dt, top] = 1.0

        # Hold selection between rebalances; only rebalance-date rows carry
        # signal, all other rows are forward-filled from the last rebalance.
        result = signal.copy()
        mask_non_rebal = ~result.index.isin(rebalance_dates)
        result.loc[mask_non_rebal] = np.nan
        result = result.ffill().fillna(0.0)
        return result
