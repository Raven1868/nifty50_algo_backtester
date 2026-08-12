"""
Mean-Reversion Strategy.

Methodology:
  - rolling mean & std of price over `lookback` sessions
  - z = (price - rolling_mean) / rolling_std
  - go long when z <= entry_z (price sufficiently below its mean)
  - exit (go flat) when z >= exit_z (price has reverted toward the mean)
  - optional short leg (symmetric) when allow_short=True
  - stop-loss and max-holding-period are enforced by the RISK layer, not here
    (strategy vs risk separation per project spec) — this module only emits
    the raw target signal.

All rolling stats use only data up to and including the current row, so no
lookahead is introduced here; the engine still shifts by one day before
execution.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    name = "mean_reversion"

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        lookback = int(p.get("lookback", 20))
        entry_z = float(p.get("entry_z", -1.5))
        exit_z = float(p.get("exit_z", -0.3))
        allow_short = bool(p.get("allow_short", False))

        roll_mean = prices.rolling(lookback, min_periods=lookback).mean()
        roll_std = prices.rolling(lookback, min_periods=lookback).std()
        z = (prices - roll_mean) / roll_std.replace(0, np.nan)

        signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Stateful long/flat (and optionally short) logic per column,
        # vectorized state machine via forward-fill of entry/exit events.
        for col in prices.columns:
            zc = z[col]
            state = pd.Series(np.nan, index=zc.index)
            state[zc <= entry_z] = 1.0
            state[zc >= exit_z] = 0.0
            if allow_short:
                state[zc >= -entry_z] = -1.0
                state[zc <= -exit_z] = 0.0
            state = state.ffill().fillna(0.0)
            signal[col] = state

        return signal.fillna(0.0)
