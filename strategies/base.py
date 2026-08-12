"""
Base strategy interface. Every strategy consumes a price panel (dates x
tickers) and returns a SIGNAL panel of the same shape with values in
[-1, +1] representing desired target weight direction/strength on the day
the signal is computed (SIGNAL DATE).

*** LOOKAHEAD-BIAS CONTRACT ***
Strategies must only use data available up to and including the row's own
date (rolling windows, .shift() already applied internally). Strategies must
NOT shift signals themselves — the backtesting engine is solely responsible
for the signal->execution shift (t -> t+1), so this logic lives in exactly
one place and can be unit-tested once (see tests/test_lookahead.py).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    name: str = "base"

    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            prices: wide DataFrame, index=dates, columns=tickers, values=
                adjusted close prices. Must be sorted ascending by date.
        Returns:
            DataFrame same shape/index/columns as `prices`, values in
            [-1, 1] (0 = flat). This is the SIGNAL as of that date — the
            engine executes it at t+1.
        """
        raise NotImplementedError
