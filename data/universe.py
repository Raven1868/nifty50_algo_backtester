"""
Universe module — defines the tradable stock universe.

*** SURVIVORSHIP BIAS DISCLOSURE ***
The NIFTY50_CURRENT list below reflects TODAY's index constituents (as of the
last time this file was updated). It is NOT a historically accurate,
point-in-time constituent list. Companies that were removed from the index
(e.g. due to bankruptcy, mergers, or under-performance) are absent, and
companies added later appear as if they were always members.

Effect: backtests run on this list will have a mild positive survivorship
bias — you are only testing on stocks that were "good enough" to still be
in the index today.

Fix (not implemented here, free data does not provide this reliably):
supply a historical constituent-change CSV (date, ticker, action) and have
`get_point_in_time_universe(date)` filter accordingly. The interface below
is written so this can be dropped in later without touching other modules.
"""
from __future__ import annotations
from datetime import date
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# NSE tickers use the ".NS" suffix for yfinance. List reflects NIFTY 50
# constituents as commonly published; verify against nseindia.com before
# using for anything beyond a portfolio-demo backtest.
NIFTY50_CURRENT: List[str] = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS",
    "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS",
    "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS",
    "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS", "ITC.NS",
    "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS", "MARUTI.NS",
    "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS",
    "SBILIFE.NS", "SBIN.NS", "SHRIRAMFIN.NS", "SUNPHARMA.NS", "TATACONSUM.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS",
    "TRENT.NS", "ULTRACEMCO.NS", "UPL.NS", "WIPRO.NS", "LTIM.NS",
]


def get_universe(mode: str = "nifty50", custom_tickers: Optional[List[str]] = None) -> List[str]:
    """Return the list of tickers to backtest on.

    Args:
        mode: "nifty50" for the current-constituent list, or "custom".
        custom_tickers: required when mode == "custom".
    """
    if mode == "nifty50":
        logger.warning(
            "Using CURRENT NIFTY50 constituents for the full backtest period. "
            "This introduces survivorship bias — see module docstring."
        )
        return list(NIFTY50_CURRENT)
    if mode == "custom":
        if not custom_tickers:
            raise ValueError("custom_tickers must be provided when mode='custom'")
        return list(custom_tickers)
    raise ValueError(f"Unknown universe mode: {mode}")


def get_point_in_time_universe(as_of: date) -> List[str]:
    """Placeholder for point-in-time (survivorship-bias-free) universe lookup.

    Not implemented: free data sources do not reliably provide historical
    NIFTY50 constituent-change history. Raises to make the limitation explicit
    rather than silently returning a biased list under a misleading name.
    """
    raise NotImplementedError(
        "Point-in-time universe requires a historical constituent-change file "
        "(date, ticker, action). Supply one and implement filtering here."
    )
