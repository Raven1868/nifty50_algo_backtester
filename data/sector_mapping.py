"""
Sector mapping — free, static reference data.

*** LIMITATION DISCLOSURE ***
There is no free, reliably-licensed API for NSE sectoral classification
that updates automatically. SECTOR_MAP below is a manually-curated,
point-in-time snapshot (broad GICS-like buckets, not the exact NSE
sectoral-index taxonomy) for the NIFTY50_CURRENT list in
`data/universe.py`. It will drift out of date as:
  - company business mix changes (e.g. conglomerates get reclassified)
  - NIFTY50 constituents change (new entrants won't be in this map)

Effect: sector exposure / concentration figures are directional, not
audit-grade. Verify against nseindia.com's official sectoral indices
before using for anything beyond a portfolio-demo analysis.

To replace with a proper source later: any CSV with columns
(ticker, sector) can be loaded and passed to `get_sector_map(path=...)`
without touching the analytics code that consumes this mapping.
"""
from __future__ import annotations
from typing import Dict, Optional
import logging
import pandas as pd

logger = logging.getLogger(__name__)

SECTOR_MAP: Dict[str, str] = {
    "ADANIENT.NS": "Diversified/Conglomerate", "ADANIPORTS.NS": "Infrastructure",
    "APOLLOHOSP.NS": "Healthcare", "ASIANPAINT.NS": "Consumer Discretionary",
    "AXISBANK.NS": "Financials", "BAJAJ-AUTO.NS": "Automobile",
    "BAJFINANCE.NS": "Financials", "BAJAJFINSV.NS": "Financials",
    "BEL.NS": "Industrials/Defence", "BHARTIARTL.NS": "Telecom",
    "CIPLA.NS": "Healthcare", "COALINDIA.NS": "Energy/Mining",
    "DRREDDY.NS": "Healthcare", "EICHERMOT.NS": "Automobile",
    "GRASIM.NS": "Materials", "HCLTECH.NS": "Information Technology",
    "HDFCBANK.NS": "Financials", "HDFCLIFE.NS": "Financials",
    "HEROMOTOCO.NS": "Automobile", "HINDALCO.NS": "Materials",
    "HINDUNILVR.NS": "Consumer Staples", "ICICIBANK.NS": "Financials",
    "INDUSINDBK.NS": "Financials", "INFY.NS": "Information Technology",
    "ITC.NS": "Consumer Staples", "JSWSTEEL.NS": "Materials",
    "KOTAKBANK.NS": "Financials", "LT.NS": "Industrials",
    "M&M.NS": "Automobile", "MARUTI.NS": "Automobile",
    "NESTLEIND.NS": "Consumer Staples", "NTPC.NS": "Utilities",
    "ONGC.NS": "Energy/Mining", "POWERGRID.NS": "Utilities",
    "RELIANCE.NS": "Energy/Conglomerate", "SBILIFE.NS": "Financials",
    "SBIN.NS": "Financials", "SHRIRAMFIN.NS": "Financials",
    "SUNPHARMA.NS": "Healthcare", "TATACONSUM.NS": "Consumer Staples",
    "TATAMOTORS.NS": "Automobile", "TATASTEEL.NS": "Materials",
    "TCS.NS": "Information Technology", "TECHM.NS": "Information Technology",
    "TITAN.NS": "Consumer Discretionary", "TRENT.NS": "Consumer Discretionary",
    "ULTRACEMCO.NS": "Materials", "UPL.NS": "Materials",
    "WIPRO.NS": "Information Technology", "LTIM.NS": "Information Technology",
}


def get_sector_map(path: Optional[str] = None) -> Dict[str, str]:
    """Return {ticker: sector}. Pass `path` to a CSV with columns
    (ticker, sector) to override the built-in static map — this is the
    drop-in point for a better/paid data source later."""
    if path:
        df = pd.read_csv(path)
        return dict(zip(df["ticker"], df["sector"]))
    return dict(SECTOR_MAP)


def map_tickers_to_sectors(tickers, sector_map: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    sector_map = sector_map or get_sector_map()
    result = {}
    unmapped = []
    for t in tickers:
        if t in sector_map:
            result[t] = sector_map[t]
        else:
            result[t] = "Unclassified"
            unmapped.append(t)
    if unmapped:
        logger.warning(f"No sector mapping for {len(unmapped)} ticker(s): {unmapped} — labeled 'Unclassified'")
    return result
