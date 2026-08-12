"""
Data provider layer — pluggable interface over yfinance (free, ₹0 cost).

Design note: sequential per-ticker yf.download() calls trigger Yahoo Finance
rate limiting. We batch tickers per request with group_by="ticker" and use
exponential backoff retries, which is the pattern that reliably avoids
throttling in practice.

To add a paid/alternate provider later (e.g. NSE historical bhavcopy,
a broker API), implement a class with the same `fetch(tickers, start, end)`
signature and swap it in `loader.py`.
"""
from __future__ import annotations
import time
import logging
from typing import Dict, List
import pandas as pd

logger = logging.getLogger(__name__)


class YFinanceProvider:
    """Free data provider using Yahoo Finance via yfinance."""

    def __init__(self, batch_size: int = 15, max_retries: int = 4, base_backoff: float = 2.0):
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.base_backoff = base_backoff

    def fetch(self, tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV for all tickers. Returns {ticker: DataFrame}."""
        import yfinance as yf

        results: Dict[str, pd.DataFrame] = {}
        batches = [tickers[i:i + self.batch_size] for i in range(0, len(tickers), self.batch_size)]

        for batch_num, batch in enumerate(batches, start=1):
            logger.info(f"Fetching batch {batch_num}/{len(batches)}: {batch}")
            attempt = 0
            while attempt <= self.max_retries:
                try:
                    data = yf.download(
                        tickers=batch, start=start, end=end,
                        group_by="ticker", auto_adjust=False,
                        threads=True, progress=False,
                    )
                    break
                except Exception as e:
                    attempt += 1
                    wait = self.base_backoff * (2 ** (attempt - 1))
                    logger.warning(f"Batch {batch_num} attempt {attempt} failed ({e}); retrying in {wait:.1f}s")
                    time.sleep(wait)
            else:
                logger.error(f"Batch {batch_num} failed after {self.max_retries} retries; skipping {batch}")
                continue

            # Robustly detect column shape rather than assuming based on
            # batch length: newer yfinance versions return MultiIndex
            # columns (ticker, field) for a LIST input even when the list
            # has only one ticker — flat columns only happen when a single
            # ticker STRING (not a list) is passed. Trust data.columns, not len(batch).
            is_multiindex = isinstance(data.columns, pd.MultiIndex)

            if not is_multiindex:
                # Flat columns: entire `data` frame belongs to the one ticker.
                df = data.copy()
                if not df.empty and "Close" in df.columns and df["Close"].notna().any():
                    results[batch[0]] = df
                else:
                    logger.warning(f"No usable data returned for {batch[0]}")
            else:
                for ticker in batch:
                    try:
                        df = data[ticker].copy()
                        if not df.empty and "Close" in df.columns and df["Close"].notna().any():
                            results[ticker] = df
                        else:
                            logger.warning(f"No data returned for {ticker}")
                    except KeyError:
                        logger.warning(f"Could not extract data for {ticker}")

            time.sleep(1.0)  # be polite between batches

        return results

    def fetch_single(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        result = self.fetch([ticker], start, end)
        return result.get(ticker, pd.DataFrame())
