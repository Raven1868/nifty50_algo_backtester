"""
Data loader — orchestrates provider + cache + cleaner, and produces the
normalized wide-format panels (one DataFrame per field, columns=tickers)
that the rest of the system consumes.
"""
from __future__ import annotations
import logging
import os
from typing import Dict, List, Tuple
import pandas as pd

from data.provider import YFinanceProvider
from data.cleaner import clean_ohlcv

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, cache_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.cache_dir = cache_dir
        self.processed_dir = processed_dir
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        self.provider = YFinanceProvider()

    def _cache_path(self, ticker: str) -> str:
        return os.path.join(self.cache_dir, f"{ticker.replace('.', '_')}.parquet")

    def load(self, tickers: List[str], start: str, end: str,
              use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """Fetch + clean per-ticker OHLCV, using local parquet cache (₹0 cost,
        avoids re-hitting Yahoo Finance on repeated runs)."""
        cleaned: Dict[str, pd.DataFrame] = {}
        to_fetch = []

        for t in tickers:
            path = self._cache_path(t)
            if use_cache and os.path.exists(path):
                df = pd.read_parquet(path)
                df.index = pd.to_datetime(df.index)
                # Refetch if cache doesn't cover the requested range
                if df.index.min() <= pd.Timestamp(start) and df.index.max() >= pd.Timestamp(end) - pd.Timedelta(days=5):
                    cleaned[t] = df
                    continue
            to_fetch.append(t)

        if to_fetch:
            raw = self.provider.fetch(to_fetch, start, end)
            for t, df in raw.items():
                df = clean_ohlcv(df, t)
                if not df.empty:
                    df.to_parquet(self._cache_path(t))
                    cleaned[t] = df

        missing = set(tickers) - set(cleaned.keys())
        if missing:
            logger.warning(f"No usable data for: {sorted(missing)}")

        return cleaned

    @staticmethod
    def to_panel(data: Dict[str, pd.DataFrame], field: str = "Adj Close",
                 fallback_field: "str | None" = None) -> pd.DataFrame:
        """Convert {ticker: OHLCV df} into a single wide DataFrame for one field,
        aligned on the union of trading dates (NaN where a ticker has no data
        for that date — engine handles this explicitly, never silently).

        Args:
            field: primary field to extract (e.g. "Adj Close").
            fallback_field: if a ticker's `field` column exists but is
                ENTIRELY NaN for that ticker, fall back to this field
                instead (e.g. "Close"). This matters for index tickers like
                "^NSEI" — Yahoo Finance frequently returns an all-NaN
                "Adj Close" for indices (no dividend adjustments apply to
                an index the way they do to a stock), which would otherwise
                silently produce an invisible/empty benchmark series.
        """
        series = {}
        for t, df in data.items():
            if field in df.columns and df[field].notna().any():
                series[t] = df[field]
            elif fallback_field and fallback_field in df.columns and df[fallback_field].notna().any():
                logger.info(f"[{t}] '{field}' was entirely NaN — using '{fallback_field}' instead")
                series[t] = df[fallback_field]
            elif field in df.columns:
                series[t] = df[field]  # keep it (all-NaN) so the gap is visible/debuggable, not silently dropped
        panel = pd.DataFrame(series).sort_index()
        return panel
