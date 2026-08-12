"""
Data cleaning layer. Every cleaning decision is logged — nothing is silently
modified. Handles: missing values, duplicate dates, timezone normalization,
abnormal single-day price spikes (likely bad ticks / unadjusted split
artifacts), and standardizes columns.
"""
from __future__ import annotations
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def clean_ohlcv(df: pd.DataFrame, ticker: str, max_daily_move: float = 0.35) -> pd.DataFrame:
    """Clean a single ticker's OHLCV frame.

    Args:
        df: raw OHLCV frame with a DatetimeIndex.
        ticker: for logging context.
        max_daily_move: |return| above this is flagged as a likely bad tick
            (not silently dropped — logged, and left in place unless it is a
            zero/negative price which IS invalid and is dropped).
    """
    if df.empty:
        return df

    df = df.copy()

    # Defensive safety net: if a MultiIndex-column frame slips through from
    # the provider (e.g. a future yfinance change), flatten it here rather
    # than silently producing a broken "Close" column downstream.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
        logger.warning(f"[{ticker}] received MultiIndex columns from provider; flattened to {list(df.columns)}")

    # Normalize index: drop timezone, sort, drop exact-duplicate dates
    if df.index.tz is not None:  # type: ignore[attr-defined]
        df.index = df.index.tz_localize(None)  # type: ignore[attr-defined]
    df = df.sort_index()
    n_dupes = df.index.duplicated().sum()
    if n_dupes:
        logger.info(f"[{ticker}] dropping {n_dupes} duplicate-date rows (keeping last)")
        df = df[~df.index.duplicated(keep="last")]

    # Ensure required columns exist
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        logger.warning(f"[{ticker}] missing columns {missing_cols}; filling with NaN")
        for c in missing_cols:
            df[c] = np.nan

    # Invalid prices (<=0) are true data errors -> drop those rows
    invalid_price_mask = (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
    n_invalid = invalid_price_mask.sum()
    if n_invalid:
        logger.info(f"[{ticker}] dropping {n_invalid} rows with non-positive OHLC values")
        df = df[~invalid_price_mask]

    # Forward-fill isolated missing values (max 2 consecutive sessions) — do
    # NOT fill long gaps, that would fabricate trading history.
    na_before = df["Close"].isna().sum()
    df[REQUIRED_COLS] = df[REQUIRED_COLS].ffill(limit=2)
    na_after = df["Close"].isna().sum()
    if na_before != na_after:
        logger.info(f"[{ticker}] forward-filled {na_before - na_after} short NaN gaps (limit=2 sessions)")

    # Drop rows that are still NaN in Close (real gaps — likely delisting or
    # provider outage). Left out rather than filled to avoid inventing data.
    remaining_na = df["Close"].isna().sum()
    if remaining_na:
        logger.info(f"[{ticker}] dropping {remaining_na} rows still NaN after limited ffill")
        df = df.dropna(subset=["Close"])

    # Flag (log only) abnormal single-day moves — likely unadjusted corporate
    # actions or bad ticks. We do not auto-correct these without a corporate
    # actions feed; flagging keeps the researcher aware.
    if len(df) > 1:
        daily_ret = df["Close"].pct_change()
        abnormal = daily_ret.abs() > max_daily_move
        n_abnormal = abnormal.sum()
        if n_abnormal:
            dates = df.index[abnormal].strftime("%Y-%m-%d").tolist()  # type: ignore[attr-defined]
            logger.warning(
                f"[{ticker}] {n_abnormal} sessions with |return| > {max_daily_move:.0%} "
                f"(possible unadjusted split/bad tick): {dates[:5]}{'...' if n_abnormal > 5 else ''}"
            )

    return df
