"""
Market Regime Analysis.

Classifies each trading day into a regime using ONLY backward-looking
information (trailing trend + trailing volatility of the benchmark), then
evaluates strategy performance conditioned on the regime active that day.

Trend regime: benchmark close vs its trailing 200-session SMA
  - "bull"     : close > SMA200
  - "bear"     : close < SMA200

Volatility regime: trailing 21-session realized vol vs its own trailing
252-session median (an adaptive, non-lookahead threshold)
  - "high_vol" : above trailing median
  - "low_vol"  : at or below trailing median

Regimes are assigned using data available up to and including that day, so
labeling a day's regime is itself lookahead-safe. Strategy returns used
here should already be the engine's (already-shifted) daily_net_return
series, so no additional shift is applied in this module.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def classify_regimes(benchmark_prices: pd.Series, trend_window: int = 200,
                      vol_window: int = 21, vol_median_window: int = 252) -> pd.DataFrame:
    sma = benchmark_prices.rolling(trend_window, min_periods=trend_window).mean()
    trend = np.where(benchmark_prices > sma, "bull", "bear")

    daily_ret = benchmark_prices.pct_change()
    realized_vol = daily_ret.rolling(vol_window, min_periods=vol_window).std() * np.sqrt(252)
    trailing_median_vol = realized_vol.rolling(vol_median_window, min_periods=vol_window).median()
    vol_regime = np.where(realized_vol > trailing_median_vol, "high_vol", "low_vol")

    regimes = pd.DataFrame({"trend": trend, "volatility": vol_regime}, index=benchmark_prices.index)
    regimes.loc[sma.isna(), "trend"] = "unclassified"
    regimes.loc[trailing_median_vol.isna(), "volatility"] = "unclassified"
    return regimes


def performance_by_regime(strategy_daily_returns: pd.Series, regimes: pd.DataFrame) -> pd.DataFrame:
    """Break down strategy return statistics by trend regime, vol regime,
    and the combination of both."""
    from analytics.performance import sharpe_ratio, annualized_vol

    df = pd.concat([strategy_daily_returns.rename("ret"), regimes], axis=1).dropna(subset=["ret"])
    rows = []

    for label, group in df.groupby("trend"):
        rows.append(_regime_row(f"trend={label}", group["ret"]))
    for label, group in df.groupby("volatility"):
        rows.append(_regime_row(f"volatility={label}", group["ret"]))
    for (t, v), group in df.groupby(["trend", "volatility"]):
        rows.append(_regime_row(f"trend={t} & volatility={v}", group["ret"]))

    return pd.DataFrame(rows)


def _regime_row(label: str, returns: pd.Series) -> dict:
    from analytics.performance import sharpe_ratio, annualized_vol
    n = len(returns)
    return {
        "regime": label,
        "n_days": n,
        "mean_daily_return_pct": round(returns.mean() * 100, 4) if n else np.nan,
        "annualized_vol_pct": round(annualized_vol(returns) * 100, 2) if n > 5 else np.nan,
        "sharpe": round(sharpe_ratio(returns), 3) if n > 5 else np.nan,
        "pct_positive_days": round((returns > 0).mean() * 100, 1) if n else np.nan,
    }
