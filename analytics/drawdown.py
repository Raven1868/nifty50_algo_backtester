"""Drawdown analytics: series, duration, recovery period, average drawdown."""
from __future__ import annotations
import pandas as pd
import numpy as np


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    running_max = equity_curve.cummax()
    return equity_curve / running_max - 1


def drawdown_periods(equity_curve: pd.Series) -> pd.DataFrame:
    """Identify each distinct drawdown episode: start (peak), trough, end
    (recovery to new high), depth, and duration/recovery in calendar days."""
    dd = drawdown_series(equity_curve)
    in_dd = dd < 0
    episodes: list[dict] = []
    start: "pd.Timestamp | None" = None
    peak_val = None
    for i, (date, is_dd) in enumerate(in_dd.items()):
        date = pd.Timestamp(date)  # type: ignore[arg-type]
        if is_dd and start is None:
            start = date
            peak_val = equity_curve.loc[:date].iloc[:-1].max() if i > 0 else equity_curve.iloc[0]
        elif not is_dd and start is not None:
            segment = dd.loc[start:date]
            trough_date = pd.Timestamp(segment.idxmin())
            episodes.append({
                "peak_date": start,
                "trough_date": trough_date,
                "recovery_date": date,
                "depth_pct": round(segment.min() * 100, 2),
                "duration_days": (trough_date - start).days,
                "recovery_days": (date - trough_date).days,
            })
            start = None

    if start is not None:  # ongoing drawdown at end of series
        segment = dd.loc[start:]
        trough_date = pd.Timestamp(segment.idxmin())
        episodes.append({
            "peak_date": start,
            "trough_date": trough_date,
            "recovery_date": None,
            "depth_pct": round(segment.min() * 100, 2),
            "duration_days": (trough_date - start).days,
            "recovery_days": None,
        })

    return pd.DataFrame(episodes)


def average_drawdown(equity_curve: pd.Series) -> float:
    dd = drawdown_series(equity_curve)
    negative_dd = dd[dd < 0]
    return negative_dd.mean() if not negative_dd.empty else 0.0
