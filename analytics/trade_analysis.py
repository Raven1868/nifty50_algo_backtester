"""Trade-level analytics: win rate, profit factor, avg win/loss, etc."""
from __future__ import annotations
import numpy as np
import pandas as pd


def trade_statistics(trade_log: pd.DataFrame) -> dict:
    if trade_log.empty:
        return {"Number of Trades": 0}

    wins = trade_log[trade_log["gross_return_pct"] > 0]
    losses = trade_log[trade_log["gross_return_pct"] <= 0]
    win_rate = len(wins) / len(trade_log) if len(trade_log) else np.nan

    gross_profit = wins["gross_return_pct"].sum()
    gross_loss = abs(losses["gross_return_pct"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

    return {
        "Number of Trades": len(trade_log),
        "Win Rate (%)": round(win_rate * 100, 2),
        "Average Win (%)": round(wins["gross_return_pct"].mean(), 3) if not wins.empty else np.nan,
        "Average Loss (%)": round(losses["gross_return_pct"].mean(), 3) if not losses.empty else np.nan,
        "Profit Factor": round(profit_factor, 3) if pd.notna(profit_factor) else np.nan,
        "Best Trade (%)": round(trade_log["gross_return_pct"].max(), 3),
        "Worst Trade (%)": round(trade_log["gross_return_pct"].min(), 3),
        "Average Holding Period (days)": round(trade_log["holding_days"].mean(), 1),
    }
