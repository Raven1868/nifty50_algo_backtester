"""
Performance metrics. Risk-free rate assumption: 6.5% annualized (approx.
Indian 10Y G-Sec / T-Bill proxy as of early-2026) — override via function
argument if needed. This is a simplification (flat rate over the whole
period); document as such in any report generated.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252
DEFAULT_RISK_FREE_ANNUAL = 0.065


def cagr(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return np.nan
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    if years <= 0:
        return np.nan
    return (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1


def annualized_vol(daily_returns: pd.Series) -> float:
    return daily_returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(daily_returns: pd.Series, risk_free_annual: float = DEFAULT_RISK_FREE_ANNUAL) -> float:
    rf_daily = risk_free_annual / TRADING_DAYS
    excess = daily_returns - rf_daily
    std = excess.std()
    if np.isnan(std) or np.isclose(std, 0.0, atol=1e-12):
        return np.nan
    return (excess.mean() / std) * np.sqrt(TRADING_DAYS)


def sortino_ratio(daily_returns: pd.Series, risk_free_annual: float = DEFAULT_RISK_FREE_ANNUAL) -> float:
    rf_daily = risk_free_annual / TRADING_DAYS
    excess = daily_returns - rf_daily
    downside = excess[excess < 0]
    downside_dev = downside.std()
    if np.isnan(downside_dev) or np.isclose(downside_dev, 0.0, atol=1e-12):
        return np.nan
    return (excess.mean() / downside_dev) * np.sqrt(TRADING_DAYS)


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return drawdown.min()


def calmar_ratio(equity_curve: pd.Series) -> float:
    mdd = max_drawdown(equity_curve)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return cagr(equity_curve) / abs(mdd)


def value_at_risk(daily_returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR (daily), e.g. 95% -> loss not expected to be exceeded
    on 95% of days."""
    return daily_returns.quantile(1 - confidence)


def full_summary(equity_curve: pd.Series, daily_returns: pd.Series,
                  initial_capital: float, risk_free_annual: float = DEFAULT_RISK_FREE_ANNUAL) -> dict:
    return {
        "Initial Capital (INR)": round(initial_capital, 2),
        "Final Capital (INR)": round(equity_curve.iloc[-1], 2),
        "Total Return (%)": round((equity_curve.iloc[-1] / initial_capital - 1) * 100, 2),
        "CAGR (%)": round(cagr(equity_curve) * 100, 2),
        "Annualized Volatility (%)": round(annualized_vol(daily_returns) * 100, 2),
        "Sharpe Ratio": round(sharpe_ratio(daily_returns, risk_free_annual), 3),
        "Sortino Ratio": round(sortino_ratio(daily_returns, risk_free_annual), 3),
        "Max Drawdown (%)": round(max_drawdown(equity_curve) * 100, 2),
        "Calmar Ratio": round(calmar_ratio(equity_curve), 3),
        "Daily VaR 95% (%)": round(value_at_risk(daily_returns, 0.95) * 100, 3),
        "Risk-Free Rate Assumption (%)": round(risk_free_annual * 100, 2),
    }
