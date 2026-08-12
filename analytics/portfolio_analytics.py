"""
Portfolio Analytics — sector exposure, correlation, risk contribution,
concentration, and rolling beta for multi-stock portfolios.

*** LOOKAHEAD-SAFETY NOTE ***
Covariance/correlation/beta here are always computed on a TRAILING window
ending at (and including) the date in question — i.e. using only
information available up to that day, consistent with the rest of the
codebase. Weights passed in should already be the engine's execution
weights (already t+1-shifted), so no additional shift is applied here.
"""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- sector ---
def sector_exposure_timeseries(weights: pd.DataFrame, sector_map: Dict[str, str]) -> pd.DataFrame:
    """Sum signed portfolio weight by sector, for every date. Columns are
    sector names; an 'Unclassified' column appears if any held ticker has
    no sector mapping."""
    sectors = pd.Series({t: sector_map.get(t, "Unclassified") for t in weights.columns})
    exposure = weights.T.groupby(sectors).sum().T
    return exposure.reindex(sorted(exposure.columns), axis=1)


def sector_exposure_snapshot(weights_row: pd.Series, sector_map: Dict[str, str]) -> pd.DataFrame:
    """Sector breakdown for a single date (e.g. the last date in the
    backtest) — active positions only."""
    df = weights_row.to_frame("weight")
    df["sector"] = [sector_map.get(t, "Unclassified") for t in df.index]
    df = df[df["weight"] != 0]
    grouped: pd.DataFrame = df.groupby("sector")[["weight"]].sum().sort_values("weight", ascending=False)
    grouped["weight_pct"] = (grouped["weight"] * 100).round(2)
    return grouped


# ------------------------------------------------------------ correlation ---
def correlation_matrix(prices: pd.DataFrame, as_of=None, lookback: int = 252) -> pd.DataFrame:
    """Trailing correlation of daily returns over `lookback` sessions ending
    at `as_of` (defaults to the last available date). Uses only price
    history up to and including `as_of`."""
    returns = prices.pct_change()
    if as_of is not None:
        returns = returns.loc[:as_of]
    window = returns.tail(lookback).dropna(axis=1, how="all")
    return window.corr()


# ------------------------------------------------------- risk contribution ---
def risk_contribution_snapshot(weights_row: pd.Series, prices: pd.DataFrame,
                                as_of=None, lookback: int = 60) -> pd.DataFrame:
    """Decompose portfolio volatility into each held ticker's contribution,
    using a trailing covariance matrix. Standard Euler decomposition:
        portfolio_var  = w' Σ w
        marginal_i     = (Σ w)_i
        contribution_i = w_i * marginal_i          (sums to portfolio_var)
        pct_contrib_i  = contribution_i / portfolio_var
    """
    active = weights_row[weights_row != 0]
    if active.empty:
        return pd.DataFrame(columns=["weight", "pct_risk_contribution"])

    returns = prices[active.index].pct_change()
    if as_of is not None:
        returns = returns.loc[:as_of]
    cov = returns.tail(lookback).cov() * 252  # annualized covariance

    w = active.reindex(cov.columns).fillna(0.0).values
    port_var = w @ cov.values @ w
    if port_var <= 0 or np.isnan(port_var):
        return pd.DataFrame({"weight": active}).assign(pct_risk_contribution=np.nan)

    marginal = cov.values @ w
    contribution = w * marginal
    pct_contribution = contribution / port_var

    result = pd.DataFrame({
        "weight": active.reindex(cov.columns).fillna(0.0),
        "risk_contribution": contribution,
        "pct_risk_contribution": (pct_contribution * 100).round(2),
    }).sort_values("pct_risk_contribution", ascending=False)
    return result


def sector_risk_contribution(risk_contrib: pd.DataFrame, sector_map: Dict[str, str]) -> pd.DataFrame:
    df = risk_contrib.copy()
    df["sector"] = [sector_map.get(t, "Unclassified") for t in df.index]
    return df.groupby("sector")[["weight", "risk_contribution", "pct_risk_contribution"]].sum().sort_values(
        "pct_risk_contribution", ascending=False)


# ------------------------------------------------------------ concentration ---
def concentration_hhi(weights: pd.DataFrame) -> pd.Series:
    """Herfindahl-Hirschman Index of absolute weights per date — a single
    number summarizing concentration (1/N = fully diversified across N
    equal positions; 1.0 = fully concentrated in one name)."""
    abs_w = weights.abs()
    gross = abs_w.sum(axis=1).replace(0, np.nan)
    norm_w = abs_w.div(gross, axis=0)
    return (norm_w ** 2).sum(axis=1)


# --------------------------------------------------------------------- beta ---
def rolling_beta(strategy_returns: pd.Series, benchmark_returns: pd.Series, window: int = 126) -> pd.Series:
    """Trailing rolling beta of strategy returns vs benchmark returns."""
    aligned = pd.concat([strategy_returns.rename("s"), benchmark_returns.rename("b")], axis=1).dropna()
    cov = aligned["s"].rolling(window).cov(aligned["b"])
    var = aligned["b"].rolling(window).var()
    return (cov / var).rename("rolling_beta")
