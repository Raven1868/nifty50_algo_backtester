"""
Parameter Sensitivity Analysis.

Runs a strategy across a 2D grid of two parameters (e.g. MA short/long
windows) and reports CAGR / Sharpe / Max Drawdown at every combination.
The goal is NOT to find the single best combination (that's optimizer.py's
job, and doing it here would just be optimization by another name) — it's
to see whether performance is stable across a broad PLATEAU of nearby
parameter values, or whether it collapses sharply away from one isolated
peak. A plateau suggests the strategy's edge is real and robust; an
isolated spike surrounded by mediocre-to-poor results is a classic
overfitting red flag.
"""
from __future__ import annotations
from typing import List, Type
import numpy as np
import pandas as pd

from backtesting.engine import BacktestEngine
from backtesting.costs import CostModel
from analytics import performance as perf


def parameter_sensitivity_2d(
    strategy_cls: Type, base_params: dict, param1_name: str, param1_values: List,
    param2_name: str, param2_values: List, prices: pd.DataFrame, initial_capital: float,
    cost_model: CostModel, position_sizing_cfg: dict, rebalance_freq: str = "W",
) -> dict:
    """Returns {'cagr': DataFrame, 'sharpe': DataFrame, 'max_drawdown': DataFrame}
    each indexed by param1_values (rows) x param2_values (columns)."""
    cagr_matrix = pd.DataFrame(index=param1_values, columns=param2_values, dtype=float)
    sharpe_matrix = pd.DataFrame(index=param1_values, columns=param2_values, dtype=float)
    mdd_matrix = pd.DataFrame(index=param1_values, columns=param2_values, dtype=float)

    for v1 in param1_values:
        for v2 in param2_values:
            params = {**base_params, param1_name: v1, param2_name: v2}
            try:
                if strategy_cls.__name__ == "MomentumStrategy":
                    strategy = strategy_cls(params, rebalance_freq=rebalance_freq)
                else:
                    strategy = strategy_cls(params)
                signal = strategy.generate_signals(prices)
                engine = BacktestEngine(initial_capital, cost_model, position_sizing_cfg)
                result = engine.run(prices, signal)
                cagr_matrix.loc[v1, v2] = perf.cagr(result.equity_curve)
                sharpe_matrix.loc[v1, v2] = perf.sharpe_ratio(result.daily_net_return)
                mdd_matrix.loc[v1, v2] = perf.max_drawdown(result.equity_curve)
            except Exception:
                pass  # invalid combo (e.g. short_window >= long_window) -> leave NaN

    cagr_matrix.index.name = param1_name
    cagr_matrix.columns.name = param2_name
    sharpe_matrix.index.name = param1_name
    sharpe_matrix.columns.name = param2_name
    mdd_matrix.index.name = param1_name
    mdd_matrix.columns.name = param2_name

    return {"cagr": cagr_matrix, "sharpe": sharpe_matrix, "max_drawdown": mdd_matrix}


def plateau_diagnostic(sharpe_matrix: pd.DataFrame) -> str:
    """Cheap heuristic verdict on plateau vs isolated-optimum, for the report."""
    vals = sharpe_matrix.values.astype(float)
    valid = vals[~np.isnan(vals)]
    if len(valid) < 4:
        return "Insufficient valid combinations to assess robustness."
    best = np.nanmax(vals)
    median = np.nanmedian(valid)
    spread = np.nanstd(valid)
    if median <= 0:
        return "Median Sharpe across the grid is non-positive — no robust edge detected in this parameter region."
    ratio = (best - median) / (abs(median) + 1e-9)
    if ratio > 2.0:
        return (f"Best Sharpe ({best:.2f}) is far above the grid median ({median:.2f}) — "
                "looks like an ISOLATED OPTIMUM, not a robust plateau. Treat with caution.")
    return (f"Best Sharpe ({best:.2f}) is reasonably close to the grid median ({median:.2f}) — "
            "performance appears relatively stable across nearby parameters (a plateau).")
