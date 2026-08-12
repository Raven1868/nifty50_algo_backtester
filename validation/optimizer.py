"""
Parameter Optimization — controlled grid search.

Guardrails against overfitting (per project spec §17):
  - warns if the parameter grid is large relative to the sample (too many
    combinations tested on too little data invites curve-fitting)
  - warns if the training window is short (<252 sessions, ~1yr)
  - warns when the best-vs-median performance gap is extreme (isolated
    optimum rather than a robust plateau — see sensitivity.py for the
    fuller picture)
  - never optimizes on the same data used for out-of-sample reporting —
    callers (e.g. walk_forward.py) are responsible for only ever calling
    this on a TRAIN slice
"""
from __future__ import annotations
import itertools
import logging
from typing import Callable, Dict, List, Type
import numpy as np
import pandas as pd

from backtesting.engine import BacktestEngine
from backtesting.costs import CostModel
from analytics import performance as perf

logger = logging.getLogger(__name__)

OBJECTIVES: Dict[str, Callable[[pd.Series, pd.Series], float]] = {
    "sharpe": lambda eq, ret: perf.sharpe_ratio(ret),
    "sortino": lambda eq, ret: perf.sortino_ratio(ret),
    "cagr": lambda eq, ret: perf.cagr(eq),
    "calmar": lambda eq, ret: perf.calmar_ratio(eq),
}

MAX_SAFE_COMBINATIONS = 200  # heuristic threshold for the overfitting warning
MIN_SAFE_TRAIN_SESSIONS = 252


def _param_grid_combinations(param_grid: Dict[str, List]) -> List[Dict]:
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    return [dict(zip(keys, c)) for c in combos]


def optimize_strategy(strategy_cls: Type, base_params: dict, param_grid: Dict[str, List],
                       prices: pd.DataFrame, initial_capital: float, cost_model: CostModel,
                       position_sizing_cfg: dict, objective: str = "sharpe",
                       rebalance_freq: str = "W") -> pd.DataFrame:
    """Run a grid search over `param_grid`, holding `base_params` fixed for
    keys not being swept. Returns a results DataFrame sorted by objective
    (best first), one row per parameter combination."""
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective '{objective}'. Choose from {list(OBJECTIVES)}")

    combos = _param_grid_combinations(param_grid)
    n_sessions = len(prices)

    if len(combos) > MAX_SAFE_COMBINATIONS:
        logger.warning(
            f"Testing {len(combos)} parameter combinations — this risks overfitting, "
            f"especially with only {n_sessions} training sessions. Consider narrowing the grid."
        )
    if n_sessions < MIN_SAFE_TRAIN_SESSIONS:
        logger.warning(
            f"Training window has only {n_sessions} sessions (<{MIN_SAFE_TRAIN_SESSIONS} ~1yr). "
            "Optimized parameters on this little data are unreliable — treat with caution."
        )

    records = []
    for combo in combos:
        params = {**base_params, **combo}
        try:
            if strategy_cls.__name__ == "MomentumStrategy":
                strategy = strategy_cls(params, rebalance_freq=rebalance_freq)
            else:
                strategy = strategy_cls(params)
            signal = strategy.generate_signals(prices)
            engine = BacktestEngine(initial_capital, cost_model, position_sizing_cfg)
            result = engine.run(prices, signal)
            score = OBJECTIVES[objective](result.equity_curve, result.daily_net_return)
        except Exception as e:
            logger.warning(f"Combo {combo} failed: {e}")
            score = np.nan

        records.append({**combo, "objective_value": score})

    results = pd.DataFrame(records).sort_values("objective_value", ascending=False).reset_index(drop=True)

    valid = results["objective_value"].dropna()
    if len(valid) >= 5:
        best, median = valid.iloc[0], valid.median()
        if median != 0 and abs(best - median) > 3 * abs(median):
            logger.warning(
                "Best parameter combination is a large outlier vs the median result — "
                "likely an isolated optimum, not a robust plateau. Check sensitivity.py before trusting it."
            )

    return results


def best_params(results: pd.DataFrame, param_keys: List[str]) -> dict:
    """Extract the best parameter dict from an optimizer results DataFrame."""
    if results.empty or results["objective_value"].isna().all():
        raise ValueError("No valid parameter combination produced a finite objective value")
    top = results.iloc[0]
    return {k: top[k] for k in param_keys}
