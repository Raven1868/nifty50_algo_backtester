"""
Walk-Forward Validation.

Splits the price history into successive TRAIN -> TEST windows. On each
TRAIN window, parameters are optimized (grid search, see optimizer.py).
Those parameters are then FROZEN and applied, unmodified, to the following
unseen TEST window. The window then rolls (or expands) forward and the
process repeats. Combining every TEST-window result gives a single
out-of-sample equity curve that was never used for parameter selection —
this is the honest performance estimate; the in-sample numbers are shown
alongside only for comparison, never as the headline result.

window_type:
  "rolling"   — train window is a fixed number of sessions, slides forward
  "expanding" — train window always starts at date 0 and grows
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Type
import numpy as np
import pandas as pd

from backtesting.engine import BacktestEngine
from backtesting.costs import CostModel
from analytics import performance as perf
from validation.optimizer import optimize_strategy, best_params

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    oos_equity_curve: pd.Series
    oos_daily_returns: pd.Series
    period_log: pd.DataFrame          # one row per fold: train/test dates, chosen params, IS vs OOS metrics
    is_summary: dict
    oos_summary: dict


def walk_forward_validate(
    strategy_cls: Type, base_params: dict, param_grid: Dict[str, List],
    prices: pd.DataFrame, initial_capital: float, cost_model: CostModel,
    position_sizing_cfg: dict, train_window: int = 756, test_window: int = 126,
    step: "int | None" = None, window_type: str = "rolling", objective: str = "sharpe",
    rebalance_freq: str = "W",
) -> WalkForwardResult:
    step = step or test_window
    dates = prices.index
    n = len(dates)

    if train_window + test_window > n:
        raise ValueError(
            f"train_window({train_window}) + test_window({test_window}) exceeds "
            f"available sessions ({n}). Shorten windows or extend the price history."
        )

    fold_records = []
    oos_returns_segments = []

    start = 0
    fold_num = 0
    while start + train_window + test_window <= n:
        fold_num += 1
        train_start_idx = 0 if window_type == "expanding" else start
        train_end_idx = start + train_window
        test_end_idx = train_end_idx + test_window

        train_prices = prices.iloc[train_start_idx:train_end_idx]
        test_prices = prices.iloc[max(0, train_end_idx - _max_lookback(param_grid, base_params)):test_end_idx]

        logger.info(
            f"Fold {fold_num}: train {train_prices.index[0].date()}–{train_prices.index[-1].date()} "
            f"({len(train_prices)} sessions) | test {dates[train_end_idx].date()}–{dates[test_end_idx-1].date()}"
        )

        opt_results = optimize_strategy(
            strategy_cls, base_params, param_grid, train_prices, initial_capital,
            cost_model, position_sizing_cfg, objective=objective, rebalance_freq=rebalance_freq,
        )
        chosen = best_params(opt_results, list(param_grid.keys()))
        full_params = {**base_params, **chosen}

        # In-sample performance with chosen (frozen) params, on the train window
        is_strategy = _build_strategy(strategy_cls, full_params, rebalance_freq)
        is_signal = is_strategy.generate_signals(train_prices)
        is_engine = BacktestEngine(initial_capital, cost_model, position_sizing_cfg)
        is_result = is_engine.run(train_prices, is_signal)

        # Out-of-sample: apply the SAME frozen params to test_prices (which
        # includes enough pre-test history for indicator warm-up, but the
        # OOS segment we KEEP starts strictly at train_end_idx).
        oos_strategy = _build_strategy(strategy_cls, full_params, rebalance_freq)
        oos_signal = oos_strategy.generate_signals(test_prices)
        oos_engine = BacktestEngine(initial_capital, cost_model, position_sizing_cfg)
        oos_result_full = oos_engine.run(test_prices, oos_signal)

        keep_from = dates[train_end_idx]
        oos_ret_segment = oos_result_full.daily_net_return.loc[oos_result_full.daily_net_return.index >= keep_from]
        oos_returns_segments.append(oos_ret_segment)

        fold_records.append({
            "fold": fold_num,
            "train_start": train_prices.index[0], "train_end": train_prices.index[-1],
            "test_start": keep_from, "test_end": dates[test_end_idx - 1],
            **{f"param_{k}": v for k, v in chosen.items()},
            "IS_objective": opt_results.iloc[0]["objective_value"],
            "OOS_sharpe": perf.sharpe_ratio(oos_ret_segment) if len(oos_ret_segment) > 5 else np.nan,
            "OOS_cagr_proxy_return_pct": round(((1 + oos_ret_segment).prod() - 1) * 100, 2),
        })

        start += step

    if not oos_returns_segments:
        raise ValueError("No walk-forward folds could be generated — check window sizes vs data length.")

    combined_oos_returns = pd.concat(oos_returns_segments)
    combined_oos_returns = combined_oos_returns[~combined_oos_returns.index.duplicated(keep="first")].sort_index()
    oos_equity = initial_capital * (1 + combined_oos_returns).cumprod()

    period_log = pd.DataFrame(fold_records)
    is_summary_row = period_log["IS_objective"].mean()
    oos_summary = perf.full_summary(oos_equity, combined_oos_returns, initial_capital)

    logger.info(
        f"Walk-forward complete: {fold_num} folds | mean IS objective={is_summary_row:.3f} "
        f"| OOS Sharpe={oos_summary.get('Sharpe Ratio')}"
    )

    return WalkForwardResult(
        oos_equity_curve=oos_equity,
        oos_daily_returns=combined_oos_returns,
        period_log=period_log,
        is_summary={"mean_IS_objective": round(is_summary_row, 3) if pd.notna(is_summary_row) else np.nan},
        oos_summary=oos_summary,
    )


def _build_strategy(strategy_cls: Type, params: dict, rebalance_freq: str):
    if strategy_cls.__name__ == "MomentumStrategy":
        return strategy_cls(params, rebalance_freq=rebalance_freq)
    return strategy_cls(params)


def _max_lookback(param_grid: Dict[str, List], base_params: dict) -> int:
    """Estimate the longest lookback-like parameter so the OOS test slice
    includes enough warm-up history for indicators to be valid from its
    first day (without this, the first sessions of each fold would be
    silently flat due to NaN indicators, understating OOS activity)."""
    candidates = []
    all_params = {**base_params, **{k: max(v) for k, v in param_grid.items() if all(isinstance(x, (int, float)) for x in v)}}
    for key in ("lookback", "long_window", "short_window"):
        if key in all_params:
            candidates.append(int(all_params[key]))
    return max(candidates) if candidates else 0
