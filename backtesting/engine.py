"""
Vectorized Backtesting Engine.

*** THE SINGLE PLACE WHERE THE t -> t+1 EXECUTION SHIFT HAPPENS ***
Strategies emit a signal using information available up to and including
day t (SIGNAL DATE). This engine applies `target_weight.shift(1)` so that
the weight decided using day-t information is only realized against day
t+1's return (EXECUTION happens at t+1, return earned is close[t+1] vs
close[t]). This is the one and only shift in the codebase — see
tests/test_lookahead.py for the regression test that verifies it.

All operations are vectorized (pandas/NumPy) across the full date x ticker
panel — no per-day Python loops over individual securities.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from backtesting.costs import CostModel
from backtesting.positions import size_positions


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    daily_gross_return: pd.Series
    daily_net_return: pd.Series
    weights: pd.DataFrame              # execution weights (already t+1-shifted)
    turnover: pd.Series
    costs: pd.Series
    trade_log: pd.DataFrame
    initial_capital: float


class BacktestEngine:
    def __init__(self, initial_capital: float, cost_model: CostModel,
                 position_sizing_cfg: dict):
        self.initial_capital = initial_capital
        self.cost_model = cost_model
        self.sizing_cfg = position_sizing_cfg

    def run(self, prices: pd.DataFrame, signal: pd.DataFrame,
            volume: "pd.DataFrame | None" = None) -> BacktestResult:
        """
        Args:
            prices: date x ticker adjusted close panel.
            signal: date x ticker raw strategy signal panel.
            volume: OPTIONAL date x ticker share-volume panel. If provided
                and cost_model.volume_aware_slippage is True, slippage
                scales up for trades large relative to trailing average
                daily traded value (see backtesting/costs.py). If omitted,
                behavior is identical to the flat-cost-fraction model.
        """
        prices = prices.sort_index()
        signal = signal.reindex_like(prices).fillna(0.0)

        target_weight = size_positions(
            signal, prices,
            method=self.sizing_cfg.get("method", "equal_weight"),
            fixed_pct=self.sizing_cfg.get("fixed_pct", 0.10),
            vol_lookback=self.sizing_cfg.get("vol_lookback", 20),
            target_vol_annual=self.sizing_cfg.get("target_vol_annual", 0.15),
            max_position_size=self.sizing_cfg.get("max_position_size", 0.15),
            max_portfolio_exposure=self.sizing_cfg.get("max_portfolio_exposure", 1.0),
            min_position_size=self.sizing_cfg.get("min_position_size", 0.0),
        )

        # === THE CRITICAL LOOKAHEAD-PREVENTION LINE ===
        execution_weight = target_weight.shift(1).fillna(0.0)

        daily_return = prices.pct_change().fillna(0.0)

        # Turnover: change in weight day-over-day requires trading
        prev_weight = execution_weight.shift(1).fillna(0.0)
        weight_change = (execution_weight - prev_weight).abs()
        turnover = weight_change.sum(axis=1)  # fraction of capital traded that day

        equity_curve_provisional = self.initial_capital * (1 + (execution_weight * daily_return).sum(axis=1)).cumprod()
        prev_equity = equity_curve_provisional.shift(1).fillna(self.initial_capital)

        if volume is not None and self.cost_model.volume_aware_slippage:
            costs_inr = self._volume_aware_costs(weight_change, prices, volume, prev_equity)
        else:
            cost_pct_of_capital = turnover * self.cost_model.cost_fraction()
            costs_inr = cost_pct_of_capital * prev_equity

        gross_return = (execution_weight * daily_return).sum(axis=1)
        cost_pct_of_capital = (costs_inr / prev_equity.replace(0, np.nan)).fillna(0.0)
        net_return = gross_return - cost_pct_of_capital

        equity_curve = self.initial_capital * (1 + net_return).cumprod()

        trade_log = self._build_trade_log(execution_weight, prices, equity_curve)

        return BacktestResult(
            equity_curve=equity_curve,
            daily_gross_return=gross_return,
            daily_net_return=net_return,
            weights=execution_weight,
            turnover=turnover,
            costs=costs_inr,
            trade_log=trade_log,
            initial_capital=self.initial_capital,
        )

    def _volume_aware_costs(self, weight_change: pd.DataFrame, prices: pd.DataFrame,
                             volume: pd.DataFrame, prev_equity: pd.Series) -> pd.Series:
        """Vectorized per-ticker, per-day cost using CostModel.apply_volume_aware
        logic, driven by a TRAILING (shift(1)) average daily traded value so
        no same-day volume information leaks into the cost estimate."""
        volume = volume.reindex_like(prices).fillna(0.0)
        dollar_volume = (prices * volume).rolling(20, min_periods=5).mean().shift(1)

        traded_notional = weight_change.mul(prev_equity, axis=0)

        participation = (traded_notional.abs() / dollar_volume.replace(0, np.nan))
        slippage_frac = self.cost_model.slippage_bps / 10000
        non_slippage_frac = self.cost_model.non_slippage_fraction()
        excess_ratio = (participation / self.cost_model.max_participation_rate).clip(lower=1.0).fillna(1.0)
        effective_slippage_frac = slippage_frac * excess_ratio

        cost_per_ticker = traded_notional.abs() * (non_slippage_frac + effective_slippage_frac)
        return cost_per_ticker.sum(axis=1)

    @staticmethod
    def _build_trade_log(weights: pd.DataFrame, prices: pd.DataFrame,
                          equity_curve: pd.Series) -> pd.DataFrame:
        """Reconstruct discrete trades (entry->exit) per ticker from the
        continuous weight series, for reporting purposes."""
        records = []
        for ticker in weights.columns:
            w = weights[ticker]
            in_pos = w != 0
            changes = in_pos.astype(int).diff().fillna(0)
            entries = w.index[changes == 1]
            exits = w.index[changes == -1]

            # align entries/exits (handle position open at end of series)
            exit_iter = list(exits)
            for entry in entries:
                exit_date = next((e for e in exit_iter if e > entry), None)
                entry_price = prices.at[entry, ticker] if entry in prices.index else np.nan
                if exit_date is not None:
                    exit_price = prices.at[exit_date, ticker] if exit_date in prices.index else np.nan
                    holding_days = (exit_date - entry).days
                else:
                    exit_price = prices[ticker].iloc[-1]
                    exit_date = prices.index[-1]
                    holding_days = (exit_date - entry).days

                direction = float(np.sign(w.loc[entry])) if entry in w.index else 1.0
                entry_price_f = float(entry_price) if pd.notna(entry_price) else float("nan")  # type: ignore[arg-type]
                exit_price_f = float(exit_price) if pd.notna(exit_price) else float("nan")  # type: ignore[arg-type]
                gross_ret_pct = direction * (exit_price_f / entry_price_f - 1) if entry_price_f else float("nan")

                records.append({
                    "symbol": ticker,
                    "entry_date": entry,
                    "exit_date": exit_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "direction": "LONG" if direction >= 0 else "SHORT",
                    "holding_days": holding_days,
                    "gross_return_pct": round(gross_ret_pct * 100, 3) if pd.notna(gross_ret_pct) else np.nan,
                })

        log = pd.DataFrame(records)
        if not log.empty:
            log = log.sort_values("entry_date").reset_index(drop=True)
        return log
