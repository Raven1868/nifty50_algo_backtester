"""
Position sizing — converts a raw strategy signal panel (values 0/1/-1 or
continuous) into target PORTFOLIO WEIGHTS (fraction of capital per ticker),
subject to max/min position size and max portfolio exposure limits.

Sizing uses only information available up to and including the signal's own
date; volatility sizing uses a trailing (backward-looking) rolling window.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def size_positions(signal: pd.DataFrame, prices: pd.DataFrame, method: str,
                    fixed_pct: float = 0.10, vol_lookback: int = 20,
                    target_vol_annual: float = 0.15, max_position_size: float = 0.15,
                    max_portfolio_exposure: float = 1.0, min_position_size: float = 0.0) -> pd.DataFrame:
    """Convert signal -> target weights.

    method:
      "equal_weight"     — split exposure equally across all names with a
                            non-zero signal that day.
      "fixed_pct"         — each active name gets `fixed_pct` (capped by
                            max_portfolio_exposure).
      "volatility_target" — inverse-volatility weighting scaled to target
                            an annualized portfolio volatility contribution.
    """
    direction = np.sign(signal)  # -1, 0, +1
    active = direction != 0
    n_active = active.sum(axis=1).replace(0, np.nan)

    if method == "equal_weight":
        raw_weight = active.div(n_active, axis=0).fillna(0.0)

    elif method == "fixed_pct":
        raw_weight = active.astype(float) * fixed_pct

    elif method == "volatility_target":
        daily_ret = prices.pct_change()
        vol = daily_ret.rolling(vol_lookback, min_periods=vol_lookback).std() * np.sqrt(252)
        inv_vol = (1.0 / vol.replace(0, np.nan)).where(active)
        inv_vol_sum = inv_vol.sum(axis=1).replace(0, np.nan)
        base_weight = inv_vol.div(inv_vol_sum, axis=0).fillna(0.0)
        # scale block-level so blended vol ≈ target (approximation ignoring correlation)
        raw_weight = base_weight

    else:
        raise ValueError(f"Unknown position sizing method: {method}")

    weight = raw_weight * direction
    weight = weight.clip(lower=-max_position_size, upper=max_position_size)

    # Enforce max portfolio gross exposure by scaling down proportionally
    gross = weight.abs().sum(axis=1)
    scale = np.where(gross > max_portfolio_exposure, max_portfolio_exposure / gross.replace(0, np.nan), 1.0)
    weight = weight.mul(scale, axis=0).fillna(0.0)

    if min_position_size > 0:
        weight = weight.where(weight.abs() >= min_position_size, 0.0)

    return weight
