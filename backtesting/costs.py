"""
Transaction cost model for Indian equity delivery trades. All rates are
ILLUSTRATIVE DEFAULTS taken from config.yaml — verify against current
NSE/SEBI/CBIC notifications before using for real capital decisions.

Cost = brokerage/exchange charges (bps) + STT (bps) + stamp duty (bps)
       + GST on brokerage + slippage (bps)
All charges are one-way (applied per buy and per sell separately, as in
real trading) and are applied to the notional value traded.

VOLUME-AWARE SLIPPAGE (optional, `volume_aware_slippage: true` in config):
A flat slippage assumption (e.g. "5 bps always") is unrealistic once trade
size becomes large relative to a stock's typical liquidity — real market
impact grows with participation rate (traded notional / average daily
traded value). When enabled, the SLIPPAGE component alone scales up linearly
once participation exceeds `max_participation_rate`; all other cost
components (brokerage, STT, stamp duty, GST) remain flat as those are
percentage-of-notional charges in real life, not liquidity-dependent.
This is a simplified linear impact model, not a calibrated market-impact
model (e.g. Almgren-Chriss) — treat it as directionally realistic, not exact.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class CostModel:
    transaction_cost_bps: float = 5.0
    stt_bps: float = 10.0
    stamp_duty_bps: float = 1.5
    gst_pct_on_brokerage: float = 0.18
    slippage_bps: float = 5.0
    volume_aware_slippage: bool = False
    max_participation_rate: float = 0.10

    def cost_fraction(self) -> float:
        """Total one-way cost as a fraction of notional traded, at the flat
        (non-volume-aware) slippage assumption."""
        brokerage = self.transaction_cost_bps / 10000
        gst = brokerage * self.gst_pct_on_brokerage
        stt = self.stt_bps / 10000
        stamp = self.stamp_duty_bps / 10000
        slippage = self.slippage_bps / 10000
        return brokerage + gst + stt + stamp + slippage

    def non_slippage_fraction(self) -> float:
        """Everything except slippage — the part that does NOT scale with
        trade size relative to liquidity."""
        return self.cost_fraction() - (self.slippage_bps / 10000)

    def apply(self, traded_notional: float) -> float:
        """Return the total cost (INR) for a given absolute traded notional,
        using the flat cost_fraction (ignores volume-awareness — use
        `apply_volume_aware` for that)."""
        return abs(traded_notional) * self.cost_fraction()

    def apply_volume_aware(self, traded_notional: float, avg_dollar_volume: "float | None") -> float:
        """Cost (INR) for a trade, scaling the slippage component up when
        `traded_notional` is large relative to `avg_dollar_volume` (average
        daily traded value in INR). Falls back to the flat model if
        volume_aware_slippage is disabled or avg_dollar_volume is unusable."""
        notional = abs(traded_notional)
        if not self.volume_aware_slippage or avg_dollar_volume is None or avg_dollar_volume <= 0:
            return notional * self.cost_fraction()

        participation = notional / avg_dollar_volume
        slippage_frac = self.slippage_bps / 10000
        excess_ratio = max(1.0, participation / self.max_participation_rate)
        effective_slippage_frac = slippage_frac * excess_ratio

        return notional * (self.non_slippage_fraction() + effective_slippage_frac)

