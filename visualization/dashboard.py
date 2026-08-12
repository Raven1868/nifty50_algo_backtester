"""
Streamlit Dashboard — thin UI layer only. All business logic (data loading,
signal generation, backtesting, cost model, analytics) is delegated to the
existing modules; this file contains no independent strategy or P&L logic.

Run with:
    streamlit run visualization/dashboard.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from data.universe import get_universe, NIFTY50_CURRENT
from data.loader import DataLoader
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from strategies.moving_average import MovingAverageStrategy
from backtesting.engine import BacktestEngine
from backtesting.costs import CostModel
from analytics.performance import full_summary
from analytics.trade_analysis import trade_statistics
from analytics.drawdown import drawdown_series
from validation.walk_forward import walk_forward_validate
from analytics.portfolio_analytics import (
    sector_exposure_timeseries, correlation_matrix, risk_contribution_snapshot, concentration_hhi,
)
from data.sector_mapping import map_tickers_to_sectors

st.set_page_config(page_title="NIFTY 50 Backtester", layout="wide")
st.title("📈 NIFTY 50 Algorithmic Trading Backtester")
st.caption("Free data (Yahoo Finance) · ₹0 cost · Educational / portfolio-demo use only — not investment advice.")

STRATEGIES = {
    "Moving Average Crossover": ("moving_average", MovingAverageStrategy,
                                  {"short_window": 20, "long_window": 100, "ma_type": "SMA"}),
    "Mean Reversion": ("mean_reversion", MeanReversionStrategy,
                        {"lookback": 20, "entry_z": -1.5, "exit_z": -0.3, "allow_short": False}),
    "Momentum (Cross-Sectional)": ("momentum", MomentumStrategy,
                                    {"lookback": 126, "top_n": 10, "skip_recent_days": 5}),
}

# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.header("Configuration")
    strategy_label = st.selectbox("Strategy", list(STRATEGIES.keys()))
    strategy_key, strategy_cls, default_params = STRATEGIES[strategy_label]

    universe_mode = st.radio("Universe", ["Full NIFTY 50", "Single ticker"])
    if universe_mode == "Single ticker":
        ticker = st.selectbox("Ticker", NIFTY50_CURRENT)
        tickers = [ticker]
    else:
        tickers = NIFTY50_CURRENT

    start_date = st.date_input("Start date", pd.Timestamp("2021-01-01"))
    end_date = st.date_input("End date", pd.Timestamp.today())
    capital = st.number_input("Initial capital (INR)", min_value=10000, value=1000000, step=50000)

    st.subheader("Strategy Parameters")
    params = {}
    if strategy_key == "moving_average":
        params["short_window"] = st.slider("Short window", 5, 60, default_params["short_window"])
        params["long_window"] = st.slider("Long window", 50, 250, default_params["long_window"])
        params["ma_type"] = st.selectbox("MA type", ["SMA", "EMA"])
    elif strategy_key == "mean_reversion":
        params["lookback"] = st.slider("Lookback", 5, 60, default_params["lookback"])
        params["entry_z"] = st.slider("Entry Z-score", -3.0, 0.0, default_params["entry_z"])
        params["exit_z"] = st.slider("Exit Z-score", -1.5, 1.0, default_params["exit_z"])
        params["allow_short"] = st.checkbox("Allow short leg", False)
    elif strategy_key == "momentum":
        params["lookback"] = st.slider("Lookback (days)", 20, 252, default_params["lookback"])
        params["top_n"] = st.slider("Top N stocks", 2, 20, default_params["top_n"])
        params["skip_recent_days"] = st.slider("Skip recent days", 0, 20, default_params["skip_recent_days"])

    rebalance_freq = st.selectbox("Rebalance frequency", ["D", "W", "M"], index=1)

    st.subheader("Costs & Slippage")
    tc_bps = st.number_input("Transaction cost (bps)", value=5.0)
    stt_bps = st.number_input("STT (bps)", value=10.0)
    slippage_bps = st.number_input("Slippage (bps)", value=5.0)

    st.subheader("Position Sizing")
    sizing_method = st.selectbox("Method", ["equal_weight", "fixed_pct", "volatility_target"])
    max_position = st.slider("Max position size", 0.02, 1.0, 0.15)

    run_button = st.button("Run Backtest", type="primary")
    run_wf = st.checkbox("Also run walk-forward validation (slower)", False)


@st.cache_data(show_spinner="Fetching price data...")
def load_prices(tickers, start, end):
    loader = DataLoader()
    raw = loader.load(tickers, str(start), str(end))
    prices = loader.to_panel(raw, field="Adj Close", fallback_field="Close").dropna(axis=1, how="all")
    bench_raw = loader.load(["^NSEI"], str(start), str(end))
    bench = loader.to_panel(bench_raw, field="Adj Close", fallback_field="Close").iloc[:, 0] if bench_raw else None
    return prices, bench


if run_button:
    prices, bench = load_prices(tuple(tickers), start_date, end_date)
    if prices.empty:
        st.error("No data returned. Check tickers, dates, and network access.")
        st.stop()

    strategy = (strategy_cls(params, rebalance_freq=rebalance_freq)
                if strategy_key == "momentum" else strategy_cls(params))
    signal = strategy.generate_signals(prices)

    cost_model = CostModel(transaction_cost_bps=tc_bps, stt_bps=stt_bps,
                            slippage_bps=slippage_bps, stamp_duty_bps=1.5, gst_pct_on_brokerage=0.18)
    sizing_cfg = {"method": sizing_method, "fixed_pct": 0.10, "vol_lookback": 20,
                  "target_vol_annual": 0.15, "max_position_size": max_position,
                  "max_portfolio_exposure": 1.0, "min_position_size": 0.0}
    engine = BacktestEngine(capital, cost_model, sizing_cfg)
    result = engine.run(prices, signal)

    summary = full_summary(result.equity_curve, result.daily_net_return, capital)
    trade_stats = trade_statistics(result.trade_log)

    # ---------------- Metrics ----------------
    cols = st.columns(5)
    cols[0].metric("Final Capital", f"₹{summary['Final Capital (INR)']:,.0f}")
    cols[1].metric("CAGR", f"{summary['CAGR (%)']:.2f}%")
    cols[2].metric("Sharpe", f"{summary['Sharpe Ratio']:.2f}")
    cols[3].metric("Max Drawdown", f"{summary['Max Drawdown (%)']:.2f}%")
    cols[4].metric("Trades", trade_stats.get("Number of Trades", 0))

    # ---------------- Equity curve ----------------
    st.subheader("Equity Curve vs Benchmark")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(result.equity_curve.index, result.equity_curve.values, label="Strategy")
    if bench is not None and not bench.empty and bench.notna().any():
        bench_norm = bench.reindex(result.equity_curve.index).ffill().bfill()
        if bench_norm.notna().any():
            bench_norm = bench_norm / bench_norm.dropna().iloc[0] * capital
            ax.plot(bench_norm.index, bench_norm.values, label="NIFTY 50", linestyle="--", color="gray")
        else:
            st.caption("⚠️ Benchmark data fetched but had no usable overlapping dates — line omitted.")
    else:
        st.caption("⚠️ Benchmark (NIFTY 50) data unavailable for this run — showing strategy only.")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Drawdown")
    fig2, ax2 = plt.subplots(figsize=(11, 2.5))
    dd = drawdown_series(result.equity_curve) * 100
    ax2.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.5)
    st.pyplot(fig2)

    st.subheader("Portfolio Exposure")
    fig3, ax3 = plt.subplots(figsize=(11, 2.5))
    ax3.plot(result.weights.index, result.weights.abs().sum(axis=1) * 100, color="green")
    st.pyplot(fig3)

    # ---------------- Tables ----------------
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Performance Summary")
        st.dataframe(pd.Series(summary, name="Value"))
    with c2:
        st.subheader("Trade Statistics")
        st.dataframe(pd.Series(trade_stats, name="Value"))

    st.subheader("Trade Log")
    st.dataframe(result.trade_log, use_container_width=True)
    st.download_button("Download Trade Log (CSV)", result.trade_log.to_csv(index=False),
                        file_name=f"{strategy_key}_trade_log.csv")

    # ---------------- Portfolio Analytics (multi-stock only) ----------------
    if prices.shape[1] > 1:
        st.subheader("Sector Exposure Over Time")
        sector_map = map_tickers_to_sectors(prices.columns.tolist())
        sector_ts = sector_exposure_timeseries(result.weights, sector_map)
        st.area_chart(sector_ts.clip(lower=0))

        held = result.weights.iloc[-1][result.weights.iloc[-1] != 0].index.tolist()
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Correlation Matrix (trailing 252d)")
            if held:
                corr = correlation_matrix(prices[held], as_of=result.weights.index[-1], lookback=252)
                fig4, ax4 = plt.subplots(figsize=(5, 5))
                im = ax4.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
                ax4.set_xticks(range(len(corr.columns))); ax4.set_xticklabels(corr.columns, rotation=90, fontsize=6)
                ax4.set_yticks(range(len(corr.index))); ax4.set_yticklabels(corr.index, fontsize=6)
                fig4.colorbar(im, ax=ax4, shrink=0.75)
                st.pyplot(fig4)
        with c4:
            st.subheader("Risk Contribution (trailing 60d)")
            if held:
                rc = risk_contribution_snapshot(result.weights.iloc[-1], prices, as_of=result.weights.index[-1], lookback=60)
                st.dataframe(rc)

        st.subheader("Portfolio Concentration (HHI)")
        st.line_chart(concentration_hhi(result.weights))

    # ---------------- Walk-forward (optional) ----------------
    if run_wf:
        st.subheader("Walk-Forward Validation (Out-of-Sample)")
        param_grid = {"short_window": [10, 20, 30]} if strategy_key == "moving_average" else \
                     {"lookback": [10, 20, 30]} if strategy_key == "mean_reversion" else \
                     {"top_n": [5, 10, 15]}
        try:
            with st.spinner("Running walk-forward folds..."):
                wf = walk_forward_validate(
                    strategy_cls, params, param_grid, prices, capital, cost_model, sizing_cfg,
                    train_window=504, test_window=63, rebalance_freq=rebalance_freq,
                )
            st.line_chart(wf.oos_equity_curve)
            st.dataframe(wf.period_log)
            st.json(wf.oos_summary)
        except Exception as e:
            st.warning(f"Walk-forward validation could not complete: {e}")
else:
    st.info("Configure your backtest in the sidebar and click **Run Backtest**.")
