"""
Main CLI entry point.

Examples:
    python main.py --strategy momentum --universe nifty50
    python main.py --strategy moving_average --ticker RELIANCE.NS
    python main.py --strategy mean_reversion --universe nifty50 --capital 500000
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
import pandas as pd

from config_schema import load_and_validate_config
from data.universe import get_universe
from data.loader import DataLoader
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from strategies.moving_average import MovingAverageStrategy
from backtesting.engine import BacktestEngine
from backtesting.costs import CostModel
from analytics.performance import full_summary
from analytics.drawdown import drawdown_periods
from analytics.trade_analysis import trade_statistics
from visualization.charts import (
    generate_all_charts, plot_parameter_sensitivity_heatmap, plot_walk_forward_equity,
    plot_sector_exposure_stacked, plot_correlation_heatmap, plot_risk_contribution_bar,
    plot_rolling_beta, plot_concentration,
)
from validation.walk_forward import walk_forward_validate
from validation.sensitivity import parameter_sensitivity_2d, plateau_diagnostic
from analytics.regime import classify_regimes, performance_by_regime
from analytics.portfolio_analytics import (
    sector_exposure_timeseries, sector_exposure_snapshot, correlation_matrix,
    risk_contribution_snapshot, sector_risk_contribution, concentration_hhi, rolling_beta,
)
from data.sector_mapping import map_tickers_to_sectors

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main")

STRATEGY_MAP = {
    "mean_reversion": MeanReversionStrategy,
    "momentum": MomentumStrategy,
    "moving_average": MovingAverageStrategy,
}


def load_config(path: str = "config.yaml") -> dict:
    """Validates config.yaml against config_schema.AppConfig (fails fast with
    a clear message on typos/bad values), then returns a plain dict so the
    rest of the pipeline's existing cfg["section"]["key"] access is unchanged."""
    try:
        validated = load_and_validate_config(path)
    except Exception as e:
        logger.error(f"config.yaml failed validation:\n{e}")
        sys.exit(1)
    return validated.model_dump()


def build_strategy(name: str, cfg: dict):
    params = cfg["strategy"].get(name, {})
    if name == "momentum":
        return MomentumStrategy(params, rebalance_freq=cfg["backtest"]["rebalance_freq"])
    return STRATEGY_MAP[name](params)


def setup_file_logging(logs_dir: str, strategy_name: str) -> str:
    """Adds a file handler so every run leaves a permanent, timestamped audit
    trail (data warnings, cost assumptions, walk-forward folds, etc.) in
    addition to the human-readable console report. Returns the log file path."""
    os.makedirs(logs_dir, exist_ok=True)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f"{strategy_name}_{ts}.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)
    return log_path


def run(args):
    cfg = load_config(args.config)

    if args.capital:
        cfg["backtest"]["initial_capital"] = args.capital
    if args.start:
        cfg["backtest"]["start_date"] = args.start
    if args.end:
        cfg["backtest"]["end_date"] = args.end

    strategy_name = args.strategy or cfg["strategy"]["name"]
    if strategy_name not in STRATEGY_MAP:
        logger.error(f"Unknown strategy '{strategy_name}'. Choose from {list(STRATEGY_MAP)}")
        sys.exit(1)

    log_path = setup_file_logging(cfg["output"].get("logs_dir", "logs"), strategy_name)
    logger.info(f"Audit log for this run: {log_path}")

    # --- Universe ---
    if args.ticker:
        tickers = [args.ticker]
    elif args.universe == "custom":
        tickers = cfg["universe"]["custom_tickers"]
    else:
        tickers = get_universe(mode=cfg["universe"]["mode"], custom_tickers=cfg["universe"].get("custom_tickers"))

    logger.info(f"Universe: {len(tickers)} ticker(s) | Strategy: {strategy_name}")

    # --- Data ---
    loader = DataLoader(cache_dir=cfg["output"]["cache_dir"], processed_dir=cfg["output"]["processed_dir"])
    start, end = cfg["backtest"]["start_date"], cfg["backtest"]["end_date"]
    raw_data = loader.load(tickers, start, end)
    if not raw_data:
        logger.error("No data fetched for any ticker. Check tickers / network / date range.")
        sys.exit(1)
    prices = loader.to_panel(raw_data, field="Adj Close", fallback_field="Close")
    prices = prices.dropna(axis=1, how="all")
    volume_panel = loader.to_panel(raw_data, field="Volume") if cfg["execution"].get("volume_aware_slippage") else None
    logger.info(f"Price panel: {prices.shape[0]} sessions x {prices.shape[1]} tickers usable")

    benchmark_data = loader.load([cfg["benchmark"]["ticker"]], start, end)
    benchmark_prices = loader.to_panel(benchmark_data, field="Adj Close", fallback_field="Close").iloc[:, 0] if benchmark_data else None
    if benchmark_prices is not None and benchmark_prices.notna().any():
        benchmark_norm = benchmark_prices.dropna()
        benchmark_norm = benchmark_norm / benchmark_norm.iloc[0] * cfg["backtest"]["initial_capital"]
    else:
        if benchmark_data:
            logger.warning(f"Benchmark '{cfg['benchmark']['ticker']}' returned no usable price data — benchmark comparison will be skipped.")
        benchmark_norm = None

    # --- Strategy signal ---
    strategy = build_strategy(strategy_name, cfg)
    signal = strategy.generate_signals(prices)

    # --- Backtest ---
    cost_model = CostModel(**cfg["execution"])
    engine = BacktestEngine(
        initial_capital=cfg["backtest"]["initial_capital"],
        cost_model=cost_model,
        position_sizing_cfg=cfg["position_sizing"],
    )
    result = engine.run(prices, signal, volume=volume_panel)

    # --- Analytics ---
    summary = full_summary(result.equity_curve, result.daily_net_return, result.initial_capital)
    gross_summary = full_summary(
        (1 + result.daily_gross_return).cumprod() * result.initial_capital,
        result.daily_gross_return, result.initial_capital,
    )
    trade_stats = trade_statistics(result.trade_log)
    dd_periods = drawdown_periods(result.equity_curve)

    if benchmark_norm is not None:
        bench_ret = benchmark_norm.pct_change().reindex(result.daily_net_return.index).fillna(0.0)
        bench_summary = full_summary(benchmark_norm.reindex(result.equity_curve.index).ffill(), bench_ret, cfg["backtest"]["initial_capital"])
    else:
        bench_summary = {}

    # --- Output ---
    os.makedirs(cfg["output"]["reports_dir"], exist_ok=True)
    reports_dir = cfg["output"]["reports_dir"]

    print("\n" + "=" * 60)
    print(f"BACKTEST SUMMARY — {strategy_name.upper()} | {start} to {end}")
    print("=" * 60)
    print("\n--- NET (after costs & slippage) ---")
    for k, v in summary.items():
        print(f"{k:35s}: {v}")
    print("\n--- GROSS (before costs) ---")
    for k, v in gross_summary.items():
        print(f"{k:35s}: {v}")
    if bench_summary:
        print("\n--- BENCHMARK (NIFTY 50 buy & hold) ---")
        for k, v in bench_summary.items():
            print(f"{k:35s}: {v}")
    print("\n--- TRADE STATISTICS ---")
    for k, v in trade_stats.items():
        print(f"{k:35s}: {v}")
    print("=" * 60 + "\n")

    # Also log the same summary to the audit file (so results, not just
    # process events, are captured in the permanent record of the run)
    logger.info(
        "RUN SUMMARY | strategy=%s | net=%s | gross=%s | benchmark=%s | trades=%s",
        strategy_name, summary, gross_summary, bench_summary or "n/a", trade_stats,
    )

    # Save trade log
    trade_log_path_csv = os.path.join(reports_dir, f"{strategy_name}_trade_log.csv")
    trade_log_path_xlsx = os.path.join(reports_dir, f"{strategy_name}_trade_log.xlsx")
    result.trade_log.to_csv(trade_log_path_csv, index=False)
    try:
        result.trade_log.to_excel(trade_log_path_xlsx, index=False)
    except Exception as e:
        logger.warning(f"Could not write Excel trade log ({e}); CSV was saved.")

    # Save metrics summary
    summary_df = pd.DataFrame([
        {"Metric": k, "Net": v, "Gross": gross_summary.get(k), "Benchmark": bench_summary.get(k)}
        for k, v in summary.items()
    ])
    summary_df.to_csv(os.path.join(reports_dir, f"{strategy_name}_summary.csv"), index=False)
    dd_periods.to_csv(os.path.join(reports_dir, f"{strategy_name}_drawdown_periods.csv"), index=False)

    # Charts
    generate_all_charts(
        result.equity_curve, benchmark_norm, result.daily_net_return,
        result.weights, result.turnover, result.trade_log,
        charts_dir=cfg["output"]["charts_dir"], strategy_name=strategy_name,
    )

    logger.info(f"Reports saved to '{reports_dir}/', charts saved to '{cfg['output']['charts_dir']}/'")

    # --- Portfolio analytics: sector exposure, correlation, risk contribution ---
    if args.portfolio_analytics:
        if prices.shape[1] < 2:
            logger.warning("Portfolio analytics needs >1 ticker; skipping (single-ticker run).")
        else:
            logger.info("Running sector exposure / correlation / risk contribution analytics...")
            sector_map = map_tickers_to_sectors(prices.columns.tolist())

            # Sector exposure over time
            sector_ts = sector_exposure_timeseries(result.weights, sector_map)
            sector_ts.to_csv(os.path.join(reports_dir, f"{strategy_name}_sector_exposure_timeseries.csv"))
            plot_sector_exposure_stacked(sector_ts, os.path.join(cfg["output"]["charts_dir"], f"{strategy_name}_sector_exposure.png"))

            last_date = result.weights.index[-1]
            sector_snap = sector_exposure_snapshot(result.weights.iloc[-1], sector_map)
            sector_snap.to_csv(os.path.join(reports_dir, f"{strategy_name}_sector_exposure_latest.csv"))
            print(f"\n--- SECTOR EXPOSURE (as of {last_date.date()}) ---")
            print(sector_snap.to_string())

            # Correlation matrix (trailing 252 sessions of the invested universe)
            held_tickers = result.weights.iloc[-1][result.weights.iloc[-1] != 0].index.tolist()
            corr_universe = held_tickers if held_tickers else prices.columns.tolist()
            corr = correlation_matrix(prices[corr_universe], as_of=last_date, lookback=252)
            corr.to_csv(os.path.join(reports_dir, f"{strategy_name}_correlation_matrix.csv"))
            plot_correlation_heatmap(corr, os.path.join(cfg["output"]["charts_dir"], f"{strategy_name}_correlation.png"))

            # Risk contribution (trailing 60 sessions), by position and by sector
            risk_contrib = risk_contribution_snapshot(result.weights.iloc[-1], prices, as_of=last_date, lookback=60)
            risk_contrib.to_csv(os.path.join(reports_dir, f"{strategy_name}_risk_contribution.csv"))
            plot_risk_contribution_bar(risk_contrib, os.path.join(cfg["output"]["charts_dir"], f"{strategy_name}_risk_contribution.png"))
            print(f"\n--- RISK CONTRIBUTION BY POSITION (trailing 60d, as of {last_date.date()}) ---")
            print(risk_contrib.to_string())

            if not risk_contrib.empty:
                sector_risk = sector_risk_contribution(risk_contrib, sector_map)
                sector_risk.to_csv(os.path.join(reports_dir, f"{strategy_name}_sector_risk_contribution.csv"))
                print("\n--- RISK CONTRIBUTION BY SECTOR ---")
                print(sector_risk.to_string())

            # Concentration (HHI) over time
            hhi = concentration_hhi(result.weights)
            hhi.to_csv(os.path.join(reports_dir, f"{strategy_name}_concentration_hhi.csv"))
            plot_concentration(hhi, os.path.join(cfg["output"]["charts_dir"], f"{strategy_name}_concentration.png"))
            print(f"\nLatest concentration (HHI): {hhi.iloc[-1]:.4f}  "
                  f"(1/{len(held_tickers) if held_tickers else '-'} = {1/len(held_tickers):.4f} if perfectly equal-weighted)"
                  if held_tickers else "")

            # Rolling beta vs benchmark
            if benchmark_prices is not None:
                bench_ret = benchmark_prices.pct_change().reindex(result.daily_net_return.index)
                beta_series = rolling_beta(result.daily_net_return, bench_ret, window=126)
                beta_series.to_csv(os.path.join(reports_dir, f"{strategy_name}_rolling_beta.csv"))
                plot_rolling_beta(beta_series.dropna(), os.path.join(cfg["output"]["charts_dir"], f"{strategy_name}_rolling_beta.png"), window=126)
                if beta_series.dropna().shape[0]:
                    print(f"\nLatest 126-day rolling beta vs NIFTY 50: {beta_series.dropna().iloc[-1]:.2f}")

    # --- Regime analysis (optional) ---
    if args.regime and benchmark_prices is not None:
        logger.info("Running market regime analysis...")
        regimes = classify_regimes(benchmark_prices.reindex(result.daily_net_return.index).ffill())
        regime_perf = performance_by_regime(result.daily_net_return, regimes)
        regime_perf.to_csv(os.path.join(reports_dir, f"{strategy_name}_regime_performance.csv"), index=False)
        print("\n--- PERFORMANCE BY MARKET REGIME ---")
        print(regime_perf.to_string(index=False))

    # --- Parameter sensitivity (optional) ---
    if args.sensitivity:
        logger.info("Running parameter sensitivity analysis...")
        grid_cfg = cfg["validation"]["param_grids"].get(strategy_name)
        if grid_cfg and len(grid_cfg) >= 2:
            keys = list(grid_cfg.keys())[:2]
            matrices = parameter_sensitivity_2d(
                STRATEGY_MAP[strategy_name], cfg["strategy"].get(strategy_name, {}),
                keys[0], grid_cfg[keys[0]], keys[1], grid_cfg[keys[1]],
                prices, cfg["backtest"]["initial_capital"], cost_model, cfg["position_sizing"],
                rebalance_freq=cfg["backtest"]["rebalance_freq"],
            )
            for metric_name, matrix in matrices.items():
                matrix.to_csv(os.path.join(reports_dir, f"{strategy_name}_sensitivity_{metric_name}.csv"))
                plot_parameter_sensitivity_heatmap(
                    matrix, os.path.join(cfg["output"]["charts_dir"], f"{strategy_name}_sensitivity_{metric_name}.png"),
                    title=f"{strategy_name} — {metric_name.upper()} sensitivity",
                )
            print("\n--- PARAMETER SENSITIVITY (Sharpe grid) ---")
            print(matrices["sharpe"])
            print(plateau_diagnostic(matrices["sharpe"]))
        else:
            logger.warning(f"No 2-parameter grid configured for '{strategy_name}' in config.yaml validation.param_grids")

    # --- Walk-forward validation (optional) ---
    if args.walkforward:
        logger.info("Running walk-forward validation (this can take a while)...")
        vcfg = cfg["validation"]
        grid = vcfg["param_grids"].get(strategy_name, {})
        if not grid:
            logger.warning(f"No parameter grid configured for '{strategy_name}'; skipping walk-forward.")
        else:
            wf = walk_forward_validate(
                STRATEGY_MAP[strategy_name], cfg["strategy"].get(strategy_name, {}), grid, prices,
                cfg["backtest"]["initial_capital"], cost_model, cfg["position_sizing"],
                train_window=vcfg["train_window"], test_window=vcfg["test_window"], step=vcfg["step"],
                window_type=vcfg["window_type"], objective=vcfg["objective"],
                rebalance_freq=cfg["backtest"]["rebalance_freq"],
            )
            wf.period_log.to_csv(os.path.join(reports_dir, f"{strategy_name}_walkforward_folds.csv"), index=False)
            pd.DataFrame([wf.oos_summary]).to_csv(os.path.join(reports_dir, f"{strategy_name}_walkforward_oos_summary.csv"), index=False)
            plot_walk_forward_equity(
                wf.oos_equity_curve, os.path.join(cfg["output"]["charts_dir"], f"{strategy_name}_walkforward_equity.png"),
                fold_boundaries=wf.period_log["test_start"].tolist(),
            )
            print("\n--- WALK-FORWARD: IN-SAMPLE (selection) vs OUT-OF-SAMPLE (honest) ---")
            print(f"Mean in-sample objective across folds : {wf.is_summary['mean_IS_objective']}")
            print("Out-of-sample summary (never used for parameter selection):")
            for k, v in wf.oos_summary.items():
                print(f"{k:35s}: {v}")
            print("\nPer-fold detail:")
            print(wf.period_log.to_string(index=False))


def parse_args():
    p = argparse.ArgumentParser(description="NIFTY 50 Algorithmic Trading Backtester (₹0 cost, free data)")
    p.add_argument("--strategy", choices=list(STRATEGY_MAP.keys()), help="Strategy to run")
    p.add_argument("--universe", choices=["nifty50", "custom"], default="nifty50")
    p.add_argument("--ticker", help="Single-ticker mode, e.g. RELIANCE.NS (overrides --universe)")
    p.add_argument("--capital", type=float, help="Initial capital in INR")
    p.add_argument("--start", help="Start date YYYY-MM-DD (overrides config.yaml)")
    p.add_argument("--end", help="End date YYYY-MM-DD (overrides config.yaml)")
    p.add_argument("--config", default="config.yaml", help="Path to config file")
    p.add_argument("--walkforward", action="store_true", help="Run walk-forward validation (train/test rolling folds)")
    p.add_argument("--sensitivity", action="store_true", help="Run 2D parameter sensitivity analysis")
    p.add_argument("--regime", action="store_true", help="Break down performance by market regime (bull/bear/vol)")
    p.add_argument("--portfolio-analytics", action="store_true", dest="portfolio_analytics",
                    help="Sector exposure, correlation matrix, risk contribution, concentration, rolling beta")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
