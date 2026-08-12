"""
Professional financial visualizations. Uses the Agg backend (no display
needed — headless-safe, only saves PNGs).
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import os

from analytics.drawdown import drawdown_series

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})


def _inr_formatter(x, pos):
    return f"₹{x:,.0f}"


def plot_equity_curve(equity: pd.Series, benchmark: pd.Series, out_path: str, title: str):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(equity.index, equity.values, label="Strategy", linewidth=1.6, color="#1f6feb")
    if benchmark is not None:
        ax.plot(benchmark.index, benchmark.values, label="Benchmark (NIFTY 50)", linewidth=1.2, color="#888", linestyle="--")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_inr_formatter))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.set_title(title)
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_drawdown(equity: pd.Series, out_path: str, title: str = "Drawdown"):
    dd = drawdown_series(equity) * 100
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.fill_between(dd.index, dd.values, 0, color="#d1242f", alpha=0.5)
    ax.plot(dd.index, dd.values, color="#d1242f", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("Drawdown (%)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_rolling_sharpe(daily_returns: pd.Series, out_path: str, window: int = 63):
    rf_daily = 0.065 / 252
    excess = daily_returns - rf_daily
    rolling_sharpe = (excess.rolling(window).mean() / excess.rolling(window).std()) * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, color="#8250df")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_rolling_volatility(daily_returns: pd.Series, out_path: str, window: int = 21):
    rolling_vol = daily_returns.rolling(window).std() * np.sqrt(252) * 100
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(rolling_vol.index, rolling_vol.values, color="#bf8700")
    ax.set_title(f"Rolling {window}-Day Annualized Volatility (%)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_monthly_returns_heatmap(daily_returns: pd.Series, out_path: str):
    monthly = (1 + daily_returns).resample("ME").prod() - 1
    df = monthly.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="year", columns="month", values="ret") * 100
    pivot = pivot.reindex(columns=range(1, 13))

    fig, ax = plt.subplots(figsize=(11, max(2.5, 0.5 * len(pivot))))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=10)
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7)
    ax.set_title("Monthly Returns Heatmap (%)")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_exposure(weights: pd.DataFrame, out_path: str):
    gross_exposure = weights.abs().sum(axis=1) * 100
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(gross_exposure.index, gross_exposure.values, color="#1a7f37")
    ax.set_title("Gross Portfolio Exposure (%)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_turnover(turnover: pd.Series, out_path: str):
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.bar(turnover.index, turnover.values * 100, width=1.0, color="#57606a")
    ax.set_title("Daily Portfolio Turnover (%)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_trade_pnl_distribution(trade_log: pd.DataFrame, out_path: str):
    if trade_log.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(trade_log["gross_return_pct"].dropna(), bins=30, color="#1f6feb", alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Trade P&L Distribution (%)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_parameter_sensitivity_heatmap(matrix: pd.DataFrame, out_path: str, title: str, value_fmt: str = "{:.2f}"):
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(matrix.columns) + 2), max(4, 0.5 * len(matrix.index) + 2)))
    vals = matrix.values.astype(float)
    im = ax.imshow(vals, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel(matrix.columns.name or "")
    ax.set_ylabel(matrix.index.name or "")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if not np.isnan(v):
                ax.text(j, i, value_fmt.format(v), ha="center", va="center", fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_walk_forward_equity(oos_equity: pd.Series, out_path: str, fold_boundaries=None):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(oos_equity.index, oos_equity.values, color="#1f6feb", linewidth=1.6, label="Out-of-Sample Equity")
    if fold_boundaries is not None:
        for b in fold_boundaries:
            ax.axvline(b, color="#888", linestyle=":", linewidth=0.8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_inr_formatter))
    ax.set_title("Walk-Forward Out-of-Sample Equity Curve (dotted lines = fold boundaries)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_sector_exposure_stacked(sector_exposure: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(11, 5))
    positive = sector_exposure.clip(lower=0)
    ax.stackplot(positive.index, [positive[c] * 100 for c in positive.columns],
                 labels=positive.columns, alpha=0.85)
    ax.set_title("Sector Exposure Over Time (%)")
    ax.set_ylabel("Portfolio Weight (%)")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_correlation_heatmap(corr: pd.DataFrame, out_path: str, title: str = "Return Correlation Matrix"):
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(corr) + 2), max(5, 0.45 * len(corr) + 2)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=6)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_risk_contribution_bar(risk_contrib: pd.DataFrame, out_path: str, title: str = "Risk Contribution by Position (%)"):
    if risk_contrib.empty:
        return
    fig, ax = plt.subplots(figsize=(9, max(3, 0.3 * len(risk_contrib))))
    data = risk_contrib.sort_values("pct_risk_contribution")
    colors = np.where(data["pct_risk_contribution"] >= 0, "#1a7f37", "#d1242f")
    ax.barh(data.index, data["pct_risk_contribution"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_rolling_beta(beta: pd.Series, out_path: str, window: int):
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(beta.index, beta.values, color="#0969da")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(f"Rolling {window}-Day Beta vs Benchmark")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_concentration(hhi: pd.Series, out_path: str):
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(hhi.index, hhi.values, color="#9a6700")
    ax.set_title("Portfolio Concentration (Herfindahl-Hirschman Index)")
    ax.set_ylabel("HHI (1/N = diversified, 1.0 = concentrated)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def generate_all_charts(equity, benchmark, daily_returns, weights, turnover, trade_log,
                         charts_dir: str, strategy_name: str):
    os.makedirs(charts_dir, exist_ok=True)
    prefix = os.path.join(charts_dir, strategy_name)
    plot_equity_curve(equity, benchmark, f"{prefix}_equity_curve.png", f"{strategy_name} — Equity Curve")
    plot_drawdown(equity, f"{prefix}_drawdown.png")
    plot_rolling_sharpe(daily_returns, f"{prefix}_rolling_sharpe.png")
    plot_rolling_volatility(daily_returns, f"{prefix}_rolling_volatility.png")
    plot_monthly_returns_heatmap(daily_returns, f"{prefix}_monthly_heatmap.png")
    plot_exposure(weights, f"{prefix}_exposure.png")
    plot_turnover(turnover, f"{prefix}_turnover.png")
    plot_trade_pnl_distribution(trade_log, f"{prefix}_trade_pnl_dist.png")
