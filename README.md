# NIFTY 50 Algorithmic Trading Backtester

A modular, vectorized backtesting engine for systematic equity strategies on
the Indian (NSE) market, built entirely on free data (Yahoo Finance via
`yfinance`) — ₹0 cost to run.

> Status: Full pipeline built — data ingestion, 3 strategies, vectorized
> engine with realistic Indian costs, performance analytics, 10+ charts,
> walk-forward validation, parameter optimization/sensitivity, market-regime
> analysis, a Streamlit dashboard, and a lookahead-bias test suite.

## 1. Problem Statement
Retail/portfolio backtests routinely overstate performance through
lookahead bias, survivorship bias, ignored transaction costs, and
over-fit parameters. This project is built to avoid those specific traps
and make the trade-offs explicit rather than hidden.

## 2. Key Features
- Three strategies: Mean Reversion, Cross-Sectional Momentum, MA Crossover
- Single shift(1) point in the whole codebase enforcing signal (day t) →
  execution (day t+1) — see `backtesting/engine.py`
- Automated lookahead-bias regression test (`tests/test_lookahead.py`)
- Realistic Indian transaction cost model (brokerage, STT, stamp duty, GST,
  slippage) — `backtesting/costs.py`
- Vectorized engine (pandas/NumPy, no per-day loops)
- Full performance suite: CAGR, Sharpe, Sortino, Calmar, Max Drawdown, VaR
- Walk-forward validation: rolling/expanding train→test folds, frozen
  out-of-sample parameters, combined OOS equity curve
- Grid-search parameter optimization with overfitting guardrails
- 2D parameter sensitivity heatmaps + plateau-vs-isolated-optimum verdict
- Market regime analysis: performance split by bull/bear trend and
  high/low volatility (backward-looking classification only)
- Portfolio analytics: sector exposure over time, trailing correlation
  matrix, Euler risk-contribution decomposition (per position & per
  sector), Herfindahl concentration index, rolling beta vs benchmark
- Trade log (CSV + Excel) with entry/exit, holding period, P&L
- 10+ chart types incl. sensitivity heatmaps and walk-forward equity curves
- Streamlit dashboard (`visualization/dashboard.py`) — thin UI layer,
  zero duplicated business logic
- Local parquet caching — re-runs don't re-hit Yahoo Finance

## 3. Architecture
```
data/          universe, provider (yfinance), cleaner, loader, sector_mapping
strategies/    base interface + mean_reversion, momentum, moving_average
backtesting/   engine (core), costs, positions (sizing)
analytics/     performance, drawdown, trade_analysis, regime, portfolio_analytics
validation/    optimizer (grid search), walk_forward, sensitivity
visualization/ charts (matplotlib), dashboard (Streamlit)
tests/         test_lookahead.py, test_strategies.py, test_metrics.py, test_backtester.py, test_portfolio_analytics.py
main.py        CLI entry point
config.yaml    all parameters — never hard-coded in source
```

## 4. Lookahead Bias Prevention
Every strategy's `generate_signals()` may only use price history up to and
including the current row. The engine alone applies
`target_weight.shift(1)` before multiplying by returns — signal computed
with day-t information is realized against day t+1's return. This is
tested directly: `test_lookahead.py` reruns each strategy with future
prices scrambled and asserts pre-cutoff signals are byte-identical.

## 5. Survivorship Bias — Known Limitation
`data/universe.py` uses today's NIFTY 50 constituent list for the
entire backtest period. Free data does not provide reliable historical,
point-in-time constituent-change history. This means backtests are subject
to mild positive survivorship bias (only currently-successful stocks are
tested). The module is structured so a historical constituent file can be
plugged in later (`get_point_in_time_universe()` is stubbed and raises
`NotImplementedError` rather than silently returning a biased result under
a misleading name).

## 6. Transaction Costs
All rates in `config.yaml` under `execution:` are illustrative — verify
against the current Finance Act / SEBI / CBIC / exchange circulars before
using for real capital. The engine reports gross vs net performance
side-by-side so cost drag is always visible.

## 7. Walk-Forward Validation
`validation/walk_forward.py` splits history into successive TRAIN → TEST
folds. Parameters are grid-searched on TRAIN only, frozen, then applied
unmodified to the following unseen TEST window. All TEST-window results are
stitched into one out-of-sample equity curve — this, not the in-sample
number, is the headline result. Configure via `config.yaml` → `validation:`.

## 8. Parameter Sensitivity
`validation/sensitivity.py` runs a strategy across a 2D grid of two
parameters and reports CAGR/Sharpe/MaxDD at every point, plus a plain-
English verdict on whether performance forms a robust plateau or an
isolated optimum (overfitting red flag).

## 9. Sector Data — Known Limitation
`data/sector_mapping.py` uses a manually-curated, static sector map for the
NIFTY50 list (broad GICS-like buckets, not the official NSE sectoral-index
taxonomy). It will drift as constituents or business mixes change — verify
against nseindia.com's sectoral indices before relying on it. A CSV
(ticker, sector) can be swapped in via `get_sector_map(path=...)` without
touching any analytics code.

## 10. Data Limitations
- Yahoo Finance data via `yfinance` — free, generally reliable for large-cap
  NSE names, but can have gaps/adjustment quirks around corporate actions.
- No dividend-adjusted total-return series beyond what yfinance's "Adj
  Close" provides.
- Rate-limited: the provider batches requests (15 tickers/batch) with
  exponential backoff; a full NIFTY50 pull can take 1–3 minutes.

## 11. Engineering / Production Readiness
Beyond the financial methodology, the project includes standard software
engineering practices for anyone extending it further:

- Config validation (`config_schema.py`) — `config.yaml` is validated
  against a pydantic schema at startup. A typo (bad date, invalid
  `rebalance_freq`, `short_window >= long_window`, an unknown top-level key)
  fails immediately with a clear message, instead of deep inside the
  pipeline with a confusing pandas/numpy error.
- Audit logging — every `main.py` run writes a timestamped log to
  `logs/` (data-cleaning warnings, cost assumptions, walk-forward folds,
  the final run summary), in addition to the human-readable console report.
- Volume-aware slippage (`execution.volume_aware_slippage: true`) —
  slippage scales up for trades that are large relative to a stock's
  trailing average daily traded value, instead of assuming the same flat
  bps regardless of order size. Disabled by default (matches the original
  flat-cost model exactly when off).
- Static type checking — `mypy.ini` is configured for this codebase;
  `python -m mypy . --config-file mypy.ini` currently reports zero errors.
  Wired into CI (see below) so a shape-mismatch bug (like the yfinance
  MultiIndex-columns issue this project hit once) is caught before merge.
- CI (`.github/workflows/ci.yml`) — runs the full test suite on Python
  3.11 and 3.12 on every push/PR, runs mypy, and separately validates that
  the shipped `config.yaml` itself passes schema validation.
- Secrets scaffolding (`secrets_config.py`, `.env.example`) — nothing in
  the current backtester needs credentials, but if this is ever extended
  toward paper/live trading (see §14), API keys must be loaded via
  `secrets_config.get_secret()` from a git-ignored `.env` file — never
  placed in `config.yaml` or committed to git.
- Docker (`Dockerfile`, `.dockerignore`) — pins the Python version and
  exact dependency versions so "works on my machine" issues are caught in a
  controlled image rather than whatever a fresh `pip install` resolves to.
  Not build-tested in every environment — verify `docker build .` works on
  your machine before relying on it.

## 12. Installation (VS Code / local machine)
See the step-by-step VS Code guide provided alongside this project.
Quick version:
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests/ -v      # confirm the lookahead test passes locally
python -m mypy . --config-file mypy.ini   # optional: static type check
python main.py --strategy momentum --universe nifty50
```

Or with Docker:
```bash
docker build -t nifty50-backtester .
docker run -v $(pwd)/reports:/app/reports -v $(pwd)/charts:/app/charts nifty50-backtester
```

## 13. Usage
```bash
python main.py --strategy moving_average --ticker RELIANCE.NS
python main.py --strategy momentum --universe nifty50
python main.py --strategy mean_reversion --universe nifty50 --capital 500000 --start 2020-01-01

# Add validation / analysis layers:
python main.py --strategy moving_average --universe nifty50 --walkforward
python main.py --strategy moving_average --universe nifty50 --sensitivity
python main.py --strategy moving_average --universe nifty50 --regime
python main.py --strategy moving_average --universe nifty50 --portfolio-analytics

# Dashboard:
streamlit run visualization/dashboard.py
```
Outputs land in `reports/` (CSV/Excel trade log + metrics summary) and
`charts/` (PNG charts).

## 14. Limitations
- Survivorship bias in the NIFTY50 universe (see §5)
- No corporate-action-specific adjustment beyond yfinance's own handling
- Single flat risk-free rate assumption (6.5% annualized) for Sharpe/Sortino
- Volatility-target position sizing is an approximation (ignores
  cross-asset correlation when scaling to a target portfolio vol)
- Backtested performance does not guarantee future performance. Nothing
  in this repository should be read as investment advice.

## 15. Future Improvements
- Point-in-time NIFTY50 constituent history (survivorship-bias-free)
- Official NSE sectoral-index mapping (replace the static curated map)
- Random-search optimization as an alternative to full grid search
- Multi-strategy portfolio blending

