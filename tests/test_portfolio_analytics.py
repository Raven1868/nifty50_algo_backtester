import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from analytics.portfolio_analytics import (
    sector_exposure_timeseries, sector_exposure_snapshot, correlation_matrix,
    risk_contribution_snapshot, sector_risk_contribution, concentration_hhi, rolling_beta,
)
from data.sector_mapping import map_tickers_to_sectors, get_sector_map


def _prices(n_days=300, tickers=("RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"), seed=2):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    data = 100 * np.cumprod(1 + rng.normal(0.0005, 0.013, size=(n_days, len(tickers))), axis=0)
    return pd.DataFrame(data, index=dates, columns=list(tickers))


def _weights(prices, seed=1):
    rng = np.random.default_rng(seed)
    w = pd.DataFrame(rng.choice([0.0, 0.25], size=prices.shape, p=[0.4, 0.6]), index=prices.index, columns=prices.columns)
    return w


def test_sector_mapping_covers_known_tickers():
    m = get_sector_map()
    assert m["TCS.NS"] == "Information Technology"
    assert m["HDFCBANK.NS"] == "Financials"


def test_map_tickers_labels_unknown_as_unclassified():
    mapped = map_tickers_to_sectors(["TCS.NS", "FAKE_TICKER.NS"])
    assert mapped["FAKE_TICKER.NS"] == "Unclassified"
    assert mapped["TCS.NS"] == "Information Technology"


def test_sector_exposure_timeseries_sums_correctly():
    prices = _prices()
    weights = _weights(prices)
    sector_map = map_tickers_to_sectors(prices.columns.tolist())
    exposure = sector_exposure_timeseries(weights, sector_map)
    # total exposure across sectors must equal total exposure across tickers, every day
    np.testing.assert_allclose(exposure.sum(axis=1).values, weights.sum(axis=1).values, atol=1e-9)


def test_sector_exposure_snapshot_only_active_positions():
    prices = _prices()
    weights = _weights(prices)
    sector_map = map_tickers_to_sectors(prices.columns.tolist())
    snap = sector_exposure_snapshot(weights.iloc[-1], sector_map)
    assert (snap["weight"] != 0).all()


def test_correlation_matrix_diagonal_is_one():
    prices = _prices()
    corr = correlation_matrix(prices, lookback=100)
    np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-9)
    # symmetric
    np.testing.assert_allclose(corr.values, corr.values.T, atol=1e-9)


def test_risk_contribution_sums_to_100_pct():
    prices = _prices()
    weights = _weights(prices)
    rc = risk_contribution_snapshot(weights.iloc[-1], prices, as_of=weights.index[-1], lookback=100)
    if not rc.empty and rc["pct_risk_contribution"].notna().all():
        assert abs(rc["pct_risk_contribution"].sum() - 100.0) < 1.0  # Euler decomposition sums to ~100%


def test_risk_contribution_empty_when_no_active_weight():
    prices = _prices()
    zero_weights = pd.Series(0.0, index=prices.columns)
    rc = risk_contribution_snapshot(zero_weights, prices, as_of=prices.index[-1])
    assert rc.empty


def test_sector_risk_contribution_groups_correctly():
    prices = _prices()
    weights = _weights(prices)
    sector_map = map_tickers_to_sectors(prices.columns.tolist())
    rc = risk_contribution_snapshot(weights.iloc[-1], prices, as_of=weights.index[-1], lookback=100)
    if not rc.empty:
        sector_rc = sector_risk_contribution(rc, sector_map)
        assert abs(sector_rc["weight"].sum() - rc["weight"].sum()) < 1e-9


def test_concentration_hhi_bounds():
    prices = _prices()
    weights = _weights(prices)
    hhi = concentration_hhi(weights)
    valid = hhi.dropna()
    assert (valid >= 0).all() and (valid <= 1.0001).all()


def test_concentration_fully_concentrated_equals_one():
    dates = pd.bdate_range("2023-01-02", periods=5)
    w = pd.DataFrame({"A": [1.0]*5, "B": [0.0]*5}, index=dates)
    hhi = concentration_hhi(w)
    np.testing.assert_allclose(hhi.values, 1.0, atol=1e-9)


def test_rolling_beta_of_series_with_itself_is_one():
    dates = pd.bdate_range("2023-01-02", periods=300)
    rng = np.random.default_rng(4)
    ret = pd.Series(rng.normal(0.0005, 0.01, 300), index=dates)
    beta = rolling_beta(ret, ret, window=60)
    valid = beta.dropna()
    np.testing.assert_allclose(valid.values, 1.0, atol=1e-6)
