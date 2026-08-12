import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from data.cleaner import clean_ohlcv


def _flat_ohlcv(n=10):
    dates = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame({
        "Open": np.linspace(100, 110, n), "High": np.linspace(101, 111, n),
        "Low": np.linspace(99, 109, n), "Close": np.linspace(100, 110, n),
        "Adj Close": np.linspace(100, 110, n), "Volume": np.full(n, 1000),
    }, index=dates)


def test_clean_ohlcv_basic_pass_through():
    df = _flat_ohlcv()
    result = clean_ohlcv(df, "TEST.NS")
    assert len(result) == len(df)
    assert "Close" in result.columns


def test_clean_ohlcv_handles_multiindex_columns_defensively():
    """Regression test: a single-ticker fetch (e.g. a benchmark index like
    ^NSEI) can come back with MultiIndex columns (ticker, field) from
    yfinance depending on how it's called (list vs string input). clean_ohlcv
    must flatten these rather than crash with KeyError on 'Close' downstream."""
    flat = _flat_ohlcv()
    multi = flat.copy()
    multi.columns = pd.MultiIndex.from_product([["^NSEI"], flat.columns])

    result = clean_ohlcv(multi, "^NSEI")
    assert "Close" in result.columns
    assert not result.empty
    assert result["Close"].notna().all()


def test_clean_ohlcv_drops_duplicate_dates_keeping_last():
    df = _flat_ohlcv()
    dup = pd.concat([df, df.iloc[[0]]])  # duplicate the first date
    result = clean_ohlcv(dup, "TEST.NS")
    assert not result.index.duplicated().any()


def test_clean_ohlcv_drops_non_positive_prices():
    df = _flat_ohlcv()
    df.iloc[3, df.columns.get_loc("Close")] = -5.0
    result = clean_ohlcv(df, "TEST.NS")
    assert (result["Close"] > 0).all()


def test_clean_ohlcv_empty_input_returns_empty():
    result = clean_ohlcv(pd.DataFrame(), "TEST.NS")
    assert result.empty


def test_to_panel_uses_fallback_when_primary_field_entirely_nan():
    """Regression test: index tickers like ^NSEI frequently return an
    all-NaN 'Adj Close' from yfinance. to_panel must fall back to 'Close'
    for that ticker instead of silently producing an invisible/empty
    benchmark series (this was the root cause of the missing NIFTY 50 line
    in the dashboard's equity-vs-benchmark chart)."""
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from data.loader import DataLoader

    good = _flat_ohlcv()
    bad_adj_close = _flat_ohlcv()
    bad_adj_close["Adj Close"] = float("nan")  # simulate ^NSEI's typical response

    panel = DataLoader.to_panel(
        {"NORMAL.NS": good, "^NSEI": bad_adj_close},
        field="Adj Close", fallback_field="Close",
    )
    assert panel["NORMAL.NS"].notna().all()
    assert panel["^NSEI"].notna().all()  # fell back to Close instead of staying all-NaN
    import pandas.testing as pdt
    pdt.assert_series_equal(panel["^NSEI"], bad_adj_close["Close"], check_names=False)


def test_to_panel_without_fallback_keeps_original_behavior():
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from data.loader import DataLoader

    bad_adj_close = _flat_ohlcv()
    bad_adj_close["Adj Close"] = float("nan")
    panel = DataLoader.to_panel({"^NSEI": bad_adj_close}, field="Adj Close")  # no fallback
    assert panel["^NSEI"].isna().all()  # unchanged behavior when fallback not requested
