import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import yaml
from pydantic import ValidationError
from config_schema import AppConfig, load_and_validate_config


def _valid_config_dict():
    return {
        "backtest": {"start_date": "2020-01-01", "end_date": "2023-01-01",
                     "initial_capital": 1000000, "rebalance_freq": "W"},
        "universe": {"mode": "nifty50", "custom_tickers": None},
        "strategy": {
            "name": "moving_average",
            "mean_reversion": {"lookback": 20, "entry_z": -1.5, "exit_z": -0.3},
            "momentum": {"lookback": 126, "top_n": 10},
            "moving_average": {"short_window": 20, "long_window": 100},
        },
        "execution": {"transaction_cost_bps": 5, "stt_bps": 10, "stamp_duty_bps": 1.5,
                      "gst_pct_on_brokerage": 0.18, "slippage_bps": 5},
        "position_sizing": {"method": "equal_weight", "max_position_size": 0.15, "max_portfolio_exposure": 1.0},
        "risk": {},
        "benchmark": {"ticker": "^NSEI"},
        "output": {"reports_dir": "reports", "charts_dir": "charts", "cache_dir": "data/raw", "processed_dir": "data/processed"},
        "validation": {"train_window": 504, "test_window": 126, "step": 126},
    }


def test_valid_config_passes():
    cfg = AppConfig(**_valid_config_dict())
    assert cfg.backtest.initial_capital == 1000000


def test_rejects_unknown_top_level_key():
    d = _valid_config_dict()
    d["totally_unknown_section"] = {"x": 1}
    with pytest.raises(ValidationError):
        AppConfig(**d)


def test_rejects_bad_rebalance_freq():
    d = _valid_config_dict()
    d["backtest"]["rebalance_freq"] = "Weekly"  # must be "W"
    with pytest.raises(ValidationError):
        AppConfig(**d)


def test_rejects_start_after_end():
    d = _valid_config_dict()
    d["backtest"]["start_date"] = "2025-01-01"
    d["backtest"]["end_date"] = "2020-01-01"
    with pytest.raises(ValidationError):
        AppConfig(**d)


def test_rejects_negative_capital():
    d = _valid_config_dict()
    d["backtest"]["initial_capital"] = -100
    with pytest.raises(ValidationError):
        AppConfig(**d)


def test_rejects_ma_short_window_not_less_than_long():
    d = _valid_config_dict()
    d["strategy"]["moving_average"] = {"short_window": 100, "long_window": 50}
    with pytest.raises(ValidationError):
        AppConfig(**d)


def test_rejects_mean_reversion_entry_not_more_extreme_than_exit():
    d = _valid_config_dict()
    d["strategy"]["mean_reversion"] = {"lookback": 20, "entry_z": -0.3, "exit_z": -1.5}
    with pytest.raises(ValidationError):
        AppConfig(**d)


def test_rejects_custom_universe_without_tickers():
    d = _valid_config_dict()
    d["universe"] = {"mode": "custom", "custom_tickers": None}
    with pytest.raises(ValidationError):
        AppConfig(**d)


def test_rejects_invalid_position_sizing_method():
    d = _valid_config_dict()
    d["position_sizing"]["method"] = "not_a_real_method"
    with pytest.raises(ValidationError):
        AppConfig(**d)


def test_load_and_validate_config_reads_real_file(tmp_path):
    d = _valid_config_dict()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(d))
    cfg = load_and_validate_config(str(path))
    assert cfg.strategy.name == "moving_average"


def test_load_and_validate_actual_project_config():
    """The real config.yaml shipped with the project must itself be valid."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_and_validate_config(os.path.join(root, "config.yaml"))
    assert cfg.backtest.initial_capital > 0
