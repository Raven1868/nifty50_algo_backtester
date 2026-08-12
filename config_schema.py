"""
Config schema — validates config.yaml at load time via pydantic.

Before this module existed, a typo like `initial_capital: "1000000abc"` or
`rebalance_freq: "Weekly"` (instead of "W") would fail deep inside the
pipeline with a confusing pandas/numpy error, far from the actual mistake.
Now it fails immediately, at startup, with a clear field-level message.
"""
from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class BacktestConfig(BaseModel):
    start_date: str
    end_date: str
    initial_capital: float = Field(gt=0)
    rebalance_freq: str

    @field_validator("rebalance_freq")
    @classmethod
    def _valid_freq(cls, v: str) -> str:
        allowed = {"D", "W", "M"}
        if v not in allowed:
            raise ValueError(f"rebalance_freq must be one of {allowed}, got '{v}'")
        return v

    @field_validator("start_date", "end_date")
    @classmethod
    def _valid_date(cls, v: str) -> str:
        import pandas as pd
        try:
            pd.Timestamp(v)
        except Exception as e:
            raise ValueError(f"'{v}' is not a valid date (expected YYYY-MM-DD): {e}")
        return v

    @model_validator(mode="after")
    def _start_before_end(self):
        import pandas as pd
        if pd.Timestamp(self.start_date) >= pd.Timestamp(self.end_date):
            raise ValueError(f"start_date ({self.start_date}) must be before end_date ({self.end_date})")
        return self


class UniverseConfig(BaseModel):
    mode: str
    custom_tickers: Optional[List[str]] = None

    @field_validator("mode")
    @classmethod
    def _valid_mode(cls, v: str) -> str:
        if v not in {"nifty50", "custom"}:
            raise ValueError(f"universe.mode must be 'nifty50' or 'custom', got '{v}'")
        return v

    @model_validator(mode="after")
    def _custom_needs_tickers(self):
        if self.mode == "custom" and not self.custom_tickers:
            raise ValueError("universe.mode='custom' requires a non-empty custom_tickers list")
        return self


class MeanReversionParams(BaseModel):
    lookback: int = Field(gt=1)
    entry_z: float
    exit_z: float
    allow_short: bool = False
    stop_loss_pct: Optional[float] = None
    max_holding_days: Optional[int] = None

    @model_validator(mode="after")
    def _entry_more_extreme_than_exit(self):
        if self.entry_z >= self.exit_z:
            raise ValueError(
                f"mean_reversion.entry_z ({self.entry_z}) should be more negative than "
                f"exit_z ({self.exit_z}) — otherwise the strategy exits before it ever enters."
            )
        return self


class MomentumParams(BaseModel):
    lookback: int = Field(gt=1)
    top_n: int = Field(gt=0)
    weighting: str = "equal"
    skip_recent_days: int = Field(ge=0, default=5)

    @field_validator("weighting")
    @classmethod
    def _valid_weighting(cls, v: str) -> str:
        if v not in {"equal", "volatility"}:
            raise ValueError(f"momentum.weighting must be 'equal' or 'volatility', got '{v}'")
        return v


class MovingAverageParams(BaseModel):
    short_window: int = Field(gt=1)
    long_window: int = Field(gt=1)
    ma_type: str = "SMA"

    @field_validator("ma_type")
    @classmethod
    def _valid_ma_type(cls, v: str) -> str:
        v_upper = v.upper()
        if v_upper not in {"SMA", "EMA"}:
            raise ValueError(f"moving_average.ma_type must be 'SMA' or 'EMA', got '{v}'")
        return v_upper

    @model_validator(mode="after")
    def _short_less_than_long(self):
        if self.short_window >= self.long_window:
            raise ValueError(
                f"moving_average.short_window ({self.short_window}) must be less than "
                f"long_window ({self.long_window})"
            )
        return self


class StrategyConfig(BaseModel):
    name: str
    mean_reversion: MeanReversionParams
    momentum: MomentumParams
    moving_average: MovingAverageParams

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        allowed = {"mean_reversion", "momentum", "moving_average"}
        if v not in allowed:
            raise ValueError(f"strategy.name must be one of {allowed}, got '{v}'")
        return v


class ExecutionConfig(BaseModel):
    transaction_cost_bps: float = Field(ge=0)
    stt_bps: float = Field(ge=0)
    stamp_duty_bps: float = Field(ge=0)
    gst_pct_on_brokerage: float = Field(ge=0, le=1)
    slippage_bps: float = Field(ge=0)
    volume_aware_slippage: bool = False
    max_participation_rate: float = Field(gt=0, le=1, default=0.10)


class PositionSizingConfig(BaseModel):
    method: str
    fixed_pct: float = Field(gt=0, le=1, default=0.10)
    vol_lookback: int = Field(gt=1, default=20)
    target_vol_annual: float = Field(gt=0, default=0.15)
    max_position_size: float = Field(gt=0, le=1)
    max_portfolio_exposure: float = Field(gt=0, le=5)
    min_position_size: float = Field(ge=0, default=0.0)

    @field_validator("method")
    @classmethod
    def _valid_method(cls, v: str) -> str:
        allowed = {"equal_weight", "fixed_pct", "volatility_target"}
        if v not in allowed:
            raise ValueError(f"position_sizing.method must be one of {allowed}, got '{v}'")
        return v


class RiskConfig(BaseModel):
    stop_loss_pct: Optional[float] = None
    max_drawdown_limit: Optional[float] = None


class BenchmarkConfig(BaseModel):
    ticker: str


class OutputConfig(BaseModel):
    reports_dir: str
    charts_dir: str
    cache_dir: str
    processed_dir: str
    logs_dir: str = "logs"


class ValidationConfig(BaseModel):
    train_window: int = Field(gt=0)
    test_window: int = Field(gt=0)
    step: int = Field(gt=0)
    window_type: str = "rolling"
    objective: str = "sharpe"
    param_grids: Dict[str, Dict[str, List]] = Field(default_factory=dict)

    @field_validator("window_type")
    @classmethod
    def _valid_window_type(cls, v: str) -> str:
        if v not in {"rolling", "expanding"}:
            raise ValueError(f"validation.window_type must be 'rolling' or 'expanding', got '{v}'")
        return v

    @field_validator("objective")
    @classmethod
    def _valid_objective(cls, v: str) -> str:
        if v not in {"sharpe", "sortino", "cagr", "calmar"}:
            raise ValueError(f"validation.objective must be one of sharpe/sortino/cagr/calmar, got '{v}'")
        return v


class AppConfig(BaseModel):
    """Root schema — mirrors config.yaml's top-level keys exactly."""
    backtest: BacktestConfig
    universe: UniverseConfig
    strategy: StrategyConfig
    execution: ExecutionConfig
    position_sizing: PositionSizingConfig
    risk: RiskConfig
    benchmark: BenchmarkConfig
    output: OutputConfig
    validation: ValidationConfig

    model_config = {"extra": "forbid"}  # catch typo'd top-level keys, e.g. "backtets:"


def load_and_validate_config(path: str = "config.yaml") -> AppConfig:
    """Load config.yaml and validate it against AppConfig. Raises a clear
    pydantic.ValidationError (with the exact field and reason) on any
    malformed or missing value, instead of failing deep in the pipeline."""
    import yaml
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return AppConfig(**raw)
