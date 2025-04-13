# src/config/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from datetime import date, timedelta, datetime
try:
    from src.config.enums import TimeInterval, OptionType, Moneyness
except ImportError:
    from enum import Enum

    class TimeInterval(Enum):
        MINUTE = "minute"
        HOUR = "hour"
        DAILY = "day"
        WEEKLY = "week"
        MONTHLY = "month"
        QUARTERLY = "quarter"
        YEARLY = "year"

    class OptionType(Enum):
        CALL = "call"
        PUT = "put"

    class Moneyness(Enum):
        ITM = "ITM"
        ATM = "ATM"
        OTM = "OTM"
from pydantic import Field
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_OUTPUT_BASE_DIR = "pipeline_output_final"


def four_years_ago_iso() -> str: return "2021-05-01"
def today_iso() -> str: return "2025-04-01"


class Settings(BaseSettings):
    polygon_api_key: str = Field(default=os.getenv(
        "POLYGON_API_KEY", "YOUR_DEFAULT_KEY_HERE"))

    output_base_dir: str = DEFAULT_OUTPUT_BASE_DIR
    raw_data_subdir: str = "raw_data"
    graph_data_subdir: str = "graph_data"

    benchmark_input_subdir: str = "benchmark_input_data"

    start_date: str = Field(default_factory=four_years_ago_iso)
    end_date: str = Field(default_factory=today_iso)
    resolution_vola_calc: TimeInterval = TimeInterval.DAILY
    resolution_analysis: TimeInterval = TimeInterval.DAILY
    vola_select_n_tickers: int = 10
    vola_rolling_window_days: int = 21
    dte_targets: List[int] = Field(default_factory=lambda: [90, 180, 365])
    option_types: List[OptionType] = Field(
        default_factory=lambda: [OptionType.CALL, OptionType.PUT])
    selector_top_n_candidates: int = 20
    analysis_min_volume: Optional[int] = 0
    option_completeness_threshold: float = 0.75

    benchmark_risk_free_rate: float = 0.04
    benchmark_n_steps_binomial: int = 101
    benchmark_n_runs: int = 100
    benchmark_input_lookback_days: int = 252

    moneyness_levels: List[Moneyness] = Field(
        default_factory=lambda: [Moneyness.ITM, Moneyness.ATM, Moneyness.OTM])
    strikes_around_atm_pct: float = 0.05

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8")

    benchmark_tree_steps_list: List[int] = Field(default_factory=lambda: [
        11, 13, 15, 25, 51, 101, 201, 301, 401, 501, 751, 1001, 2_501
    ])

    benchmark_risk_free_rate: float = 0.04
