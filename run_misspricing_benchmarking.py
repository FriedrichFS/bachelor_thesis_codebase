import os
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timedelta, date, timezone
from typing import List, Dict, Optional, Any, Type
from dotenv import load_dotenv, find_dotenv


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


try:
    from src.config.settings import Settings

    from src.config.enums import TimeInterval, OptionType
    SETTINGS_IMPORTED = True
except ImportError as e:
    logger.error(
        f"FATAL: Could not import Settings or TimeInterval from src.config: {e}")
    SETTINGS_IMPORTED = False

    class Settings:
        pass

    from enum import Enum

    class TimeInterval(Enum):
        DAILY = "day"
        HOUR = "hour"

try:
    from src.models.option_pricing_models import (
        AmericanOptionPricerBase,
        CRRBinomialAmericanPricer,
        LeisenReimerAmericanPricer,
        BjerksundStensland2002Pricer,
        OptionType
    )
    MODELS_IMPORTED = True
    logger.info(
        "Successfully imported pricing models and OptionType from src.models")
except ImportError as e:
    logger.error(
        f"FATAL: Could not import pricing models or OptionType from 'src.models.option_pricing_models_american': {e}")
    MODELS_IMPORTED = False

    class AmericanOptionPricerBase:
        pass

    class CRRBinomialAmericanPricer(AmericanOptionPricerBase):
        pass

    class LeisenReimerAmericanPricer(AmericanOptionPricerBase):
        pass

    class BjerksundStensland2002Pricer(AmericanOptionPricerBase):
        pass


dotenv_path = find_dotenv()
if dotenv_path:
    logger.info(f"Loading .env: {dotenv_path}")
    load_dotenv(dotenv_path)
else:
    logger.warning(".env not found.")


def calculate_time_to_expiry(current_dt_aware: datetime, expiry_date_str: str) -> float:
    """Calculates time to expiry in years, assuming current_dt is market close."""
    try:
        expiry_date = date.fromisoformat(expiry_date_str)
        # Assume expiry happens at end of day. Use consistent time e.g. 21:00 UTC (~4/5 PM ET)
        expiry_dt_aware = datetime.combine(
            expiry_date, datetime.min.time(), tzinfo=timezone.utc).replace(hour=21)
        if current_dt_aware.tzinfo is None:
            current_dt_aware = current_dt_aware.replace(
                tzinfo=timezone.utc)  # Ensure aware
        # Ensure current_dt also represents market close time for accurate calculation
        current_dt_aware = current_dt_aware.replace(
            hour=21, minute=0, second=0, microsecond=0)
        time_delta = expiry_dt_aware - current_dt_aware
        days_to_expiry = time_delta.days
        # If delta is negative or zero but should be positive (e.g. same day expiry after close), return tiny value
        if days_to_expiry <= 0 and expiry_dt_aware > current_dt_aware:
            return 1.0 / 365.25  # Approx 1 day
        elif days_to_expiry <= 0:
            return 1e-9  # Effectively expired or already past
        # Return years, min small positive
        return max(days_to_expiry / 365.25, 1e-9)
    except (ValueError, TypeError) as e:
        logger.warning(
            f"Could not calc TTE for {expiry_date_str} from {current_dt_aware}: {e}")
        return np.nan


def calculate_historical_volatility(daily_equity_prices_df: pd.DataFrame, calc_dt_aware: datetime, window_days: int) -> float:
    """Calculates annualized historical volatility using DAILY data up to calc_dt_aware."""
    try:
        if daily_equity_prices_df is None or daily_equity_prices_df.empty:
            return np.nan
        # Ensure index is datetime and UTC aware
        if not isinstance(daily_equity_prices_df.index, pd.DatetimeIndex):
            daily_equity_prices_df.index = pd.to_datetime(
                daily_equity_prices_df.index, utc=True)
        elif daily_equity_prices_df.index.tz is None:
            daily_equity_prices_df = daily_equity_prices_df.tz_localize('UTC')

        hist_data = daily_equity_prices_df[daily_equity_prices_df.index <= calc_dt_aware].copy(
        )
        if len(hist_data) < 2:
            return np.nan  # Need 2 points for return

        hist_data['log_return'] = np.log(
            hist_data['close'] / hist_data['close'].shift(1))
        min_p = max(2, window_days // 2)  # Min periods for rolling std
        if len(hist_data['log_return'].dropna()) < min_p:
            logger.debug(
                f" Not enough data points for rolling HV ending {calc_dt_aware.date()}")
            return np.nan

        rolling_std = hist_data['log_return'].rolling(
            window=window_days, min_periods=min_p).std()

        # Get the standard deviation for the window ending AT calc_dt_aware
        calc_date_only = calc_dt_aware.date()  # Use date part
        matching_timestamps = rolling_std.index[rolling_std.index.date == calc_date_only]

        if not matching_timestamps.empty:
            # Use latest ts on that date
            std_dev = rolling_std.loc[matching_timestamps[-1]]
        elif not rolling_std.empty:
            std_dev = rolling_std.iloc[-1]  # Fallback to last calculated value
            logger.debug(
                f" Using last available rolling std dev for HV calc at {calc_dt_aware.date()}")
        else:
            return np.nan  # Edge Case -> no data available

        if pd.isna(std_dev):
            return np.nan
        hv = std_dev * np.sqrt(252)
        return hv if pd.notna(hv) else np.nan
    except Exception as e:
        logger.warning(
            f"Could not calculate HV ending {calc_dt_aware.date()}: {e}", exc_info=True)
        return np.nan


def calculate_continuous_dividend_yield(equity_events_df: pd.DataFrame, underlying_ticker: str, spot_price: float, calc_dt_aware: datetime, lookback_days: int) -> float:
    """Calculates simple annualized continuous dividend yield looking back."""
    if spot_price <= 0:
        return 0.0
    try:
        start_lookback = calc_dt_aware - timedelta(days=lookback_days)
        if 'event_date' not in equity_events_df.columns:
            return 0.0
        if not pd.api.types.is_datetime64_any_dtype(equity_events_df['event_date']):
            equity_events_df['event_date'] = pd.to_datetime(
                equity_events_df['event_date'], errors='coerce')

        div_events = equity_events_df.loc[
            (equity_events_df['underlying_ticker'] == underlying_ticker) &
            (equity_events_df['event_type'] == 'dividend') &
            equity_events_df['event_date'].notna()
        ].copy()
        if div_events.empty:
            return 0.0
        if div_events['event_date'].dt.tz is None:
            div_events['event_date'] = div_events['event_date'].dt.tz_localize(
                'UTC')
        if calc_dt_aware.tzinfo is None:
            calc_dt_aware = calc_dt_aware.replace(tzinfo=timezone.utc)

        relevant_dividends = div_events.loc[
            (div_events['event_date'] >= start_lookback) &
            (div_events['event_date'] <= calc_dt_aware)
        ].copy()
        if relevant_dividends.empty:
            return 0.0
        relevant_dividends.loc[:, 'cash_amount'] = pd.to_numeric(
            relevant_dividends['cash_amount'], errors='coerce')
        total_dividends = relevant_dividends['cash_amount'].sum()
        if pd.isna(total_dividends) or total_dividends <= 0:
            return 0.0
        annualized_yield = (total_dividends / spot_price) * \
            (365.25 / lookback_days)
        return max(0.0, annualized_yield)
    except Exception as e:
        logger.warning(
            f"Could not calculate div yield for {underlying_ticker}: {e}", exc_info=True)
        return 0.0


def run_pricing_benchmark():
    logger.info(
        "--- Starting Option Model Pricing Error & Runtime Benchmark ---")

    if not SETTINGS_IMPORTED:
        logger.error("FATAL: Settings/Enums not imported. Exiting.")
        return
    if not MODELS_IMPORTED:
        logger.error("FATAL: Pricing models not imported. Exiting.")
        return

    settings = Settings()
    if not hasattr(settings, 'polygon_api_key') or not settings.polygon_api_key or settings.polygon_api_key == "YOUR_DEFAULT_KEY_HERE":
        logger.error("FATAL: API Key missing or invalid in settings.")
        return

    output_base_dir = settings.output_base_dir
    raw_data_dir = os.path.join(output_base_dir, settings.raw_data_subdir)
    graph_data_dir = os.path.join(output_base_dir, settings.graph_data_subdir)
    benchmark_input_dir = os.path.join(
        output_base_dir, settings.benchmark_input_subdir)
    os.makedirs(graph_data_dir, exist_ok=True)
    logger.info(
        f"Reading Benchmark input data from: {os.path.abspath(benchmark_input_dir)}")
    logger.info(
        f"Reading Option market prices from: {os.path.abspath(raw_data_dir)}")
    logger.info(
        f"Saving Benchmark results to: {os.path.abspath(graph_data_dir)}")

    input_filename = os.path.join(
        benchmark_input_dir, "benchmark_input_data_discrete_div.csv")
    try:
        benchmark_input_df = pd.read_csv(input_filename)
        benchmark_input_df['timestamp_dt'] = pd.to_datetime(
            benchmark_input_df['timestamp'], errors='coerce', utc=True)
        benchmark_input_df.dropna(subset=['timestamp_dt'], inplace=True)
    except FileNotFoundError:
        logger.error(f"FATAL: Benchmark input file missing: {input_filename}")
        return
    except Exception as e:
        logger.error(f"FATAL: Error loading benchmark input: {e}")
        return

    options_ohlcv_filename = os.path.join(
        raw_data_dir, "final_liquid_options_historical_ohlcv.csv")
    try:
        options_ohlcv_df = pd.read_csv(options_ohlcv_filename, parse_dates=[
                                       'timestamp'], index_col='timestamp')
        if options_ohlcv_df.index.tz is None:
            options_ohlcv_df = options_ohlcv_df.tz_localize('UTC')
    except FileNotFoundError:
        logger.error(
            f"FATAL: Option OHLCV file missing: {options_ohlcv_filename}")
        return
    except Exception as e:
        logger.error(f"FATAL: Error loading option OHLCV: {e}")
        return

    daily_prices_path = os.path.join(
        raw_data_dir, "all_sp500_daily_prices.csv")
    try:
        daily_equity_prices_df = pd.read_csv(daily_prices_path, parse_dates=[
                                             'timestamp'], index_col='timestamp')
        if daily_equity_prices_df.index.tz is None:
            daily_equity_prices_df = daily_equity_prices_df.tz_localize('UTC')
    except FileNotFoundError:
        logger.error(
            f"FATAL: Daily equity prices file missing: {daily_prices_path}. Cannot calculate HV.")
        return
    except Exception as e:
        logger.error(f"FATAL: Error loading daily equity prices: {e}")
        return

    events_filename = os.path.join(raw_data_dir, "final_equity_events.csv")
    try:
        events_df = pd.read_csv(events_filename)

        events_df['event_date'] = pd.to_datetime(
            # Assumes UTC !
            events_df['event_date'], errors='coerce').dt.tz_localize('UTC')
        # ! Drop rows where date conversion failed
        events_df.dropna(subset=['event_date'], inplace=True)
    except FileNotFoundError:
        logger.error(
            f"FATAL: Events file missing: {events_filename}. Dividend yield will be 0.")
        events_df = pd.DataFrame()  # Use empty df
    except Exception as e:
        logger.error(f"FATAL: Error loading events data: {e}")
        return

    logger.info(f"Loaded {len(benchmark_input_df)} benchmark input rows.")
    logger.info(f"Loaded {len(options_ohlcv_df)} option OHLCV rows.")
    logger.info(
        f"Runs per model: {settings.benchmark_n_runs}, Binomial steps: {settings.benchmark_n_steps_binomial}")

    # str model encoding for later analsis
    models_to_run: Dict[str, Type[AmericanOptionPricerBase]] = {
        "CRR": CRRBinomialAmericanPricer,
        "LeisenReimer": LeisenReimerAmericanPricer,
        "BS2002": BjerksundStensland2002Pricer
    }

    benchmark_results = []
    processed_rows = 0

    logger.info("Starting model calculations...")
    for index, input_row in benchmark_input_df.iterrows():
        if (index + 1) % 100 == 0:
            logger.info(
                f" Processing input row {index+1}/{len(benchmark_input_df)}...")

        try:
            option_ticker = input_row['option_ticker']
            calc_timestamp_aware = input_row['timestamp_dt'].to_pydatetime()
            spot_price = float(input_row['S'])
            strike_price = float(input_row['K'])
            time_to_expiry = float(input_row['T'])
            risk_free_rate = float(input_row['r'])
            hist_vol = float(input_row['sigma'])
            div_yield = float(input_row['q'])
            option_type_str = input_row['option_type']
            option_type = OptionType(option_type_str)
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logger.warning(f" Skipping row {index+1} invalid input data: {e}")
            continue

        if pd.isna(time_to_expiry) or pd.isna(hist_vol) or pd.isna(div_yield) or pd.isna(spot_price) or time_to_expiry <= 0 or hist_vol <= 1e-6:  # Check sigma > small number
            logger.warning(
                f" Skipping row {index+1} NaN/Zero/Negative inputs (T={time_to_expiry}, sigma={hist_vol}).")
            continue

        try:
            market_data_slice = options_ohlcv_df.loc[calc_timestamp_aware]
            if isinstance(market_data_slice, pd.DataFrame):
                market_data_row = market_data_slice[market_data_slice['option_ticker']
                                                    == option_ticker]
                if market_data_row.empty:
                    raise KeyError
                market_price = market_data_row['close'].iloc[0]
            else:
                if 'option_ticker' in market_data_slice and market_data_slice['option_ticker'] != option_ticker:
                    raise KeyError
                market_price = market_data_slice['close']
            if pd.isna(market_price):
                logger.debug(
                    f" Skip {option_ticker} at {calc_timestamp_aware}: Market price NaN.")
                continue
        except Exception as e:
            logger.warning(
                f" Error finding market price {option_ticker} at {calc_timestamp_aware}: {e}")
            continue

        for model_name, ModelClass in models_to_run.items():
            model_params = {'S': spot_price, 'K': strike_price, 'T': time_to_expiry,
                            'r': risk_free_rate, 'sigma': hist_vol, 'q': div_yield, 'option_type': option_type}
            if model_name in ["CRR", "LeisenReimer"]:
                model_params['N'] = settings.benchmark_n_steps_binomial
            try:
                model_instance = ModelClass(**model_params)
                run_times = []
                first_result_dict = None
                for run_num in range(settings.benchmark_n_runs):
                    result_dict = model_instance.calculate_all()
                    calc_time = result_dict.get('calc_time_sec', np.nan)
                    run_times.append(calc_time)
                    if run_num == 0:
                        first_result_dict = result_dict
                avg_calc_time = np.nanmean(run_times) if run_times else np.nan
                model_price = first_result_dict.get(
                    'price', np.nan) if first_result_dict else np.nan
                pricing_error = model_price - \
                    market_price if pd.notna(model_price) else np.nan

                benchmark_results.append({
                    "underlying_ticker": input_row['underlying_ticker'], "option_ticker": option_ticker,
                    "target_dte_group": input_row['target_dte_group'], "option_type": option_type_str,
                    "timestamp": input_row['timestamp'], "model": model_name, "binomial_steps": model_params.get('N'),
                    "avg_calc_time_sec": avg_calc_time,
                    "calculated_price": model_price, "market_price": market_price, "pricing_error": pricing_error,
                    "delta": first_result_dict.get('delta', np.nan), "gamma": first_result_dict.get('gamma', np.nan),
                    "vega": first_result_dict.get('vega', np.nan), "theta": first_result_dict.get('theta', np.nan), "rho": first_result_dict.get('rho', np.nan),
                    "input_S": spot_price, "input_K": strike_price, "input_T": time_to_expiry,
                    "input_r": risk_free_rate, "input_sigma": hist_vol, "input_q": div_yield,
                })
                logger.debug(
                    f"    {model_name}: Avg time = {avg_calc_time * 1000:.4f} ms, Price={model_price:.4f}, Error={pricing_error:.4f}")
            except ValueError as val_err:
                logger.error(
                    f"   Initialization ERROR for {model_name} on row {index+1}: {val_err}")
                logger.error(
                    f"   Passed option_type type: {type(option_type)}")
            except Exception as model_err:
                logger.error(
                    f"   RUNTIME ERROR running {model_name} for row {index+1}: {model_err}", exc_info=False)

        processed_rows += 1

    logger.info(
        f"\nPricing benchmark finished. Processed {processed_rows} input rows.")

    if benchmark_results:
        results_df = pd.DataFrame(benchmark_results)
        output_cols = ["underlying_ticker", "option_ticker", "target_dte_group", "option_type", "timestamp", "model", "binomial_steps", "avg_calc_time_sec",
                       "calculated_price", "market_price", "pricing_error", "delta", "gamma", "vega", "theta", "rho",
                       "input_S", "input_K", "input_T", "input_r", "input_sigma", "input_q"]
        results_df = results_df[[
            col for col in output_cols if col in results_df.columns]]
        output_filename = os.path.join(
            graph_data_dir, "model_pricing_errors_runtime.csv")
        results_df.to_csv(output_filename, index=False)
        logger.info(
            f"Saved pricing benchmark results ({len(results_df)} rows) to: {output_filename}")
    else:
        logger.warning("No pricing benchmark results generated.")


if __name__ == "__main__":
    run_pricing_benchmark()
