import os
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timedelta, date, timezone
from typing import List, Dict, Any, Tuple, Optional
import math

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


try:
    from src.config.settings import Settings
    from src.config.enums import TimeInterval, OptionType
    from src.data_processing.polygon_data_loaders import OptionSelector, OptionDataLoader, EquityDataLoader
    from src.interest_rates.interest_rate_loader import InterestRateLoader
    SETTINGS_IMPORTED = True

except ImportError as e:
    logger.error(f"FATAL: Could not import Settings or Loader classes: {e}")
    SETTINGS_IMPORTED = False

    class Settings:
        pass

    class OptionSelector:
        pass

    class OptionDataLoader:
        pass

    class EquityDataLoader:
        pass

try:
    from src.data_processing import sp500data

except ImportError:

    logger.erro("Missing sp500data module.")
    sp500dat = None
try:

    from src.analysis.volatility import VolatilityAnalysis
except ImportError:
    logger.error("Missing VolatilityAnalysis class.")
    VolatilityAnalysis = None


def calculate_time_to_expiry(current_dt_aware: datetime, expiry_date_str: str) -> float:
    """Calculates time to expiry in years, assuming current_dt is market close."""
    try:
        expiry_date = date.fromisoformat(expiry_date_str)
        expiry_dt_aware = datetime.combine(expiry_date, datetime.min.time(
        ), tzinfo=timezone.utc).replace(hour=21)  # Assumes EOD UTC
        if current_dt_aware.tzinfo is None:
            current_dt_aware = current_dt_aware.replace(
                tzinfo=timezone.utc)  # Ensure ttz aware
        current_dt_aware = current_dt_aware.replace(
            hour=21, minute=0, second=0, microsecond=0)  # Align calc time
        time_delta = expiry_dt_aware - current_dt_aware
        days_to_expiry = time_delta.days
        if days_to_expiry <= 0 and expiry_dt_aware > current_dt_aware:
            return 1.0 / 365.25
        elif days_to_expiry <= 0:
            return 1e-9
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
        if not isinstance(daily_equity_prices_df.index, pd.DatetimeIndex):
            daily_equity_prices_df.index = pd.to_datetime(
                daily_equity_prices_df.index, utc=True)
        elif daily_equity_prices_df.index.tz is None:
            daily_equity_prices_df = daily_equity_prices_df.tz_localize('UTC')
        if calc_dt_aware.tzinfo is None:
            calc_dt_aware = calc_dt_aware.replace(tzinfo=timezone.utc)
        hist_data = daily_equity_prices_df[daily_equity_prices_df.index <= calc_dt_aware].copy(
        )
        if len(hist_data) < 2:
            return np.nan
        # Ensures 'close' is numeric
        hist_data['close'] = pd.to_numeric(hist_data['close'], errors='coerce')
        hist_data.dropna(subset=['close'], inplace=True)
        if len(hist_data) < 2:
            return np.nan

        hist_data['log_return'] = np.log(
            hist_data['close'] / hist_data['close'].shift(1))
        min_p = max(2, window_days // 2)
        if len(hist_data['log_return'].dropna()) < min_p:
            logger.debug(
                f" Not enough data points ({len(hist_data['log_return'].dropna())}) for rolling HV ending {calc_dt_aware.date()}")
            return np.nan
        rolling_std = hist_data['log_return'].rolling(
            window=window_days, min_periods=min_p).std()
        calc_date_only = calc_dt_aware.date()
        matching_timestamps = rolling_std.index[rolling_std.index.date == calc_date_only]
        if not matching_timestamps.empty:
            std_dev = rolling_std.loc[matching_timestamps[-1]]
        elif not rolling_std.empty:
            std_dev = rolling_std.iloc[-1]
            logger.debug(
                f" Using last available rolling std dev for HV calc at {calc_dt_aware.date()}")
        else:
            return np.nan
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
        div_events = equity_events_df.loc[(equity_events_df['underlying_ticker'] == underlying_ticker) & (
            equity_events_df['event_type'] == 'dividend') & equity_events_df['event_date'].notna()].copy()
        if div_events.empty:
            return 0.0
        if div_events['event_date'].dt.tz is None:
            div_events['event_date'] = div_events['event_date'].dt.tz_localize(
                'UTC')
        if calc_dt_aware.tzinfo is None:
            calc_dt_aware = calc_dt_aware.replace(tzinfo=timezone.utc)
        relevant_dividends = div_events.loc[(div_events['event_date'] >= start_lookback) & (
            div_events['event_date'] <= calc_dt_aware)].copy()
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


def calculate_expected_bars(start_dt: datetime, end_dt: datetime, resolution: TimeInterval, hours_per_day: int = 8) -> int:
    """Estimates expected number of bars between two datetimes for US market hours."""
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)
    if resolution == TimeInterval.DAILY:
        return np.busday_count(start_dt.date(), end_dt.date() + timedelta(days=1))
    elif resolution == TimeInterval.HOUR:
        return max(0, np.busday_count(start_dt.date(), end_dt.date() + timedelta(days=1)) * hours_per_day)
    else:
        logging.warning(
            f"Expected bar calculation not implemented for {resolution}.")
        return 0


def main():
    logger.info(
        "--- Starting Data Pipeline (Prep for Benchmark with Discrete Dividends & Dynamic Rates) ---")
    settings = Settings()
    if not SETTINGS_IMPORTED:
        logger.error("FATAL: Settings/Enums failed import.")
        return
    if not settings.polygon_api_key or settings.polygon_api_key == "YOUR_DEFAULT_KEY_HERE":
        logger.error("FATAL: API Key missing.")
        return
    if VolatilityAnalysis is None or sp500data is None:
        logger.error("Missing prerequisites.")
        return

    # in & output directories
    output_base_dir = settings.output_base_dir
    raw_data_dir = os.path.join(output_base_dir, settings.raw_data_subdir)
    graph_data_dir = os.path.join(output_base_dir, settings.graph_data_subdir)
    benchmark_input_dir = os.path.join(
        output_base_dir, settings.benchmark_input_subdir)
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(graph_data_dir, exist_ok=True)
    os.makedirs(benchmark_input_dir, exist_ok=True)
    logger.info(f"Output Base Dir: {os.path.abspath(output_base_dir)}")
    logger.info(f"Raw Data Dir: {os.path.abspath(raw_data_dir)}")
    logger.info(f"Graph Data Dir: {os.path.abspath(graph_data_dir)}")
    logger.info(f"Benchmark Input Dir: {os.path.abspath(benchmark_input_dir)}")

    logger.info(
        f"Settings: Start={settings.start_date}, End={settings.end_date}, Analysis Res={settings.resolution_analysis.value}")
    logger.info(
        f"Vola N={settings.vola_select_n_tickers}, Vola Window={settings.vola_rolling_window_days}d")
    logger.info(f"Option Selector Top N: {settings.selector_top_n_candidates}")
    logger.info(
        f"Option Initial Load min_volume={settings.analysis_min_volume} (MUST be 0 or None)")
    logger.info(
        f"Option Completeness Threshold: {settings.option_completeness_threshold:.0%}")
    if not (settings.analysis_min_volume == 0 or settings.analysis_min_volume is None):
        logger.error("FATAL: analysis_min_volume MUST be 0 or None.")
        return

    equity_data_loader = EquityDataLoader(settings=settings)
    rate_loader = InterestRateLoader(
    )
    if rate_loader.rates_df is None:
        logger.error("FATAL: Failed to load interest rate data.")
        return

    # --- VOLATILITY ANALYSIS STAGE ---
    logger.info(f"\n--- Step 1: Volatility Analysis ---")
    logger.info("Loading S&P 500 tickers...")
    sp500_ticker_list = sp500data.load_sp500_tickers()
    if not sp500_ticker_list:
        logger.error("Failed S&P load.")
        return
    logger.info(
        f"Loading DAILY equity data for {len(sp500_ticker_list)} tickers for vola calc...")
    daily_equity_data_dict = equity_data_loader.load_data(
        tickers=sp500_ticker_list, resolution=settings.resolution_vola_calc)
    all_daily_prices_list = []
    all_equity_events_list_from_daily = []  # Collects events from daily load
    for ticker, data in daily_equity_data_dict.items():
        price_df = data.get('prices')
        events = data.get('events', [])
        if price_df is not None and not price_df.empty:
            price_df_copy = price_df.copy()
            price_df_copy['Ticker'] = ticker
            all_daily_prices_list.append(price_df_copy)
        for event_dict in events:
            event_dict_copy = event_dict.copy()
            event_dict_copy['underlying_ticker'] = ticker
            all_equity_events_list_from_daily.append(event_dict_copy)
    if not all_daily_prices_list:
        logger.error("No valid daily price data for vola analysis/saving.")
        return
    all_daily_prices_df_indexed = pd.concat(all_daily_prices_list)
    daily_prices_save_path = os.path.join(
        raw_data_dir, "all_sp500_daily_prices.csv")
    try:
        all_daily_prices_df_indexed.to_csv(daily_prices_save_path, index=True)
        logger.info(f"Saved daily equity prices.")
    except Exception as save_err:
        logger.error(f"Error saving daily prices: {save_err}")
        return
    vola_prep_df = all_daily_prices_df_indexed.reset_index().rename(
        columns={'timestamp': 'Date', 'close': 'Close'})[['Date', 'Ticker', 'Close']]
    logger.info(f"Running volatility analysis...")
    vola_analyzer = VolatilityAnalysis()
    rolling_vola_df, selected_tickers_dict, vola_summary_table = vola_analyzer.pick_stocks_by_volatility(
        ticker_data=vola_prep_df, n=settings.vola_select_n_tickers, rolling_window=settings.vola_rolling_window_days)
    if not selected_tickers_dict:
        logger.error("Volatility analysis selected no tickers.")
        return
    TARGET_TICKERS_FROM_VOLA = list(selected_tickers_dict.keys())
    logger.info(f"Vola selected tickers: {TARGET_TICKERS_FROM_VOLA}")
    vola_rolling_filename = os.path.join(
        graph_data_dir, "daily_rolling_vola_selected_tickers.csv")
    if not rolling_vola_df.empty:
        rolling_vola_df.to_csv(vola_rolling_filename, index=False)
    vola_table_filename = os.path.join(
        graph_data_dir, "vola_summary_table.csv")
    vola_summary_table.to_csv(vola_table_filename, index=False)
    logger.info(f"Saved volatility results to {graph_data_dir}")
    print("\nVolatility Summary Table:\n", vola_summary_table.to_string())

    # --- OPTION SELECTION (TOP N) & LOADING STAGE ---
    logger.info(
        f"\n--- Step 2: Selecting Top {settings.selector_top_n_candidates} Potential Options ---")
    option_selector = OptionSelector(
        tickers=TARGET_TICKERS_FROM_VOLA, settings=settings)
    selected_options_metadata_df = option_selector.select_options()
    if selected_options_metadata_df.empty:
        logger.warning("No potential options metadata. Exiting.")
        return
    logger.info(
        f"Generated metadata for {len(selected_options_metadata_df)} potential options.")
    raw_metadata_filename = os.path.join(
        raw_data_dir, "raw_selected_options_metadata.csv")
    selected_options_metadata_df.to_csv(raw_metadata_filename, index=False)
    logger.info(f"Saved RAW metadata.")

    option_tickers_to_load = selected_options_metadata_df['selected_option_ticker'].unique(
    ).tolist()
    historical_option_data_dict: Dict[str, pd.DataFrame] = {}
    if not option_tickers_to_load:
        logger.warning("No unique option tickers.")
    else:
        logger.info(
            f"\n--- Step 3: Loading Historical Option Data ({settings.resolution_analysis.value}, min_vol=0) ---")
        option_data_loader = OptionDataLoader(settings=settings)
        historical_option_data_dict = option_data_loader.load_data(
            option_tickers=option_tickers_to_load)

    # --- *** Step 3.5: Load Equity Data with Events ONCE *** ---
    logger.info(
        f"\n--- Step 3.5: Loading Equity Data ({settings.resolution_analysis.value}) and Events ---")

    equity_data_with_events: Dict[str, Dict[str, Any]] = equity_data_loader.load_data(
        tickers=TARGET_TICKERS_FROM_VOLA,
        resolution=settings.resolution_analysis
    )
    # Consolidates events into a single DataFrame for easier lookup later
    all_equity_events_list = []
    for ticker, data in equity_data_with_events.items():
        events = data.get('events', [])
        for event_dict in events:
            event_dict_copy = event_dict.copy()
            event_dict_copy['underlying_ticker'] = ticker
            all_equity_events_list.append(event_dict_copy)
    events_df = pd.DataFrame(all_equity_events_list)  # Creates events_df HERE
    if not events_df.empty:
        events_df['event_date'] = pd.to_datetime(
            # tz aware
            events_df['event_date'], errors='coerce').dt.tz_localize('UTC')
        events_df['cash_amount'] = pd.to_numeric(
            events_df['cash_amount'], errors='coerce')
        events_df.dropna(
            subset=['event_date', 'underlying_ticker'], inplace=True)
    logger.info(
        f"Loaded/Processed events for {len(equity_data_with_events)} tickers.")

    # Load daily prices for HV calc (when previously calculated!)
    try:
        all_daily_prices_df_indexed = pd.read_csv(
            daily_prices_save_path, index_col='timestamp', parse_dates=True)
    except NameError:
        logger.error("Daily prices path var missing.")
        return
    except FileNotFoundError:
        logger.error(f"Daily prices file missing: {daily_prices_save_path}.")
        return
    if all_daily_prices_df_indexed.index.tz is None:
        all_daily_prices_df_indexed = all_daily_prices_df_indexed.tz_localize(
            'UTC')

    # --- POST-FILTERING & BENCHMARK INPUT PREPARATION ---
    logger.info(
        f"\n--- Step 4: Post-Filtering Options & Preparing Benchmark Input ---")
    valid_candidates_by_group: Dict[Tuple, List[Dict[str, Any]]] = {}
    try:
        analysis_start_dt = datetime.fromisoformat(
            settings.start_date).replace(tzinfo=timezone.utc)
        analysis_end_dt = datetime.fromisoformat(
            settings.end_date).replace(tzinfo=timezone.utc)
        analysis_start_date_obj = analysis_start_dt.date()
        analysis_end_date_obj = analysis_end_dt.date()
    except ValueError:
        logger.error("Invalid start/end date.")
        return

    benchmark_input_rows = []
    skipped_options_filter = 0
    final_selected_metadata_rows_dicts = []
    final_historical_option_dfs = []
    selector_used_moneyness = 'target_moneyness' in selected_options_metadata_df.columns

    # Filtering Loop
    for _, meta_row in selected_options_metadata_df.iterrows():
        option_ticker = meta_row['selected_option_ticker']
        df = historical_option_data_dict.get(option_ticker)
        if df is None or df.empty:
            skipped_options_filter += 1
            continue
        first_trade_date = df.index.min().date()
        last_trade_date = df.index.max().date()
        if not ((first_trade_date <= analysis_end_date_obj) and (last_trade_date >= analysis_start_date_obj)):
            skipped_options_filter += 1
            continue
        try:
            option_exp_date = date.fromisoformat(meta_row['expiration_date'])
            option_exp_dt = datetime.combine(
                option_exp_date, datetime.max.time()).replace(tzinfo=timezone.utc)
        except:
            skipped_options_filter += 1
            continue
        first_trade_dt = df.index.min().to_pydatetime()
        last_trade_dt = df.index.max().to_pydatetime()
        relevant_start_dt = max(analysis_start_dt, first_trade_dt)
        relevant_end_dt = min(analysis_end_dt, option_exp_dt, last_trade_dt)
        if relevant_start_dt >= relevant_end_dt:
            skipped_options_filter += 1
            continue
        expected_bars = calculate_expected_bars(
            relevant_start_dt, relevant_end_dt, settings.resolution_analysis)
        actual_bars = len(df.loc[relevant_start_dt:relevant_end_dt])
        completeness_ratio = actual_bars / \
            expected_bars if expected_bars > 0 else (
                1.0 if actual_bars > 0 else 0.0)
        if completeness_ratio < settings.option_completeness_threshold:
            skipped_options_filter += 1
            logger.warning(
                f"Discard {option_ticker}: Completeness {completeness_ratio:.1%}")
            continue
        avg_volume = df.loc[relevant_start_dt:relevant_end_dt]['volume'].mean()
        avg_volume = 0 if pd.isna(avg_volume) else avg_volume
        group_key_base = (meta_row['underlying_ticker'],
                          meta_row['target_dte'], meta_row['option_type'])
        group_key = (*group_key_base, meta_row['target_moneyness']
                     ) if selector_used_moneyness else (*group_key_base, None)
        if group_key not in valid_candidates_by_group:
            valid_candidates_by_group[group_key] = []
        valid_candidates_by_group[group_key].append(
            {'metadata': meta_row, 'dataframe': df, 'avg_volume': avg_volume})
        logger.info(
            f"  PASSED checks {option_ticker}: Overlap OK, Completeness {completeness_ratio:.1%}, AvgVol {avg_volume:.1f}")

    # Final Selection and Benchmark Input Generation
    for group_key, candidates in valid_candidates_by_group.items():
        if not candidates:
            continue
        candidates.sort(key=lambda x: x['avg_volume'], reverse=True)
        best_candidate = candidates[0]
        best_meta = best_candidate['metadata']
        best_df = best_candidate['dataframe']
        log_group_key_str = f"({group_key[0]},{group_key[1]}DTE,{group_key[2]}{','+str(group_key[3]) if group_key[3] else ''})"
        logger.info(
            f"  FINAL choice for {log_group_key_str}: {best_meta['selected_option_ticker']} (AvgVol: {best_candidate['avg_volume']:.1f})")
        final_selected_metadata_rows_dicts.append(best_meta.to_dict())
        df_copy = best_df.copy()
        df_copy['option_ticker'] = best_meta['selected_option_ticker']
        final_historical_option_dfs.append(df_copy)

        calc_timestamp_aware = best_df.index[0].to_pydatetime()
        underlying_ticker = best_meta['underlying_ticker']
        strike_price = best_meta['strike_price']
        expiration_date_str = best_meta['expiration_date']
        option_type_str = best_meta['option_type']
        target_dte = best_meta['target_dte']

        # Get Spot price S (using ANALYSIS resolution prices loaded in Step 3.5)
        underlying_analysis_prices = equity_data_with_events.get(
            underlying_ticker, {}).get('prices')  # Get from dict
        if underlying_analysis_prices is None or underlying_analysis_prices.empty:
            logger.warning(
                f" No {settings.resolution_analysis.value} equity price data for {underlying_ticker}. Skip benchmark row.")
            continue
        equity_data_at_calc_time = underlying_analysis_prices[
            underlying_analysis_prices.index <= calc_timestamp_aware]
        if equity_data_at_calc_time.empty:
            logger.warning(
                f" No equity price for {underlying_ticker} at {calc_timestamp_aware}. Skip benchmark row.")
            continue
        spot_price_initial = equity_data_at_calc_time['close'].iloc[-1]

        # Calculate T, sigma
        daily_equity_prices = all_daily_prices_df_indexed[
            all_daily_prices_df_indexed['Ticker'] == underlying_ticker]
        time_to_expiry = calculate_time_to_expiry(
            calc_timestamp_aware, expiration_date_str)
        hist_vol = calculate_historical_volatility(
            daily_equity_prices, calc_timestamp_aware, settings.vola_rolling_window_days)

        # Get dynamic r
        risk_free_rate = rate_loader.get_rate(
            calc_timestamp_aware, time_to_expiry)
        if risk_free_rate is None:
            risk_free_rate = settings.benchmark_risk_free_rate
            logger.warning(" Using default r")

        # Discrete Dividend Adjustment
        sum_pv_divs = 0.0
        q_value = 0.0
        S_adj = spot_price_initial
        try:
            option_exp_date_obj = date.fromisoformat(expiration_date_str)
            future_dividends = events_df[(events_df['underlying_ticker'] == underlying_ticker) & (events_df['event_type'] == 'dividend') & (
                events_df['event_date'].dt.date > calc_timestamp_aware.date()) & (events_df['event_date'].dt.date <= option_exp_date_obj)]
            if not future_dividends.empty:
                for _, div_row in future_dividends.iterrows():
                    div_amount = div_row['cash_amount']
                    ex_div_dt = div_row['event_date']
                    time_to_div = max(
                        (ex_div_dt - calc_timestamp_aware).days / 365.25, 1e-9)
                    if pd.notna(div_amount) and div_amount > 0 and pd.notna(risk_free_rate):
                        pv_div = div_amount * \
                            math.exp(-risk_free_rate * time_to_div)
                        sum_pv_divs += pv_div
                S_adj = max(0.0, spot_price_initial - sum_pv_divs)
        except Exception as div_err:
            logger.warning(
                f"Error during div adj: {div_err}. Using S_unadjusted, q=0.")
            S_adj = spot_price_initial
            q_value = 0.0

        # Final input validation
        if pd.isna(time_to_expiry) or pd.isna(hist_vol) or pd.isna(S_adj) or S_adj <= 0 or time_to_expiry <= 0 or hist_vol <= 1e-6 or pd.isna(risk_free_rate):
            logger.warning(
                f" Skipping benchmark row for {best_meta['selected_option_ticker']} after rate/dividend adj: invalid inputs.")
            continue

        benchmark_input_rows.append({
            "underlying_ticker": underlying_ticker, "option_ticker": best_meta['selected_option_ticker'],
            "timestamp": calc_timestamp_aware.isoformat(),
            "S": S_adj, "K": strike_price, "T": time_to_expiry, "r": risk_free_rate,
            "sigma": hist_vol, "q": q_value,
            "option_type": option_type_str, "target_dte_group": target_dte,
            "expiration_date": expiration_date_str,  # Use string date
            "S_unadjusted": spot_price_initial, "sum_pv_divs": sum_pv_divs})

    logger.info(f"Prepared {len(benchmark_input_rows)} rows for benchmarking.")
    logger.info(
        f"Total options skipped during filtering: {skipped_options_filter}")

    # --- Save Benchmark Input Data ---
    if benchmark_input_rows:
        benchmark_input_df = pd.DataFrame(benchmark_input_rows)
        benchmark_input_filename = os.path.join(
            benchmark_input_dir, "benchmark_input_data_discrete_div.csv")
        output_cols_bench = ["underlying_ticker", "option_ticker", "timestamp", "S", "K", "T", "r",
                             "sigma", "q", "option_type", "target_dte_group", "expiration_date", "S_unadjusted", "sum_pv_divs"]
        benchmark_input_df = benchmark_input_df[[
            col for col in output_cols_bench if col in benchmark_input_df.columns]]
        benchmark_input_df.to_csv(benchmark_input_filename, index=False)
        logger.info(
            f"Saved benchmark input data to: {benchmark_input_filename}")
    else:
        logger.warning("No valid data rows prepared for benchmarking.")

    # --- Save Final Raw Data ---
    if not final_selected_metadata_rows_dicts:
        final_metadata_df = pd.DataFrame()
    else:
        final_metadata_df = pd.DataFrame(final_selected_metadata_rows_dicts)
    if not final_metadata_df.empty:
        filtered_metadata_filename = os.path.join(
            raw_data_dir, "final_liquid_options_metadata.csv")
        final_metadata_df.to_csv(filtered_metadata_filename, index=False)
        logger.info(f"Saved FINAL metadata ({len(final_metadata_df)} rows).")
    if final_historical_option_dfs:
        consolidated_options_df = pd.concat(final_historical_option_dfs)
        options_hist_filename = os.path.join(
            raw_data_dir, "final_liquid_options_historical_ohlcv.csv")
        consolidated_options_df.to_csv(options_hist_filename, index=True)
        logger.info(
            f"Saved FINAL option OHLCV ({len(consolidated_options_df)} rows).")

    # --- Save FINAL Equity Data (Prices at analysis res + events) ---
    final_underlying_tickers = final_metadata_df['underlying_ticker'].unique(
    ).tolist() if not final_metadata_df.empty else []
    if final_underlying_tickers:
        all_equity_prices_dfs_final = []
        all_equity_events_saved = []
        # Using the equity_data_with_events dictionary loaded in Step 3.5
        for ticker in final_underlying_tickers:
            if ticker in equity_data_with_events:
                # Get data for this ticker
                data = equity_data_with_events[ticker]
                price_df = data.get('prices')
                events = data.get('events', [])
                if price_df is not None and not price_df.empty:
                    price_df_copy = price_df.copy()
                    price_df_copy['underlying_ticker'] = ticker
                    all_equity_prices_dfs_final.append(price_df_copy)

                for event_dict in events:
                    event_dict_copy = event_dict.copy()
                    event_dict_copy['underlying_ticker'] = ticker
                    all_equity_events_saved.append(event_dict_copy)
            else:
                logger.warning(
                    f"No analysis-res price data found for {ticker} during final saving.")

        if all_equity_prices_dfs_final:
            consolidated_equity_prices_df = pd.concat(
                all_equity_prices_dfs_final)
            prices_filename = os.path.join(
                raw_data_dir, f"final_equity_historical_prices_{settings.resolution_analysis.value}.csv")
            consolidated_equity_prices_df.to_csv(prices_filename, index=True)
            logger.info(
                f"Saved FINAL equity prices ({len(consolidated_equity_prices_df)} rows).")

        if all_equity_events_saved:
            consolidated_events_df = pd.DataFrame(all_equity_events_saved)
            events_filename = os.path.join(
                raw_data_dir, "final_equity_events.csv")
            consolidated_events_df.to_csv(events_filename, index=False)
            logger.info(
                f"Saved FINAL equity events ({len(consolidated_events_df)} rows).")

    logger.info("\n--- Data Pipeline Finished ---")
    logger.info(
        f"Find benchmark input data in: {os.path.abspath(benchmark_input_dir)}")
    logger.info(f"Find other raw data in: {os.path.abspath(raw_data_dir)}")
    logger.info(f"Find graph data in: {os.path.abspath(graph_data_dir)}")


if __name__ == "__main__":
    main()
