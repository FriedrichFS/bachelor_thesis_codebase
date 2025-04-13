# src/data_processing/polygon_data_loaders.py

import pandas as pd
from polygon import RESTClient
import requests
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Tuple, Any
import time
import logging
import heapq
from src.config.settings import Settings
from src.config.enums import TimeInterval, OptionType, Moneyness


def get_polygon_timespan(interval: TimeInterval) -> Tuple[int, str]:
    if interval == TimeInterval.MINUTE:
        return 1, "minute"
    if interval == TimeInterval.HOUR:
        return 1, "hour"
    if interval == TimeInterval.DAILY:
        return 1, "day"
    if interval == TimeInterval.WEEKLY:
        return 1, "week"
    if interval == TimeInterval.MONTHLY:
        return 1, "month"
    if interval == TimeInterval.QUARTERLY:
        return 3, "month"
    if interval == TimeInterval.YEARLY:
        return 1, "year"
    logging.warning(f"Unsupported interval {interval}, defaulting to daily.")
    return 1, "day"


class OptionSelector:
    """
    Selects the TOP N potential option contracts per Underlying/TargetDTE/Type combination.
    Selection Criteria:
    1. Fetches candidates within expiration windows.
    2. Filters by Type.
    3. Ranks candidates by proximity: closest DTE, then closest Strike.
    4. Selects the top 'selector_top_n_candidates' for each group.
    Returns a DataFrame containing metadata of ALL selected potential candidates.
    Liquidity/completeness filtering happens LATER in the pipeline.
    """

    def __init__(self, tickers: List[str], settings: Settings):
        self.tickers = tickers
        self.settings = settings
        self.client = RESTClient(settings.polygon_api_key)
        self.top_n = settings.selector_top_n_candidates
        logging.info(
            f"OptionSelector initialized (Selecting Top {self.top_n} Candidates per Group)")

    def _get_equity_price_and_date_at_start(self, ticker: str, target_date_str: str) -> Optional[Tuple[float, str]]:
        """Fetches closing price and actual date on or just before target_date_str."""
        try:
            target_dt = date.fromisoformat(target_date_str)
            start_lookup = (target_dt - timedelta(days=7)).strftime('%Y-%m-%d')
            end_lookup = target_dt.strftime('%Y-%m-%d')
            aggs_list = self.client.get_aggs(
                ticker=ticker, multiplier=1, timespan="day", from_=start_lookup, to=end_lookup, sort="desc", limit=1, raw=False)
            if isinstance(aggs_list, list) and aggs_list:
                last_agg = aggs_list[0]
                if hasattr(last_agg, 'timestamp') and hasattr(last_agg, 'close'):
                    result_timestamp_ms = last_agg.timestamp
                    result_date = datetime.fromtimestamp(
                        result_timestamp_ms / 1000).date()
                    if result_date <= target_dt:
                        return last_agg.close, result_date.isoformat()
                    else:  # Fallback
                        aggs_fallback = list(self.client.list_aggs(
                            ticker=ticker, multiplier=1, timespan="day", from_=start_lookup, to=end_lookup, limit=5))
                        if aggs_fallback:
                            last_fallback_agg = aggs_fallback[-1]
                            if hasattr(last_fallback_agg, 'timestamp') and hasattr(last_fallback_agg, 'close'):
                                fallback_date = datetime.fromtimestamp(
                                    last_fallback_agg.timestamp / 1000).date()
                                if fallback_date <= target_dt:
                                    return last_fallback_agg.close, fallback_date.isoformat()
            return None
        except Exception as e:
            logging.error(f"Err fetch price {ticker}: {e}", exc_info=True)
            return None

    def _calculate_dte(self, start_date_str: str, expiration_date_str: str) -> int:
        """Calculates Days To Expiration (DTE) from start_date."""
        try:
            start = date.fromisoformat(start_date_str)
            expiration = date.fromisoformat(expiration_date_str)
            if expiration <= start:
                return -1
            return (expiration - start).days
        except ValueError:
            logging.error(
                f"Invalid date DTE: {start_date_str}, {expiration_date_str}")
            return -1

    def select_options(self) -> pd.DataFrame:
        """Selects Top N options per group based on DTE/Type/ClosestStrike and returns metadata."""
        all_selected_option_rows = []  # Store metadata dicts for all top N picks
        start_date_config = self.settings.start_date

        for ticker in self.tickers:
            logging.info(
                f"--- Selecting Top {self.top_n} options for {ticker} ---")
            price_date_tuple = self._get_equity_price_and_date_at_start(
                ticker, start_date_config)
            if price_date_tuple is None:
                continue
            spot_price, actual_selection_date_str = price_date_tuple
            try:
                actual_selection_date_obj = date.fromisoformat(
                    actual_selection_date_str)
            except ValueError:
                logging.error(
                    f"Invalid date: {actual_selection_date_str}. Skip {ticker}.")
                continue
            logging.info(
                f"Using ref spot price {spot_price:.2f} on {actual_selection_date_str} for {ticker}")

            all_candidates_for_ticker = []
            fetched_expirations = set()
            for target_dte in self.settings.dte_targets:
                target_exp_date = actual_selection_date_obj + \
                    timedelta(days=target_dte)
                exp_window_start = (
                    target_exp_date - timedelta(days=15)).isoformat()
                exp_window_end = (target_exp_date +
                                  timedelta(days=15)).isoformat()
                window_key = (exp_window_start, exp_window_end)
                if window_key in fetched_expirations:
                    continue
                logging.info(
                    f" Fetching contracts expiring {exp_window_start} to {exp_window_end}...")
                try:
                    contracts_iterator = self.client.list_options_contracts(
                        underlying_ticker=ticker, expiration_date_gte=exp_window_start, expiration_date_lte=exp_window_end, expired=True, limit=1000)
                    window_candidates = []
                    for contract in contracts_iterator:
                        if (hasattr(contract, 'ticker') and contract.ticker and hasattr(contract, 'strike_price') and isinstance(contract.strike_price, (int, float)) and hasattr(contract, 'expiration_date') and contract.expiration_date and hasattr(contract, 'contract_type') and contract.contract_type):
                            try:
                                date.fromisoformat(
                                    contract.expiration_date)
                                window_candidates.append(contract)
                            except ValueError:
                                continue
                    logging.info(
                        f"  Found {len(window_candidates)} valid candidates in window.")
                    all_candidates_for_ticker.extend(window_candidates)
                    fetched_expirations.add(window_key)
                except Exception as e:
                    logging.error(
                        f"Error fetching window {window_key}: {e}", exc_info=True)

            if not all_candidates_for_ticker:
                logging.warning(f"No candidates found for {ticker}.")
                continue
            unique_candidate_tickers = {
                c.ticker: c for c in all_candidates_for_ticker}
            candidate_contracts = list(
                unique_candidate_tickers.values())
            logging.info(
                f"Total unique candidates for {ticker}: {len(candidate_contracts)}")
            if not candidate_contracts:
                continue

            for target_dte in self.settings.dte_targets:
                for option_type in self.settings.option_types:
                    # Store (dte_diff, strike_diff_abs, contract_object) tuples
                    group_candidates = []
                    type_candidates = [
                        c for c in candidate_contracts if c.contract_type == option_type.value]
                    if not type_candidates:
                        continue

                    for contract in type_candidates:
                        actual_dte = self._calculate_dte(
                            actual_selection_date_str, contract.expiration_date)
                        if actual_dte < 1:
                            continue
                        dte_diff = abs(actual_dte - target_dte)
                        strike_diff_abs = abs(
                            contract.strike_price - spot_price)
                        group_candidates.append(
                            (dte_diff, strike_diff_abs, contract))

                    # Sort the candidates: first by DTE diff, then by strike diff
                    group_candidates.sort(key=lambda x: (x[0], x[1]))

                    # Select the top N candidates
                    # Take the first N
                    top_n_selected = group_candidates[:self.top_n]

                    logging.info(
                        f"  Group ({ticker}, {target_dte}DTE, {option_type.name}): Found {len(group_candidates)} potential, selected Top {len(top_n_selected)}.")

                    for dte_diff_val, strike_diff_val, selected_contract in top_n_selected:
                        all_selected_option_rows.append({
                            'underlying_ticker': ticker, 'target_dte': target_dte, 'option_type': option_type.value,
                            'selection_date': actual_selection_date_str, 'spot_price_at_selection': spot_price,
                            'selected_option_ticker': selected_contract.ticker, 'strike_price': selected_contract.strike_price,
                            'expiration_date': selected_contract.expiration_date,
                            'dte_at_selection': self._calculate_dte(actual_selection_date_str, selected_contract.expiration_date),
                            'rank_dte_diff': dte_diff_val,  # Store ranking criteria
                            'rank_strike_diff_abs': strike_diff_val,  # Store ranking criteria
                            'primary_exchange': getattr(selected_contract, 'primary_exchange', None),
                            'cfi': getattr(selected_contract, 'cfi', None)
                        })
                        logging.debug(
                            f"    -> Added Candidate: {selected_contract.ticker} (Rank DTE Diff: {dte_diff_val}, Rank Strike Diff: {strike_diff_val:.2f})")

        logging.info(
            f"Option selection complete. Total candidate metadata rows: {len(all_selected_option_rows)}")
        if not all_selected_option_rows:
            return pd.DataFrame()
        return pd.DataFrame(all_selected_option_rows)


class OptionDataLoader:
    """Loads historical OHLCV data SEQUENTIALLY"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = RESTClient(settings.polygon_api_key)
        logging.info("OptionDataLoader initialized.")
        logging.warning("<<< OHLCV ONLY >>>")

    def load_data(self, option_tickers: List[str]) -> Dict[str, pd.DataFrame]:
        historical_data = {}
        if not option_tickers:
            logging.warning("No option tickers provided.")
            return {}
        multiplier, timespan = get_polygon_timespan(
            self.settings.resolution_analysis)
        start_date_cfg = self.settings.start_date
        end_date_cfg = self.settings.end_date
        min_volume = self.settings.analysis_min_volume
        logging.info(
            f"Loading {timespan} OHLCV for {len(option_tickers)} options [{start_date_cfg} to {end_date_cfg}]...")
        for i, option_ticker in enumerate(option_tickers):
            logging.info(
                f"Loading {i+1}/{len(option_tickers)}: {option_ticker}")
            start_time = time.time()
            try:
                aggs_iterator = self.client.list_aggs(
                    ticker=option_ticker, multiplier=multiplier, timespan=timespan, from_=start_date_cfg, to=end_date_cfg, limit=50000, raw=False)
                bars = list(aggs_iterator)
                load_time = time.time() - start_time
                logging.info(
                    f"  Fetched {len(bars)} raw bars in {load_time:.2f} sec.")
                if not bars:
                    logging.warning(f"  No agg data.")
                    historical_data[option_ticker] = pd.DataFrame()
                    continue
                valid_bars_data = []
                for bar in bars:
                    if hasattr(bar, 'timestamp') and hasattr(bar, 'open') and hasattr(bar, 'high') and hasattr(bar, 'low') and hasattr(bar, 'close') and hasattr(bar, 'volume'):
                        bar_dict = bar.__dict__
                        valid_bars_data.append({k: bar_dict.get(k) for k in [
                                               'timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions']})
                    else:
                        logging.warning(f"  Skip malformed bar: {vars(bar)}")
                if not valid_bars_data:
                    logging.warning(f"  No valid bars.")
                    historical_data[option_ticker] = pd.DataFrame()
                    continue
                df = pd.DataFrame(valid_bars_data)
                df['timestamp'] = pd.to_datetime(
                    df['timestamp'], unit='ms', utc=True)
                df = df.set_index('timestamp').sort_index()
                initial_rows = len(df)
                rows_removed = 0
                # NOTE: analysis_min_volume should be 0 here, filtering happens later based on analysis
                if isinstance(min_volume, int) and min_volume > 0 and 'volume' in df.columns:
                    df = df[df['volume'] >= min_volume]
                    rows_removed = initial_rows - len(df)
                    if rows_removed > 0:
                        logging.info(
                            f"  Vol Filter (>{min_volume-1}): Removed {rows_removed}/{initial_rows}")
                elif min_volume == 0 or min_volume is None:
                    logging.info(f"  Vol Filter skipped (initial load).")
                else:
                    logging.warning(
                        f"  Vol Filter invalid setting: {min_volume}")
                ohlcv_cols = ['open', 'high', 'low', 'close',
                              'volume', 'vwap', 'transactions']
                existing_cols = [
                    col for col in ohlcv_cols if col in df.columns]
                df = df[existing_cols]
                if not df.empty:
                    actual_start = df.index.min().strftime('%Y-%m-%d')
                    actual_end = df.index.max().strftime('%Y-%m-%d')
                    logging.info(
                        f"  Processed {len(df)} points ({initial_rows - rows_removed} kept) from {actual_start} to {actual_end}.")
                    historical_data[option_ticker] = df
                else:
                    logging.warning(
                        f"  No data remained. Initial: {initial_rows}")
                    historical_data[option_ticker] = pd.DataFrame()
            except Exception as exc:
                logging.error(
                    f"Error processing {option_ticker}: {exc}", exc_info=True)
                historical_data[option_ticker] = pd.DataFrame()
        logging.info("Option historical OHLCV data loading complete.")
        return historical_data


class EquityDataLoader:
    """Loads historical OHLCV, and consolidated events (dividends, splits, ticker changes)"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = RESTClient(
            settings.polygon_api_key)
        self.session = requests.Session()
        logging.info("EquityDataLoader initialized.")

    def _fetch_ticker_events(self, ticker: str, start_date_obj: date, end_date_obj: date) -> List[Dict[str, Any]]:
        events = []
        url = f"https://api.polygon.io/vX/reference/tickers/{ticker.upper()}/events"
        params = {
            "apiKey": self.settings.polygon_api_key, "types": "ticker_change"}
        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "OK" and "results" in data and "events" in data["results"]:
                raw_events = data["results"]["events"]
                kept_count = 0
                for event in raw_events:
                    if event.get("type") == "ticker_change" and "date" in event and "ticker_change" in event:
                        try:
                            event_date_obj = date.fromisoformat(
                                event.get("date"))
                            if start_date_obj <= event_date_obj <= end_date_obj:
                                events.append({"event_type": "ticker_change", "event_date": event.get(
                                    "date"), "new_ticker": event.get("ticker_change", {}).get("ticker")})
                                kept_count += 1
                        except (ValueError, TypeError):
                            continue
                logging.info(
                    f" Found {len(raw_events)} raw ticker change, kept {kept_count}.")
        except Exception as e:
            logging.error(f"Failed ticker events {ticker}: {e}", exc_info=True)
        return events

    def load_data(self, tickers: List[str], resolution: Optional[TimeInterval] = None) -> Dict[str, Dict[str, object]]:
        equity_data = {}
        res_to_use = resolution if resolution is not None else self.settings.resolution_analysis
        multiplier, timespan = get_polygon_timespan(res_to_use)
        start_date_str = self.settings.start_date
        end_date_str = self.settings.end_date
        try:
            start_date_obj = date.fromisoformat(
                start_date_str)
            end_date_obj = date.fromisoformat(end_date_str)
        except ValueError:
            logging.error(f"Invalid start/end date.")
            return {}
        if not tickers:
            logging.warning("No tickers provided.")
            return {}
        for i, ticker in enumerate(tickers):
            logging.info(
                f"--- Loading equity data ({timespan}) {i+1}/{len(tickers)}: {ticker} ---")
            ticker_data = {'prices': pd.DataFrame(), 'events': []}
            processed_successfully = True
            try:  # Prices
                aggs_iterator = self.client.list_aggs(ticker=ticker, multiplier=multiplier, timespan=timespan,
                                                      from_=start_date_str, to=end_date_str, adjusted=True, limit=50000, raw=False)
                bars = list(aggs_iterator)
                if bars:
                    valid_bars_data = []
                    for bar in bars:
                        if hasattr(bar, 'timestamp') and hasattr(bar, 'open') and hasattr(bar, 'high') and hasattr(bar, 'low') and hasattr(bar, 'close') and hasattr(bar, 'volume'):
                            bar_dict = bar.__dict__
                            valid_bars_data.append({k: bar_dict.get(k) for k in [
                                                   'timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions']})
                    if valid_bars_data:
                        df = pd.DataFrame(valid_bars_data)
                        df['timestamp'] = pd.to_datetime(
                            df['timestamp'], unit='ms', utc=True)
                        df = df.set_index('timestamp').sort_index()
                        ohlcv_cols = ['open', 'high', 'low',
                                      'close', 'volume', 'vwap', 'transactions']
                        existing_cols = [
                            col for col in ohlcv_cols if col in df.columns]
                        ticker_data['prices'] = df[existing_cols]
                        logging.info(f" Loaded {len(df)} {timespan} bars.")
            except Exception as e:
                logging.error(
                    f"Price load failed: {e}", exc_info=True)
                processed_successfully = False
            try:  # Dividends
                div_iterator = self.client.list_dividends(
                    ticker=ticker, ex_dividend_date_gte=start_date_str, ex_dividend_date_lte=end_date_str, limit=1000)
                divs = list(div_iterator)
                kept_count = 0
                for d in divs:
                    if hasattr(d, 'ex_dividend_date') and d.ex_dividend_date:
                        try:
                            ex_div_date_obj = date.fromisoformat(
                                d.ex_dividend_date)
                            if start_date_obj <= ex_div_date_obj <= end_date_obj:
                                if hasattr(d, '__dict__'):
                                    event_details = d.__dict__.copy()
                                    event_details['event_type'] = 'dividend'
                                    event_details[
                                        'event_date'] = d.ex_dividend_date
                                    ticker_data['events'].append(event_details)
                                    kept_count += 1
                        except (ValueError, TypeError):
                            continue
                logging.info(
                    f" Found {len(divs)} raw dividends, kept {kept_count}.")
            except Exception as e:
                logging.error(
                    f"Dividend load failed: {e}", exc_info=True)
                processed_successfully = False
            try:  # Splits
                split_iterator = self.client.list_splits(
                    ticker=ticker, execution_date_gte=start_date_str, execution_date_lte=end_date_str, limit=1000)
                splits = list(split_iterator)
                kept_count = 0
                for s in splits:
                    if hasattr(s, 'execution_date') and s.execution_date:
                        try:
                            exec_date_obj = date.fromisoformat(
                                s.execution_date)
                            if start_date_obj <= exec_date_obj <= end_date_obj:
                                if hasattr(s, '__dict__'):
                                    event_details = s.__dict__.copy()
                                    event_details['event_type'] = 'split'
                                    event_details[
                                        'event_date'] = s.execution_date
                                    ticker_data['events'].append(event_details)
                                    kept_count += 1
                        except (ValueError, TypeError):
                            continue
                logging.info(
                    f" Found {len(splits)} raw splits, kept {kept_count}.")
            except Exception as e:
                logging.error(f"Split load failed: {e}", exc_info=True)
                processed_successfully = False
            try:  # Ticker Events
                ticker_change_events = self._fetch_ticker_events(
                    ticker, start_date_obj, end_date_obj)
                ticker_data['events'].extend(ticker_change_events)
            except Exception as e:
                logging.error(
                    f"Ticker change fetch failed: {e}", exc_info=True)
            equity_data[ticker] = ticker_data
            if not processed_successfully:
                logging.warning(f"Equity data incomplete for {ticker}.")
        logging.info("Equity data loading complete.")
        return equity_data
