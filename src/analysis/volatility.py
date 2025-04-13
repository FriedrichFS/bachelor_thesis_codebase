import pandas as pd
import numpy as np
from typing import Tuple, Dict, Union


class VolatilityAnalysis:
    """
    Class to handle historical volatility calculations for multiple tickers
    from a single DataFrame, and select a subset of stocks ranging from
    lowest to highest volatility.
    Additionally, it can build a volatility table including dynamic percentiles
    and average rolling volatilities by year segments, but restricted to only
    the n selected tickers.
    """

    def __init__(self) -> None:
        self.annual_trading_days = 252
        self.volatility_dict: Dict[str, float] = {}
        self.df: pd.DataFrame = pd.DataFrame()

    def compute_historical_volatility(self) -> Dict[str, float]:
        """
        Computes a single 'overall' annualized volatility per Ticker
        (std dev of daily log returns * sqrt(252)) across the entire date range.

        Returns
        -------
        dict
            { 'TICKER': float(volatility_value), ... } with annualized vols.
        """
        self.volatility_dict.clear()

        if 'Date' not in self.df.columns or not pd.api.types.is_datetime64_any_dtype(self.df['Date']):

            try:
                self.df['Date'] = pd.to_datetime(self.df['Date'])
            except Exception as e:
                print(f"Error converting 'Date' column to datetime: {e}")
                return {}

        grouped = self.df.groupby("Ticker", group_keys=False)

        for ticker, gdf in grouped:
            if gdf.empty or "Close" not in gdf.columns:
                self.volatility_dict[ticker] = np.nan
                continue

            gdf = gdf.sort_values("Date").copy()

            gdf['Close'] = pd.to_numeric(gdf['Close'], errors='coerce')

            gdf.dropna(subset=['Close'], inplace=True)
            if len(gdf) < 2:
                self.volatility_dict[ticker] = np.nan
                continue

            close_shifted = gdf["Close"].shift(1)
            valid_mask = (gdf["Close"] > 0) & (close_shifted > 0)
            gdf["LogReturn"] = np.nan  # Initialize
            gdf.loc[valid_mask, "LogReturn"] = np.log(
                gdf.loc[valid_mask, "Close"] / close_shifted[valid_mask])

            daily_returns = gdf["LogReturn"].dropna()

            if len(daily_returns) < 2:
                self.volatility_dict[ticker] = np.nan
            else:
                std_dev = daily_returns.std()
                vol_annualized = std_dev * np.sqrt(self.annual_trading_days)
                self.volatility_dict[ticker] = vol_annualized

        return self.volatility_dict

    @staticmethod
    def compute_rolling_volatility(df: pd.DataFrame, window_size: int = 21) -> pd.DataFrame:
        """Computes rolling volatility"""
        if df.empty or 'Close' not in df.columns or 'Date' not in df.columns:
            return pd.DataFrame()
        df = df.sort_values("Date").copy()
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df.dropna(subset=['Close'], inplace=True)
        if len(df) < 2:
            return pd.DataFrame()
        close_shifted = df["Close"].shift(1)
        valid_mask = (df["Close"] > 0) & (close_shifted > 0)
        df["LogReturn"] = np.nan
        df.loc[valid_mask, "LogReturn"] = np.log(
            df.loc[valid_mask, "Close"] / close_shifted[valid_mask])

        df["RollingVol"] = df["LogReturn"].rolling(
            window=window_size, min_periods=window_size//2).std() * np.sqrt(252)
        df = df.dropna(subset=["RollingVol"])
        return df[["Date", "RollingVol"]].copy()

    def build_volatility_table(self, original_data: pd.DataFrame, selected_tickers: Dict[str, float], rolling_window: int = 21, years_for_segments: int = 3) -> pd.DataFrame:
        """ Builds table for ONLY selected tickers """
        self.df = original_data.copy()
        self.compute_historical_volatility()
        results_list = []
        if not selected_tickers:
            return pd.DataFrame()
        if 'Date' not in self.df.columns or not pd.api.types.is_datetime64_any_dtype(self.df['Date']):
            self.df['Date'] = pd.to_datetime(self.df['Date'])
        grouped = self.df.groupby("Ticker", group_keys=False)
        st_list = sorted(selected_tickers.items(), key=lambda x: x[1])

        for (tkr, percentile_val) in st_list:
            overall_vol = self.volatility_dict.get(tkr, np.nan)
            row_data = {"Ticker": tkr, "VolatilityPercentile": f"{round(percentile_val)}th", "OverallVol": round(
                overall_vol, 3) if pd.notna(overall_vol) else np.nan}
            if tkr in grouped.groups:
                single_tkr_df = grouped.get_group(tkr).copy()
                rolling_info = self.compute_rolling_volatility(
                    single_tkr_df[["Date", "Close"]], window_size=rolling_window)
                if not rolling_info.empty:
                    n_rows = len(rolling_info)

                    segment_size = max(1, n_rows // years_for_segments)
                    seg_values = []
                    for seg_i in range(years_for_segments):
                        start_idx = seg_i * segment_size

                        end_idx = min(
                            (seg_i + 1) * segment_size, n_rows) if seg_i < years_for_segments - 1 else n_rows
                        if start_idx >= n_rows:
                            seg_values.append(np.nan)
                            continue
                        sub_df = rolling_info.iloc[start_idx:end_idx]
                        seg_values.append(
                            sub_df["RollingVol"].mean() if not sub_df.empty else np.nan)

                    for i in range(years_for_segments):
                        year_label = f"Year{years_for_segments-i}" + (
                            "(Latest)" if i == 0 else ("(Earliest)" if i == years_for_segments-1 else ""))
                        row_data[year_label] = round(seg_values[years_for_segments-1-i], 3) if i < len(
                            seg_values) and pd.notna(seg_values[years_for_segments-1-i]) else np.nan
                else:
                    for i in range(years_for_segments):
                        row_data[f"Year{years_for_segments-i}"] = np.nan
            else:
                row_data["OverallVol"] = np.nan
                for i in range(years_for_segments):
                    row_data[f"Year{years_for_segments-i}"] = np.nan
            results_list.append(row_data)
        cols = ["Ticker", "VolatilityPercentile", "OverallVol"] + \
            [f"Year{y}" for y in range(years_for_segments, 0, -1)]
        final_table = pd.DataFrame(results_list)

        rename_map = {
            f"Year{years_for_segments}": "YearN(Latest)", f"Year1": "Year1(Earliest)"}
        final_table.rename(columns=rename_map, inplace=True)
        return final_table

    def pick_stocks_by_volatility(self, ticker_data: pd.DataFrame, n: int = 5, rolling_window: int = 21, years_for_segments: int = 3) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
        """ Picks n stocks by vola percentile, calculates rolling vol, returns data & summary """
        self.df = ticker_data.copy()
        self.compute_historical_volatility()
        valid_tickers = [(tkr, vol) for tkr,
                         vol in self.volatility_dict.items() if pd.notna(vol)]
        valid_tickers.sort(key=lambda x: x[1])
        n_tickers = len(valid_tickers)
        if n_tickers == 0:
            return pd.DataFrame(), {}, pd.DataFrame()

        n = min(n, n_tickers)
        if n < 1:
            return pd.DataFrame(), {}, pd.DataFrame()

        indices = np.linspace(0, n_tickers - 1, n, dtype=int)
        selected_tickers = {}
        for idx_i, arr_i in enumerate(indices):
            ticker_chosen, _vol_chosen = valid_tickers[arr_i]
            percent_val = (idx_i / (n - 1)) * 100 if n > 1 else 100.0
            selected_tickers[ticker_chosen] = percent_val
        df_list = []
        if 'Date' not in self.df.columns or not pd.api.types.is_datetime64_any_dtype(self.df['Date']):
            self.df['Date'] = pd.to_datetime(self.df['Date'])
        grouped = self.df.groupby("Ticker", group_keys=False)
        for tkr in selected_tickers:
            if tkr in grouped.groups:
                gdf = grouped.get_group(tkr).copy()
                rolling_df = self.compute_rolling_volatility(
                    gdf[["Date", "Close"]], window_size=rolling_window)
                gdf = pd.merge(gdf, rolling_df, how="left", on="Date")
                df_list.append(gdf)
        final_df = pd.concat(
            df_list, ignore_index=True) if df_list else pd.DataFrame()
        volatility_table = self.build_volatility_table(
            original_data=self.df, selected_tickers=selected_tickers, rolling_window=rolling_window, years_for_segments=years_for_segments)
        return final_df, selected_tickers, volatility_table
