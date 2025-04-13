from itertools import combinations
from scipy.stats import wilcoxon
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import seaborn as sns
import os
import logging
from typing import List, Dict, Optional
from datetime import datetime, timezone

try:
    from src.config.settings import Settings
    from src.config.enums import TimeInterval
    CONFIG_IMPORTED = True
except ImportError:
    class Settings:
        output_base_dir = "pipeline_output_final"
        raw_data_subdir = "raw_data"
        graph_data_subdir = "graph_data"
        vola_rolling_window_days = 21
        dte_targets = [30, 90, 180, 365]
        resolution_analysis = type("Enum", (), {"value": "day"})()
    from enum import Enum

    class TimeInterval(Enum):
        DAILY = "day"
    logger = logging.getLogger(__name__)
    logger.error(
        "Could not import Settings or Enums. Using default paths and settings.")
    CONFIG_IMPORTED = False

logger = logging.getLogger(__name__)


class ThesisChartGenerator:
    """ Generates plots for thesis analysis. """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_dir = settings.output_base_dir
        self.raw_data_dir = os.path.join(
            self.base_dir, settings.raw_data_subdir)
        self.graph_data_dir = os.path.join(
            self.base_dir, settings.graph_data_subdir)
        self.plots_dir = os.path.join(self.base_dir, "thesis_plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        logger.info(f"Saving plots to: {os.path.abspath(self.plots_dir)}")
        # Styling
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams['savefig.dpi'] = 300
        self.figure_size = (10, 5)
        self.title_fontsize = 14
        self.label_fontsize = 11
        self.tick_fontsize = 9
        self.legend_fontsize = 9
        self.line_markersize = 4
        self.dpi = 300
        self._load_all_data()
        self._prepare_runtime_data()

    # --- Load Data Helpers ---
    def _load_csv(self, file_path, **kwargs) -> Optional[pd.DataFrame]:
        """Loads CSV with basic error handling."""
        full_path = os.path.abspath(file_path)
        try:
            if not os.path.exists(full_path):
                logger.error(f"NF: {full_path}")
                return None
            df = pd.read_csv(full_path, **kwargs)
            logger.info(
                f" Loaded {os.path.basename(full_path)} ({len(df)} rows)")
            return df
        except Exception as e:
            logger.error(f"Failed load {full_path}: {e}", exc_info=True)
            return None

    def _ensure_datetime_utc(self, df: Optional[pd.DataFrame], col_name: str) -> Optional[pd.DataFrame]:
        """Attempts to convert a column to timezone-aware UTC datetime."""
        if df is None or col_name not in df.columns:
            return df
        try:
            df[col_name] = pd.to_datetime(
                df[col_name], errors='coerce', utc=True)
            if df[col_name].isnull().all() and not df.empty:
                logger.warning(f"Col '{col_name}' all NaNs post conversion.")

            df = df.dropna(subset=[col_name])
        except Exception as e:
            logger.error(f"Failed convert '{col_name}' datetime UTC: {e}")
            return None
        return df

    def _ensure_index_datetime_utc(self, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Attempts to convert DataFrame index to timezone-aware UTC datetime."""
        if df is None:
            return None
        if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None or df.index.tz != timezone.utc:
            try:
                original_index_name = df.index.name
                df.index = pd.to_datetime(df.index, errors='coerce', utc=True)
                df.index.name = original_index_name
                if df.index.isnull().all() and not df.empty:
                    logger.warning(f"Index all NaNs post conversion.")
                    return None

                df = df[~df.index.isna()]
            except Exception as e:
                logger.error(f"Failed convert index datetime UTC: {e}")
                return None
        return df

    def _load_all_data(self):
        """Loads all necessary dataframes."""
        logger.info("Loading data for chart generation...")
        g = self.graph_data_dir
        r = self.raw_data_dir
        res = getattr(
            getattr(self.settings, 'resolution_analysis', None), 'value', 'day')
        paths = {
            "vola_summary": os.path.join(g, "vola_summary_table.csv"),
            "rolling_vola": os.path.join(g, "daily_rolling_vola_selected_tickers.csv"),
            "py_runtime": os.path.join(g, "py_runtime_benchmark_results.csv"),
            "cpp_runtime": os.path.join(g, "cpp_runtime_benchmark_results.csv"),
            "tree_steps": os.path.join(g, "tree_model_runtime_vs_steps.csv"),
            "pricing_error": "model_pricing_errors_runtime.csv",
            "equity_prices": os.path.join(r, f"final_equity_historical_prices_{res}.csv"),
            "events": os.path.join(r, "final_equity_events.csv")
        }

        self.vola_summary_table = self._load_csv(paths["vola_summary"])
        self.rolling_vola_df = self._load_csv(paths["rolling_vola"])
        self.py_runtime_df = self._load_csv(paths["py_runtime"])
        self.cpp_runtime_df = self._load_csv(paths["cpp_runtime"])
        self.tree_steps_df = self._load_csv(paths["tree_steps"])
        pricing_error_path = paths["pricing_error"]
        if not os.path.exists(pricing_error_path):
            pricing_error_path = os.path.join(g, pricing_error_path)
        if os.path.exists(pricing_error_path):
            self.pricing_error_df = self._load_csv(pricing_error_path)
        else:
            logger.error(f"Pricing error file not found. Plotting will fail.")
            self.pricing_error_df = None
        self.equity_prices_df = self._load_csv(
            paths["equity_prices"], index_col='timestamp')
        self.events_df = self._load_csv(paths["events"])

        logger.info("Ensuring timezone awareness...")
        self.rolling_vola_df = self._ensure_datetime_utc(
            self.rolling_vola_df, 'Date')
        if self.py_runtime_df is not None and 'timestamp' in self.py_runtime_df.columns:
            self.py_runtime_df = self._ensure_datetime_utc(
                self.py_runtime_df, 'timestamp')
        if self.cpp_runtime_df is not None and 'timestamp' in self.cpp_runtime_df.columns:
            self.cpp_runtime_df = self._ensure_datetime_utc(
                self.cpp_runtime_df, 'timestamp')
        if self.pricing_error_df is not None and 'timestamp' in self.pricing_error_df.columns:
            self.pricing_error_df = self._ensure_datetime_utc(
                self.pricing_error_df, 'timestamp')
        self.events_df = self._ensure_datetime_utc(
            self.events_df, 'event_date')
        self.equity_prices_df = self._ensure_index_datetime_utc(
            self.equity_prices_df)
        logger.info("Finished loading data.")

    def _prepare_runtime_data(self):
        """Aggregates and combines Python and C++ runtime data."""
        self.combined_runtime_avg_df = None
        if self.py_runtime_df is None or self.cpp_runtime_df is None or self.py_runtime_df.empty or self.cpp_runtime_df.empty:
            logger.warning("Missing/empty Py/C++ runtime data.")
            return
        logger.info("Preparing combined runtime data...")
        py_df = self.py_runtime_df.copy()
        cpp_df = self.cpp_runtime_df.copy()
        py_df['source'] = 'Python'
        cpp_df['source'] = 'C++'
        id_cols_base = ['target_dte_group', 'option_type', 'model']
        common_optional_cols = list(set(py_df.columns) & set(
            cpp_df.columns) - set(id_cols_base + ['calc_time_sec', 'source']))
        id_cols = id_cols_base + common_optional_cols
        if not all(col in py_df.columns for col in id_cols + ['calc_time_sec']) or not all(col in cpp_df.columns for col in id_cols + ['calc_time_sec']):
            logger.error(f"Runtime data missing required columns. Skipping.")
            return
        try:
            py_avg_df = py_df.groupby(id_cols, as_index=False, observed=True)[
                'calc_time_sec'].mean()
            cpp_avg_df = cpp_df.groupby(id_cols, as_index=False, observed=True)[
                'calc_time_sec'].mean()
        except Exception as e:
            logger.error(
                f"Error during runtime aggregation: {e}", exc_info=True)
            return
        py_avg_df['source'] = 'Python'
        cpp_avg_df['source'] = 'C++'
        self.combined_runtime_avg_df = pd.concat(
            [py_avg_df, cpp_avg_df], ignore_index=True)
        self.combined_runtime_avg_df['calc_time_ms'] = self.combined_runtime_avg_df['calc_time_sec'] * 1000
        self.combined_runtime_avg_df.dropna(
            subset=['calc_time_ms'], inplace=True)
        if self.combined_runtime_avg_df.empty:
            logger.warning("Combined runtime data empty after aggregation.")
        else:
            logger.info("Combined runtime data prepared.")
            logger.debug(
                f"Models: {self.combined_runtime_avg_df['model'].unique()}")

    def _save_plot(self, fig_or_ax, filename):
        """Helper to save plots."""
        path = os.path.join(self.plots_dir, filename)
        fig = None
        try:
            if isinstance(fig_or_ax, plt.Axes):
                fig = fig_or_ax.figure
            elif hasattr(fig_or_ax, 'figure'):
                fig = fig_or_ax.figure
            elif isinstance(fig_or_ax, plt.Figure):
                fig = fig_or_ax
            else:
                logger.error(
                    f"Invalid object type to save plot {filename}: {type(fig_or_ax)}")
                plt.close()
                return
            fig.savefig(path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved plot: {path}")
        except Exception as e:
            logger.error(f"Failed save plot {path}: {e}", exc_info=True)
        finally:
            if fig is not None:
                plt.close(fig)
            elif isinstance(fig_or_ax, plt.Figure):
                plt.close(fig_or_ax)
            else:
                plt.close()

    # --- Plotting Methods ---

    def plot_equity_price_with_dividends(self):
        """
        Plots historical equity price with markers for dividend events,
        split into two separate figures/files (4 tickers then 6 tickers).
        """
        if self.equity_prices_df is None or self.events_df is None or self.vola_summary_table is None:
            logger.warning(
                "Missing equity/events/summary data. Skipping equity/dividend plots.")
            return
        if not isinstance(self.equity_prices_df.index, pd.DatetimeIndex):
            logger.error("Equity index is not a DatetimeIndex. Skipping plot.")
            return

        tickers = self.vola_summary_table['Ticker'].unique()
        n_total_tickers = len(tickers)
        if n_total_tickers == 0:
            logger.warning(
                "No tickers found in vola_summary_table. Skipping equity/dividend plots.")
            return
        if 'underlying_ticker' not in self.equity_prices_df.columns:

            logger.error(
                "Missing 'underlying_ticker' column in equity_prices_df.")
            return

        dividends = None
        if 'event_type' in self.events_df.columns and 'event_date' in self.events_df.columns and pd.api.types.is_datetime64_any_dtype(self.events_df['event_date']):
            dividends = self.events_df[self.events_df['event_type'] == 'dividend'].copy(
            )
            if not dividends.empty:
                dividends['cash_amount_num'] = pd.to_numeric(
                    dividends['cash_amount'], errors='coerce')

                if 'underlying_ticker' not in dividends.columns:
                    logger.error(
                        "Missing 'underlying_ticker' in dividend events data.")
                    dividends = None
                else:
                    dividends.dropna(
                        subset=['event_date', 'cash_amount_num', 'underlying_ticker'], inplace=True)
                    if dividends.empty:
                        dividends = None

        tickers_1 = tickers[:4]
        n_tickers_1 = len(tickers_1)
        if n_tickers_1 > 0:
            ncols1 = 2
            nrows1 = (n_tickers_1 + ncols1 - 1) // ncols1  # Should be 2x2
            fig1, axes1 = plt.subplots(nrows=nrows1, ncols=ncols1, figsize=(
                self.figure_size[0]*ncols1*0.7, self.figure_size[1]*nrows1*0.7), sharex=True, squeeze=False)
            axes_flat1 = axes1.flatten()
            logger.info(
                f"Generating equity/dividend plot 1 ({n_tickers_1} tickers)...")

            for i, ticker in enumerate(tickers_1):
                ax = axes_flat1[i]

                ticker_prices = self.equity_prices_df[self.equity_prices_df['underlying_ticker'] == ticker]

                if ticker_prices.empty:
                    ax.set_title(f"{ticker} (No Price Data)")
                    ax.text(0.5, 0.5, 'No Data', ha='center',
                            va='center', transform=ax.transAxes)
                    continue

                sns.lineplot(data=ticker_prices, x=ticker_prices.index,
                             y='close', ax=ax, label='Close Price', linewidth=1.5)

                if dividends is not None:
                    ticker_dividends = dividends[dividends['underlying_ticker'] == ticker]
                    if not ticker_dividends.empty:
                        try:

                            ticker_prices_sorted = ticker_prices[[
                                'close']].sort_index()
                            div_plot_data = pd.merge_asof(ticker_dividends.sort_values('event_date'),
                                                          ticker_prices_sorted,
                                                          left_on='event_date',
                                                          right_index=True,
                                                          direction='nearest',
                                                          tolerance=pd.Timedelta('2 day'))
                            div_plot_data.dropna(
                                subset=['close'], inplace=True)
                            if not div_plot_data.empty:
                                sns.scatterplot(data=div_plot_data, x='event_date', y='close', size='cash_amount_num', sizes=(
                                    40, 150), color='red', marker='o', ax=ax, legend=False, label='Dividend Ex-Date', zorder=5)
                        except Exception as e:
                            logger.error(
                                f"Error merging/plotting dividends for {ticker} (Plot 1): {e}")

                ax.set_title(f"{ticker} Price & Dividends",
                             fontsize=self.label_fontsize)
                ax.set_ylabel("Price ($)", fontsize=self.label_fontsize-1)
                ax.tick_params(axis='both', which='major',
                               labelsize=self.tick_fontsize)
                ax.yaxis.set_major_formatter(
                    mticker.FormatStrFormatter('%.2f'))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
                plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
                ax.legend(fontsize=self.tick_fontsize)
                ax.grid(True, which='major', axis='both',
                        linestyle='--', linewidth=0.5)

            for j in range(n_tickers_1, nrows1 * ncols1):
                if j < len(axes_flat1):
                    fig1.delaxes(axes_flat1[j])

            fig1.suptitle("Historical Equity Prices with Dividend Events (Part 1)",
                          fontsize=self.title_fontsize, y=1.02)
            fig1.tight_layout(h_pad=2.0, rect=[0, 0.03, 1, 0.95])
            self._save_plot(fig1, "equity_prices_dividends_1.png")
        else:
            logger.warning("No tickers for the first equity/dividend plot.")

        tickers_2 = tickers[4:]
        n_tickers_2 = len(tickers_2)
        if n_tickers_2 > 0:
            ncols2 = 2
            nrows2 = (n_tickers_2 + ncols2 - 1) // ncols2  # Should be 3x2
            fig2, axes2 = plt.subplots(nrows=nrows2, ncols=ncols2, figsize=(
                self.figure_size[0]*ncols2*0.7, self.figure_size[1]*nrows2*0.7), sharex=True, squeeze=False)
            axes_flat2 = axes2.flatten()
            logger.info(
                f"Generating equity/dividend plot 2 ({n_tickers_2} tickers)...")

            for i, ticker in enumerate(tickers_2):
                if i >= len(axes_flat2):
                    logger.warning(
                        f"Index {i} out of bounds for axes_flat2 (size {len(axes_flat2)})")
                    break
                ax = axes_flat2[i]

                ticker_prices = self.equity_prices_df[self.equity_prices_df['underlying_ticker'] == ticker]

                if ticker_prices.empty:
                    ax.set_title(f"{ticker} (No Price Data)")
                    ax.text(0.5, 0.5, 'No Data', ha='center',
                            va='center', transform=ax.transAxes)
                    continue

                sns.lineplot(data=ticker_prices, x=ticker_prices.index,
                             y='close', ax=ax, label='Close Price', linewidth=1.5)
                if dividends is not None:
                    ticker_dividends = dividends[dividends['underlying_ticker'] == ticker]
                    if not ticker_dividends.empty:
                        try:
                            # Ensure index is sorted for merge_asof
                            ticker_prices_sorted = ticker_prices[[
                                'close']].sort_index()
                            div_plot_data = pd.merge_asof(ticker_dividends.sort_values('event_date'),
                                                          ticker_prices_sorted,
                                                          left_on='event_date',
                                                          right_index=True,
                                                          direction='nearest',
                                                          tolerance=pd.Timedelta('2 day'))
                            div_plot_data.dropna(
                                subset=['close'], inplace=True)
                            if not div_plot_data.empty:
                                sns.scatterplot(data=div_plot_data, x='event_date', y='close', size='cash_amount_num', sizes=(
                                    40, 150), color='red', marker='o', ax=ax, legend=False, label='Dividend Ex-Date', zorder=5)
                        except Exception as e:
                            logger.error(
                                f"Error merging/plotting dividends for {ticker} (Plot 2): {e}")

                ax.set_title(f"{ticker} Price & Dividends",
                             fontsize=self.label_fontsize)
                ax.set_ylabel("Price ($)", fontsize=self.label_fontsize-1)
                ax.tick_params(axis='both', which='major',
                               labelsize=self.tick_fontsize)
                ax.yaxis.set_major_formatter(
                    mticker.FormatStrFormatter('%.2f'))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
                plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
                ax.legend(fontsize=self.tick_fontsize)
                ax.grid(True, which='major', axis='both',
                        linestyle='--', linewidth=0.5)

            last_plot_index = i

            for j in range(last_plot_index + 1, nrows2 * ncols2):
                if j < len(axes_flat2):
                    fig2.delaxes(axes_flat2[j])

            fig2.suptitle("Historical Equity Prices with Dividend Events (Part 2)",
                          fontsize=self.title_fontsize, y=1.02)
            fig2.tight_layout(h_pad=2.0, rect=[0, 0.03, 1, 0.95])
            self._save_plot(fig2, "equity_prices_dividends_2.png")
        else:
            logger.warning("No tickers for the second equity/dividend plot.")

    def create_split_table(self):
        """Creates and saves a table of stock split events."""
        if self.events_df is None:
            logger.warning("Missing event data for split table.")
            return
        splits = self.events_df[self.events_df['event_type'] == 'split'].copy()
        if splits.empty:
            logger.info("No split events found.")
            return
        if 'event_date' in splits.columns and pd.api.types.is_datetime64_any_dtype(splits['event_date']):
            splits['event_date_str'] = splits['event_date'].dt.strftime(
                '%Y-%m-%d')
        else:
            splits['event_date_str'] = 'N/A'
        splits_table = splits[['underlying_ticker', 'event_date_str', 'split_from', 'split_to']].rename(
            columns={'underlying_ticker': 'Ticker', 'event_date_str': 'Execution Date', 'split_from': 'Split From', 'split_to': 'Split To'})
        splits_table.sort_values(by=['Ticker', 'Execution Date'], inplace=True)
        filename = os.path.join(self.plots_dir, "equity_splits_table.csv")
        try:
            splits_table.to_csv(filename, index=False)
            logger.info(f"Saved split events table to: {filename}")
            print(
                "\nStock Splits Table:")
            print(splits_table.to_string(index=False, na_rep='N/A'))
        except Exception as e:
            logger.error(f"Failed to save splits table: {e}")

    def plot_rolling_volatility(self):
        """Plots rolling volatility for ALL selected tickers on ONE graph."""
        if self.rolling_vola_df is None or self.rolling_vola_df.empty:
            logger.warning("Missing rolling vola data. Skipping plot.")
            return
        if 'Date' not in self.rolling_vola_df.columns or not pd.api.types.is_datetime64_any_dtype(self.rolling_vola_df['Date']):
            logger.error("Rolling vola 'Date' column not datetime. Skipping.")
            return
        tickers = self.rolling_vola_df['Ticker'].unique()
        if len(tickers) == 0:
            logger.info("No tickers in rolling vola data.")
            return
        logger.info(f"Plotting rolling volatility for {len(tickers)} tickers.")
        plt.figure(figsize=(self.figure_size[0]*1.1, self.figure_size[1]))
        try:
            sns.lineplot(data=self.rolling_vola_df, x='Date',
                         y='RollingVol', hue='Ticker', linewidth=1.2, alpha=0.8)
        except Exception as e:
            logger.error(f"Error during rolling volatility lineplot: {e}")
            plt.close()
            return
        plt.title(f"{self.settings.vola_rolling_window_days}-Day Rolling Volatility",
                  fontsize=self.title_fontsize)
        plt.xlabel("Date", fontsize=self.label_fontsize)
        plt.ylabel("Annualized Volatility", fontsize=self.label_fontsize)
        plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        plt.tick_params(axis='both', which='major',
                        labelsize=self.tick_fontsize)
        plt.xticks(rotation=30, ha='right')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
        plt.legend(title='Ticker', bbox_to_anchor=(
            1.02, 1), loc='upper left', borderaxespad=0., fontsize=self.tick_fontsize)
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        plt.tight_layout(rect=[0, 0, 0.88, 1])
        self._save_plot(plt.gcf(), "rolling_volatility_combined.png")

    def plot_tree_runtime_vs_steps(self):
        """
        Plots CRR/LR runtime vs N on a log-scale Y axis, showing separate
        subplots for call and put options, and includes a working legend.
        """
        if self.tree_steps_df is None or self.tree_steps_df.empty:
            logger.warning(
                "Missing tree runtime vs steps data (tree_steps_df). Skipping plot.")
            return

        required_cols = ['avg_calc_time_sec', 'N_steps',
                         'model', 'target_dte_group', 'option_type']
        if not all(c in self.tree_steps_df.columns for c in required_cols):
            logger.error(f"Tree runtime data missing required columns: {required_cols}. "
                         f"Has: {self.tree_steps_df.columns.tolist()}. Skipping plot.")
            return

        plot_df = self.tree_steps_df.copy()
        if 'avg_calc_time_ms' not in plot_df.columns:
            if 'avg_calc_time_sec' in plot_df.columns:
                plot_df['avg_calc_time_ms'] = plot_df['avg_calc_time_sec'] * 1000
            else:
                logger.error("Missing 'avg_calc_time_sec'.")
                return

        plot_df.dropna(subset=['avg_calc_time_ms', 'N_steps'], inplace=True)
        if plot_df.empty:
            logger.warning("No valid tree runtime data points.")
            return

        logger.info("Generating tree model runtime vs steps plot...")
        try:

            model_hue_order = [m for m in ['CRR', 'LeisenReimer']
                               if m in plot_df['model'].unique()]
            palette = sns.color_palette('Set2', n_colors=len(model_hue_order))
            model_palette = dict(zip(model_hue_order, palette))

            g = sns.relplot(
                data=plot_df,
                x='N_steps',
                y='avg_calc_time_ms',
                hue='model',
                hue_order=model_hue_order,
                style='target_dte_group',
                col='option_type',
                col_order=['call', 'put'],
                kind='line',
                marker='.',
                markersize=self.line_markersize,
                height=self.figure_size[1] * 0.7,
                aspect=1.1,
                facet_kws={'sharey': True},
                palette=model_palette,
                legend='full'
            )

            if g.axes.size == 0:
                logger.warning("Relplot created no axes.")
                plt.close()
                return

            g.fig.suptitle("Tree Model Runtime vs. Steps (Log Scale Y)",
                           fontsize=self.title_fontsize, y=1.06)
            g.set_axis_labels("Number of Steps (N)", "Average Calc Time (ms)",
                              fontsize=self.label_fontsize)
            g.set_titles(col_template="{col_name} Options",
                         size=self.label_fontsize)

            for ax in g.axes.flat:
                if not ax.lines:
                    continue
                ax.set_yscale('log')
                ax.grid(True, which="both", ls="--", linewidth=0.5)
                ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
                ax.yaxis.set_minor_formatter(mticker.NullFormatter())
                ax.tick_params(axis='both', which='major',
                               labelsize=self.tick_fontsize)
                ax.tick_params(axis='x', rotation=0)

            if g._legend is not None:
                g._legend.set_bbox_to_anchor(
                    (0.5, 1.0))
                g._legend.set_loc("upper center")
                g._legend.set_title("Model / DTE")

            plt.tight_layout(rect=[0, 0.03, 1, 0.92])

            self._save_plot(g.fig, "tree_runtime_vs_steps_combined.png")

        except Exception as e:
            logger.error(
                f"Error generating tree runtime vs steps plot: {e}", exc_info=True)
            plt.close(getattr(g, 'fig', plt.gcf()))

    def plot_pricing_error_magnitude_by_dte(self):
        """
        Plots the distribution of pricing errors grouped by DTE (on the y-axis),
        with pricing_error on the x-axis, colored by model.
        Separate columns for calls and puts.
        Also performs statistical comparison of calculated prices between models
        and saves the results to a CSV file.
        """
        if self.pricing_error_df is None or self.pricing_error_df.empty:
            logger.warning(
                "Missing pricing error data. Skipping error magnitude plot and stats.")
            return

        required_cols = ['pricing_error', 'calculated_price', 'market_price',
                         'model', 'target_dte_group', 'option_type',
                         'option_ticker', 'timestamp']
        if not all(col in self.pricing_error_df.columns for col in required_cols):
            missing = [
                col for col in required_cols if col not in self.pricing_error_df.columns]
            logger.error(
                f"Pricing error data missing required columns for plot/stats: Missing {missing}")
            return

        plot_data = self.pricing_error_df.dropna(subset=['pricing_error', 'model',
                                                         'target_dte_group', 'option_type']).copy()

        plot_data['pricing_error'] = pd.to_numeric(
            plot_data['pricing_error'], errors='coerce')
        plot_data.dropna(subset=['pricing_error'], inplace=True)

        logger.info("Performing statistical comparison of calculated prices...")
        stats_results = []
        stats_data = self.pricing_error_df[required_cols].copy()
        stats_data.dropna(subset=['calculated_price', 'model', 'target_dte_group',
                                  'option_type', 'option_ticker', 'timestamp'], inplace=True)

        stats_data['calculated_price'] = pd.to_numeric(
            stats_data['calculated_price'], errors='coerce')

        stats_data.dropna(subset=['calculated_price'], inplace=True)

        if not stats_data.empty:

            try:
                stats_data['unique_option_id'] = stats_data['option_ticker'].astype(
                    str) + '_' + stats_data['timestamp'].astype(str)
            except Exception as e:
                logger.error(
                    f"Failed to create unique_option_id: {e}. Check 'option_ticker' and 'timestamp' columns. Skipping stats.")
                stats_data = pd.DataFrame()

            if not stats_data.empty:

                try:
                    pivot_df = stats_data.pivot_table(
                        index=['unique_option_id',
                               'target_dte_group', 'option_type'],
                        columns='model',
                        values='calculated_price'
                    )

                    pivot_df.reset_index(inplace=True)

                except Exception as e:
                    logger.error(
                        f"Failed to pivot data for statistical comparison: {e}. Skipping stats.")
                    pivot_df = pd.DataFrame()

                if not pivot_df.empty:
                    models = sorted(
                        [m for m in stats_data['model'].unique() if m in pivot_df.columns])

                    model_pairs = list(combinations(models, 2))

                    try:
                        pivot_df['target_dte_group'] = pivot_df['target_dte_group'].astype(
                            int)
                    except (ValueError, TypeError):
                        logger.warning(
                            "Could not convert target_dte_group to int for stats grouping, keeping as is.")
                        pivot_df['target_dte_group'] = pivot_df['target_dte_group'].astype(
                            str)

                    grouped = pivot_df.groupby(
                        ['target_dte_group', 'option_type'])

                    for (dte_group, option_type), group_data in grouped:
                        n_options_total_in_group = len(group_data)

                        for model1, model2 in model_pairs:
                            comparison_slice = group_data[[
                                model1, model2]].dropna()
                            n_valid_comparisons = len(comparison_slice)

                            if n_valid_comparisons < 5:
                                logger.debug(
                                    f"Skipping {model1} vs {model2} for {dte_group} DTE {option_type} - only {n_valid_comparisons} valid pairs.")
                                continue

                            price1 = comparison_slice[model1]
                            price2 = comparison_slice[model2]

                            diff = price1 - price2

                            abs_diff = diff.abs()
                            avg_price = (price1 + price2) / 2
                            epsilon = 1e-9
                            rel_diff_pct = (
                                abs_diff / (avg_price.abs() + epsilon)) * 100

                            mean_abs_diff = abs_diff.mean()
                            median_abs_diff = abs_diff.median()
                            std_abs_diff = abs_diff.std()
                            mean_rel_diff_pct = rel_diff_pct.mean()
                            median_rel_diff_pct = rel_diff_pct.median()
                            std_rel_diff_pct = rel_diff_pct.std()

                            try:
                                stat, p_value = wilcoxon(
                                    diff, zero_method='pratt')

                                if pd.isna(p_value):

                                    p_value = 1.0 if np.allclose(
                                        diff, 0, atol=epsilon) else np.nan
                            except ValueError as e:
                                logger.warning(
                                    f"Wilcoxon test failed for {model1} vs {model2}, DTE {dte_group}, {option_type}: {e}")
                                stat, p_value = np.nan, np.nan
                                if np.allclose(diff, 0, atol=epsilon):
                                    p_value = 1.0

                            stats_results.append({
                                'target_dte_group': dte_group,
                                'option_type': option_type,
                                'model_1': model1,
                                'model_2': model2,
                                'n_valid_comparisons': n_valid_comparisons,
                                'mean_abs_diff': mean_abs_diff,
                                'median_abs_diff': median_abs_diff,
                                'std_abs_diff': std_abs_diff,
                                'mean_rel_diff_pct': mean_rel_diff_pct,
                                'median_rel_diff_pct': median_rel_diff_pct,
                                'std_rel_diff_pct': std_rel_diff_pct,
                                'wilcoxon_p_value': p_value
                            })

        if stats_results:
            stats_df = pd.DataFrame(stats_results)

            stats_df = stats_df.sort_values(
                by=['option_type', 'target_dte_group', 'model_1', 'model_2'])
            stats_filename = os.path.join(
                self.plots_dir, "model_price_comparison_stats.csv")
            try:
                stats_df.to_csv(stats_filename, index=False,
                                float_format='%.6g')
                logger.info(
                    f"Saved model price comparison statistics to: {stats_filename}")
            except Exception as e:
                logger.error(f"Failed to save comparison statistics CSV: {e}")
        else:
            logger.warning(
                "No statistical comparison results were generated (maybe insufficient data or pairs).")
        try:

            plot_data['target_dte_group_numeric'] = pd.to_numeric(
                plot_data['target_dte_group'])
            dte_order = sorted(plot_data['target_dte_group_numeric'].unique())
            plot_data['target_dte_group_plot'] = plot_data['target_dte_group_numeric'].astype(
                str)
            dte_order_plot = [str(d) for d in dte_order]

            y_axis_col = 'target_dte_group_numeric'
        except (ValueError, TypeError):
            logger.warning(
                "target_dte_group non-numeric. Using string sorting for plot order.")
            plot_data['target_dte_group_plot'] = plot_data['target_dte_group'].astype(
                str)

            try:
                dte_order_plot = sorted(
                    plot_data['target_dte_group_plot'].unique(), key=int)
            except ValueError:
                dte_order_plot = sorted(
                    plot_data['target_dte_group_plot'].unique())
            y_axis_col = 'target_dte_group_plot'

        if plot_data.empty:
            logger.info(
                "No valid data points remaining for pricing error magnitude plot.")
            return

        plot_models = sorted(plot_data['model'].unique())
        if not plot_models:
            logger.warning(
                "No models found in the data prepared for plotting.")
            return

        logger.info(
            f"Generating pricing error distribution plots by DTE Group for models: {plot_models}...")
        try:

            g = sns.catplot(
                data=plot_data,
                x='pricing_error',
                y=y_axis_col,
                hue='model',
                col='option_type',
                kind='box',
                order=dte_order_plot,
                hue_order=plot_models,
                orient='h',
                palette='Set2',

                showfliers=False,
                height=5,
                aspect=1.1
            )

            g.fig.suptitle('Pricing Error Distribution by DTE Group and Model',
                           y=1.03, fontsize=self.label_fontsize + 2)
            g.set_axis_labels("Pricing Error (Model - Market) ($)",
                              "Target DTE Group", fontsize=self.label_fontsize)

            g.set_titles(
                col_template="{col_name} Options", size=self.label_fontsize)
            for ax in g.axes.flat:

                current_title = ax.get_title()
                new_title = current_title.replace(
                    "call Options", "Call Options").replace("put Options", "Put Options")
                ax.set_title(new_title)

                ax.axvline(0, color='grey', linestyle='--',
                           linewidth=1.0, zorder=0)
                ax.grid(True, which="major", axis='x',
                        linestyle='--', linewidth=0.5)
                ax.tick_params(axis='both', which='major',
                               labelsize=self.tick_fontsize)

                ax.xaxis.set_major_formatter(
                    mticker.FormatStrFormatter('$%.2f'))

                if y_axis_col == 'target_dte_group_numeric':
                    ax.set_yticklabels(
                        [f"{int(tick)}" for tick in ax.get_yticks()])

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            self._save_plot(g, "pricing_error_magnitude_by_dte.png")

        except Exception as e:
            logger.error(
                f"Error generating pricing error magnitude by DTE plot: {e}", exc_info=True)
            plt.close(getattr(g, 'fig', plt.gcf()))

    def plot_greeks_distribution(self):
        """Plots distributions of calculated Greeks faceted by DTE and Option Type."""
        if self.pricing_error_df is None or self.pricing_error_df.empty:
            logger.warning("Missing pricing error data. Skipping Greeks plot.")
            return
        greeks_to_plot = ['delta', 'gamma', 'vega', 'theta', 'rho']
        available_greeks = [
            g for g in greeks_to_plot if g in self.pricing_error_df.columns]
        if not available_greeks:
            logger.warning("No greek columns found. Skipping Greeks plot.")
            return
        required_cols = available_greeks + \
            ['model', 'target_dte_group', 'option_type']
        plot_data_base = self.pricing_error_df.dropna(
            subset=required_cols).copy()
        for greek in available_greeks:
            plot_data_base[greek] = pd.to_numeric(
                plot_data_base[greek], errors='coerce')
        plot_data_clean = plot_data_base.dropna(subset=required_cols)
        if plot_data_clean.empty:
            logger.info("No valid data points for Greeks plot.")
            return
        logger.info(
            f"Generating distribution plots for Greeks: {available_greeks}")
        dte_order = sorted(plot_data_clean['target_dte_group'].unique())
        for greek in available_greeks:
            if plot_data_clean[greek].isnull().all():
                logger.warning(
                    f"No valid numeric data for Greek '{greek}'. Skipping.")
                continue
            try:
                g = sns.catplot(data=plot_data_clean, x='model', y=greek, col='target_dte_group', row='option_type', kind='box',
                                hue='model',
                                legend=False,
                                order=sorted(plot_data_clean['model'].unique()), col_order=dte_order, height=3.5, aspect=1.0,
                                palette='Set2', showfliers=False, linewidth=1.0, whis=[5, 95])

                g.fig.suptitle(
                    f'{greek.capitalize()} Distribution by Model, DTE, and Type', fontsize=self.title_fontsize, y=1.03)
                g.set_axis_labels("Model", f"{greek.capitalize()}")
                g.set_titles(
                    col_template="{col_name} DTE", row_template="{row_name} Options")
                for ax in g.axes.flat:
                    current_title = ax.get_title()
                    new_title = current_title
                    parts = current_title.split('|')
                    if len(parts) == 2:
                        col_part, row_part = parts[0].strip(), parts[1].strip().replace(
                            " Options", "").capitalize() + " Options"
                        new_title = f"{col_part} | {row_part}"
                    elif " Options" in current_title:
                        new_title = current_title.replace(
                            " Options", "").capitalize() + " Options"
                    ax.set_title(new_title, fontsize=self.label_fontsize)

                q_low, q_high = 0.02, 0.98
                y_min, y_max = plot_data_clean[greek].quantile(
                    q_low), plot_data_clean[greek].quantile(q_high)
                if pd.notna(y_min) and pd.notna(y_max) and y_max > y_min:
                    padding = (y_max - y_min) * 0.1
                    final_y_min, final_y_max = y_min - padding, y_max + padding
                if greek == 'theta' and final_y_max > 0:
                    final_y_max = padding * 0.1
                if greek == 'gamma' and final_y_min < 0:
                    final_y_min = -padding * 0.1
                if greek == 'vega' and final_y_min < 0:
                    final_y_min = -padding * 0.1
                if final_y_min < final_y_max:
                    g.set(ylim=(final_y_min, final_y_max))

                else:
                    logger.warning(
                        f"Could not determine quantile y-limits for {greek}.")
                g.map(plt.grid, which="major", ls="--", linewidth=0.5, axis='y')
                g.tick_params(axis='both', which='major',
                              labelsize=self.tick_fontsize)
                if g.axes.size > 0:
                    try:
                        g.set_yticklabels(
                            [f'{tick:.2g}' for tick in g.axes.flat[0].get_yticks()])
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Could not format Y-tick labels for {greek}.")
                plt.tight_layout(rect=[0, 0.03, 1, 0.96])
                self._save_plot(g, f"greek_distribution_{greek}.png")
            except Exception as e:
                logger.error(
                    f"Error generating plot for Greek '{greek}': {e}", exc_info=True)
                plt.close(getattr(g, 'fig', plt.gcf()))

    def plot_benchmark_distribution_crr_lr(self):
        """ Plots Py vs C++ runtime for CRR & LeisenReimer per DTE. Uses 'Set2' palette. """
        if self.combined_runtime_avg_df is None or self.combined_runtime_avg_df.empty:
            logger.warning(
                "Combined runtime data not prepared. Skipping CRR/LR plots.")
            return
        tree_data = self.combined_runtime_avg_df[self.combined_runtime_avg_df['model'].isin(
            ['CRR', 'LeisenReimer'])].copy()
        if tree_data.empty:
            logger.info("No CRR or LeisenReimer runtime data found.")
            return
        dte_order = sorted(tree_data['target_dte_group'].unique())
        if not dte_order:
            logger.warning("No DTE groups found for CRR/LR data.")
            return
        logger.info(
            f"Generating CRR/LR benchmark plots for DTEs: {dte_order} (Auto Y-Scale)")
        n_dte = len(dte_order)
        ncols = 2 if n_dte >= 2 else 1
        nrows = (n_dte + ncols - 1) // ncols
        model_order = ['CRR', 'LeisenReimer']
        for opt_type in ['call', 'put']:
            for source_type in ['Python', 'C++']:
                fig, axes = plt.subplots(nrows, ncols, figsize=(
                    self.figure_size[0]*ncols*0.6, self.figure_size[1]*nrows*0.7), sharey=True, squeeze=False)
                axes_flat = axes.flatten()
                fig.suptitle(
                    f'CRR & Leisen-Reimer {source_type} Avg. Runtime ({opt_type.capitalize()} Options)', fontsize=self.title_fontsize, y=1.02)
                plot_successful = False
                plot_count = 0
                source_data = tree_data[(tree_data['option_type'] == opt_type) & (
                    tree_data['source'] == source_type)]
                if source_data.empty:
                    logger.warning(
                        f"No CRR/LR data for {opt_type}/{source_type}.")
                    plt.close(fig)
                    continue
                is_log_scale = (source_type == 'C++')
                for i, dte in enumerate(dte_order):
                    if i >= len(axes_flat):
                        break
                    ax = axes_flat[i]
                    plot_data = source_data[source_data['target_dte_group'] == dte]
                    if plot_data.empty:
                        ax.set_title(f"{dte} DTE (No Data)")
                        ax.text(0.5, 0.5, 'No Data', ha='center',
                                va='center', transform=ax.transAxes)
                        ax.set_xlabel(
                            "Model", fontsize=self.label_fontsize - 1)
                        ax.set_ylabel("Avg. Calc Time (ms)" if i % ncols ==
                                      0 else "", fontsize=self.label_fontsize - 1)
                        continue
                    try:
                        sns.boxplot(data=plot_data, x='model', y='calc_time_ms', order=model_order,
                                    palette='Set2', showfliers=False, ax=ax, linewidth=1.2)
                        if is_log_scale:
                            ax.set_yscale('log')
                            ax.yaxis.set_major_formatter(
                                mticker.ScalarFormatter())
                            ax.yaxis.set_minor_formatter(
                                mticker.NullFormatter())
                        else:
                            ax.yaxis.set_major_formatter(
                                mticker.FormatStrFormatter('%.2f'))
                        ax.grid(True, which="major", ls="--",
                                linewidth=0.5, axis='y')
                        ax.tick_params(axis='both', which='major',
                                       labelsize=self.tick_fontsize)
                        ax.set_title(f"{dte} DTE Group",
                                     fontsize=self.label_fontsize)
                        ax.set_xlabel(
                            "Model", fontsize=self.label_fontsize - 1)
                        ax.set_ylabel("Avg. Calc Time (ms)" if i % ncols ==
                                      0 else "", fontsize=self.label_fontsize - 1)
                        plot_successful = True
                        plot_count += 1
                    except Exception as e:
                        logger.error(
                            f"Error plotting CRR/LR boxplot for DTE {dte}, Type {opt_type}, Source {source_type}: {e}", exc_info=True)
                        ax.set_title(f"{dte} DTE (Plotting Error)")
                        ax.text(0.5, 0.5, 'Error', ha='center',
                                va='center', transform=ax.transAxes)
                        ax.set_xlabel(
                            "Model", fontsize=self.label_fontsize - 1)
                        ax.set_ylabel("Avg. Calc Time (ms)" if i % ncols ==
                                      0 else "", fontsize=self.label_fontsize - 1)
                for j in range(plot_count, len(axes_flat)):
                    if j < len(axes_flat):
                        fig.delaxes(axes_flat[j])
                if plot_successful:
                    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                    filename = f"benchmark_distribution_crr_lr_{opt_type}_{source_type.lower()}.png"
                    self._save_plot(fig, filename)
                else:
                    logger.warning(
                        f"No plots generated for CRR/LR - {opt_type} - {source_type}.")
                    plt.close(fig)

    def plot_benchmark_distribution_bs(self):
        """ Plots Py vs C++ runtime for BS2002 across DTEs. Uses Set2 color[0]. """
        if self.combined_runtime_avg_df is None or self.combined_runtime_avg_df.empty:
            logger.warning(
                "Combined runtime data not prepared. Skipping BS plot.")
            return
        bs_data_full = self.combined_runtime_avg_df[self.combined_runtime_avg_df['model'] == 'BS2002'].copy(
        )
        if bs_data_full.empty:
            logger.info("No BS2002 runtime data found.")
            return
        dte_order = sorted(bs_data_full['target_dte_group'].unique())
        if not dte_order:
            logger.warning("No DTE groups found for BS2002 data.")
            return
        logger.info(
            f"Generating BS2002 benchmark plots for DTEs: {dte_order} (Auto Y-Scale)")
        try:
            bs_color = sns.color_palette('Set2')[0]
        except IndexError:
            logger.warning("Using fallback green for BS2002.")
            bs_color = 'green'
        for source_type in ['Python', 'C++']:
            fig, axes = plt.subplots(2, 1, figsize=(
                self.figure_size[0]*0.9, self.figure_size[1]*1.1), sharex=True, sharey=True)
            fig.suptitle(
                f'Bjerksund-Stensland (BS2002) {source_type} Avg. Runtime Distribution', fontsize=self.title_fontsize, y=1.0)
            plot_successful = False
            source_data = bs_data_full[bs_data_full['source'] == source_type]
            if source_data.empty:
                logger.warning(f"No BS2002 data for {source_type}. Skipping.")
                plt.close(fig)
                continue
            else:
                logger.info(
                    f"Found {len(source_data)} rows for BS2002 - {source_type}.")
            is_log_scale = (source_type == 'C++')
            for i, opt_type in enumerate(['call', 'put']):
                ax = axes[i]
                plot_data = source_data[source_data['option_type'] == opt_type]
                if plot_data.empty:
                    ax.set_title(f"{opt_type.capitalize()} Options (No Data)")
                    ax.text(0.5, 0.5, 'No Data', ha='center',
                            va='center', transform=ax.transAxes)
                    ax.set_ylabel("Avg. Calc Time (ms)",
                                  fontsize=self.label_fontsize)
                    ax.set_xlabel("Target DTE Group" if i ==
                                  1 else "", fontsize=self.label_fontsize)
                    continue
                try:
                    sns.boxplot(data=plot_data, x='target_dte_group', y='calc_time_ms',
                                order=dte_order, color=bs_color, showfliers=False, ax=ax, linewidth=1.2)
                    if is_log_scale:
                        ax.set_yscale('log')
                        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
                        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
                    else:
                        ax.yaxis.set_major_formatter(
                            mticker.FormatStrFormatter('%.3f'))
                    ax.grid(True, which="major", ls="--",
                            linewidth=0.5, axis='y')
                    ax.tick_params(axis='both', which='major',
                                   labelsize=self.tick_fontsize)
                    ax.set_title(f"{opt_type.capitalize()} Options",
                                 fontsize=self.label_fontsize)
                    ax.set_ylabel("Avg. Calc Time (ms)",
                                  fontsize=self.label_fontsize)
                    ax.set_xlabel("Target DTE Group" if i ==
                                  1 else "", fontsize=self.label_fontsize)
                    plot_successful = True
                except Exception as e:
                    logger.error(
                        f"Error plotting BS benchmark for {opt_type}/{source_type}: {e}", exc_info=True)
                    ax.set_title(f"{opt_type.capitalize()} Options (Error)")
                    ax.text(0.5, 0.5, 'Error', ha='center',
                            va='center', transform=ax.transAxes)
                    ax.set_ylabel("Avg. Calc Time (ms)",
                                  fontsize=self.label_fontsize)
                    ax.set_xlabel("Target DTE Group" if i ==
                                  1 else "", fontsize=self.label_fontsize)
            if plot_successful:
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                filename = f"benchmark_distribution_bs_{source_type.lower()}.png"
                self._save_plot(fig, filename)
            else:
                logger.warning(f"No plots generated for BS - {source_type}.")
                plt.close(fig)

    def plot_benchmark_distribution_bs(self):
        """ Plots Py vs C++ runtime for BS2002 across DTEs. Uses Set2 color[0]. """
        if self.combined_runtime_avg_df is None or self.combined_runtime_avg_df.empty:
            logger.warning(
                "Combined runtime data not prepared. Skipping BS plot.")
            return
        bs_data_full = self.combined_runtime_avg_df[self.combined_runtime_avg_df['model'] == 'BS2002'].copy(
        )

        if bs_data_full.empty:
            logger.info("No BS2002 runtime data found.")
            return

        try:
            dte_order = sorted(
                bs_data_full['target_dte_group'].unique(), key=int)
        except ValueError:
            logger.warning(
                "DTE groups non-integer, using standard sort for BS plot.")
            dte_order = sorted(bs_data_full['target_dte_group'].unique())

        if not dte_order:
            logger.warning("No DTE groups found for BS2002 data.")
            return

        logger.info(
            f"Generating BS2002 benchmark plots for DTEs: {dte_order} (Auto Y-Scale)")

        try:

            bs_color = sns.color_palette('Set2')[2]
        except IndexError:
            logger.warning("Using fallback color for BS2002.")
            bs_color = 'purple'

        for source_type in ['Python', 'C++']:

            fig, axes = plt.subplots(2, 1, figsize=(

                self.figure_size[0]*0.8, self.figure_size[1]*1.0),
                sharex=True, sharey=True)

            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])

            fig.suptitle(
                f'Bjerksund-Stensland (BS2002) {source_type} Avg. Runtime Distribution',
                fontsize=self.title_fontsize, y=1.01)

            plot_successful = False
            source_data = bs_data_full[bs_data_full['source'] == source_type]

            if source_data.empty:
                logger.warning(
                    f"No BS2002 data for {source_type}. Skipping plot.")
                plt.close(fig)
                continue
            else:
                logger.info(
                    f"Plotting {len(source_data)} rows for BS2002 - {source_type}.")

            is_log_scale = (source_type == 'C++')

            for i, opt_type in enumerate(['call', 'put']):
                if i >= len(axes):
                    break
                ax = axes[i]
                plot_data = source_data[source_data['option_type'] == opt_type]

                if plot_data.empty:
                    ax.set_title(f"{opt_type.capitalize()} Options (No Data)")
                    ax.text(0.5, 0.5, 'No Data', ha='center',
                            va='center', transform=ax.transAxes)

                    ax.set_ylabel("Avg. Calc Time (ms)" if i == 0 else "",
                                  fontsize=self.label_fontsize-1)

                    ax.set_xlabel("Target DTE Group" if i ==
                                  1 else "", fontsize=self.label_fontsize - 1)
                    continue

                try:

                    sns.boxplot(data=plot_data, x='target_dte_group', y='calc_time_ms',
                                order=dte_order,
                                color=bs_color,
                                showfliers=False, ax=ax, linewidth=1.2)

                    if is_log_scale:
                        ax.set_yscale('log')

                        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
                        ax.yaxis.set_minor_formatter(
                            mticker.NullFormatter())
                    else:

                        ax.yaxis.set_major_formatter(
                            mticker.FormatStrFormatter('%.4f'))

                    ax.grid(True, which="major", ls="--",
                            linewidth=0.5, axis='y')
                    ax.tick_params(axis='both', which='major',
                                   labelsize=self.tick_fontsize)
                    ax.set_title(f"{opt_type.capitalize()} Options",
                                 fontsize=self.label_fontsize)

                    ax.set_ylabel("Avg. Calc Time (ms)" if i ==
                                  0 else "", fontsize=self.label_fontsize-1)

                    ax.set_xlabel("Target DTE Group" if i ==
                                  1 else "", fontsize=self.label_fontsize - 1)
                    plot_successful = True

                except Exception as e:
                    logger.error(
                        f"Error plotting BS benchmark for {opt_type}/{source_type}: {e}", exc_info=True)
                    ax.set_title(f"{opt_type.capitalize()} Options (Error)")
                    ax.text(0.5, 0.5, 'Error', ha='center',
                            va='center', transform=ax.transAxes)
                    ax.set_ylabel("Avg. Calc Time (ms)" if i ==
                                  0 else "", fontsize=self.label_fontsize-1)
                    ax.set_xlabel("Target DTE Group" if i ==
                                  1 else "", fontsize=self.label_fontsize - 1)

            if plot_successful:

                fig.tight_layout(rect=[0, 0.03, 1, 0.96])
                filename = f"benchmark_distribution_bs_{source_type.lower()}.png"
                self._save_plot(fig, filename)
            else:
                logger.warning(f"No plots generated for BS - {source_type}.")
                plt.close(fig)

    def plot_runtime_speedup_barchart(self):
        """ Generates a bar chart summarizing C++ speedup factor over Python. """
        summary_file = os.path.join(
            self.plots_dir, "runtime_comparison_summary.csv")
        if not os.path.exists(summary_file):
            logger.warning(
                f"Summary file not found: {summary_file}. Gen it first.")
            self._generate_runtime_summary_table()
        if not os.path.exists(summary_file):
            logger.error(
                "Failed generate/find summary file. Skipping speedup chart.")
            return
        try:
            summary_df = pd.read_csv(summary_file)
        except Exception as e:
            logger.error(f"Failed load summary file {summary_file}: {e}")
            return
        try:
            summary_df['Speedup'] = summary_df['Speedup (Py / C++)'].astype(
                str).str.replace('x', '', regex=False)
            summary_df['Speedup'] = pd.to_numeric(
                summary_df['Speedup'], errors='coerce').fillna(0)
            plot_data = summary_df[summary_df['Speedup'] > 0].copy()
        except Exception as e:
            logger.error(
                f"Error preparing summary data for speedup bar chart: {e}")
            return
        if plot_data.empty:
            logger.warning("No valid speedup data to plot.")
            return
        logger.info("Generating runtime speedup bar chart...")
        dte_order = sorted(plot_data['target_dte_group'].unique())
        n_dte = len(dte_order)
        ncols = 2 if n_dte >= 2 else 1
        nrows = (n_dte + ncols - 1) // ncols
        try:
            fig, axes = plt.subplots(nrows, ncols, figsize=(
                self.figure_size[0]*ncols*0.6, self.figure_size[1]*nrows*0.7), sharey=True, squeeze=False)
            axes_flat = axes.flatten()
            fig.suptitle(
                'C++ Speedup Factor over Python (Py Time / C++ Time)', y=1.02)
            plot_successful = False
            plot_count = 0
            model_order = ['CRR', 'LeisenReimer', 'BS2002']
            opt_type_palette = {'call': 'lightgreen', 'put': 'orange'}
            for i, dte in enumerate(dte_order):
                if i >= len(axes_flat):
                    break
                ax = axes_flat[i]
                dte_data = plot_data[plot_data['target_dte_group'] == dte]
                if dte_data.empty:
                    ax.set_title(f"{dte} DTE (No Data)")
                    ax.text(0.5, 0.5, 'No Data', ha='center',
                            va='center', transform=ax.transAxes)
                    ax.set_xlabel("Model", fontsize=self.label_fontsize - 1)
                    ax.set_ylabel("Speedup Factor (x)" if i % ncols ==
                                  0 else "", fontsize=self.label_fontsize - 1)
                    continue
                try:
                    sns.barplot(data=dte_data, x='model', y='Speedup', hue='option_type', order=model_order, hue_order=[

                                'call', 'put'], palette=opt_type_palette, ax=ax, errorbar=None)
                    ax.grid(True, which="major", axis='y',
                            linestyle='--', linewidth=0.5)
                    ax.tick_params(axis='both', which='major',
                                   labelsize=self.tick_fontsize)
                    ax.yaxis.set_major_formatter(
                        mticker.FormatStrFormatter('%.1fx'))
                    ax.set_title(f"{dte} DTE Group",
                                 fontsize=self.label_fontsize)
                    ax.set_xlabel("Model", fontsize=self.label_fontsize - 1)
                    ax.set_ylabel("Speedup Factor" if i % ncols ==
                                  0 else "", fontsize=self.label_fontsize - 1)
                    if i == 0:
                        handles, labels = ax.get_legend_handles_labels()
                        ax.legend(handles, [l.capitalize(
                        ) for l in labels], title="Option Type", fontsize=self.tick_fontsize, loc='best')
                    elif ax.get_legend() is not None:
                        ax.get_legend().remove()
                    plot_successful = True
                    plot_count += 1
                except Exception as e:
                    logger.error(
                        f"Error plotting speedup bar chart for DTE {dte}: {e}", exc_info=True)
                    ax.set_title(f"{dte} DTE (Plotting Error)")
                    ax.text(0.5, 0.5, 'Error', ha='center',
                            va='center', transform=ax.transAxes)
                    ax.set_xlabel("Model", fontsize=self.label_fontsize - 1)
                    ax.set_ylabel("Speedup Factor (x)" if i % ncols ==
                                  0 else "", fontsize=self.label_fontsize - 1)
            for j in range(plot_count, len(axes_flat)):
                if j < len(axes_flat):
                    fig.delaxes(axes_flat[j])
            if plot_successful:
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                self._save_plot(fig, "runtime_speedup_barchart.png")
            else:
                logger.warning("No plots generated for speedup bar chart.")
                plt.close(fig)
        except Exception as e:
            logger.error(
                f"Overall error generating runtime speedup bar chart: {e}", exc_info=True)
            plt.close(plt.gcf())

    def _generate_runtime_summary_table_2(self):
        """ Generates and saves the Python vs C++ summary table. """
        if self.combined_runtime_avg_df is None or self.combined_runtime_avg_df.empty:
            logger.warning(
                "Combined runtime data not available, cannot generate summary table.")

            empty_df = pd.DataFrame(columns=[
                'target_dte_group', 'option_type', 'model',
                'Avg Python Time (ms)', 'Avg C++ Time (ms)',
                'Speedup (Py / C++)', 'Abs Diff (ms)', 'Abs Diff (ns)'
            ])
            compare_filename = os.path.join(
                self.plots_dir, "runtime_comparison_summary.csv")
            try:
                empty_df.to_csv(compare_filename, index=False)
                logger.info(
                    f"Saved empty runtime summary table placeholder to: {compare_filename}")
            except Exception as e:
                logger.error(
                    f"Failed to save empty runtime summary table: {e}")
            return

        logger.info("Calculating Python vs C++ runtime comparison table...")
        try:

            compare_pivot = self.combined_runtime_avg_df.pivot_table(
                index=['target_dte_group', 'option_type', 'model'],
                columns='source',
                values='calc_time_ms'
            ).reset_index()

            if 'Python' not in compare_pivot.columns or 'C++' not in compare_pivot.columns:
                logger.error(
                    "Pivot table missing 'Python' or 'C++' columns after pivoting. Cannot create summary.")
                return

            compare_pivot.dropna(subset=['Python', 'C++'], inplace=True)

            if compare_pivot.empty:
                logger.warning(
                    "No rows with both Python and C++ data found after pivot. Cannot create summary.")
                return

            epsilon_ms = 1e-6
            compare_pivot['Speedup'] = np.where(
                compare_pivot['C++'] > epsilon_ms,
                compare_pivot['Python'] / compare_pivot['C++'],
                np.inf
            )

            compare_pivot['Abs Diff (ms)'] = compare_pivot['Python'] - \
                compare_pivot['C++']

            compare_pivot['Abs Diff (ns)'] = compare_pivot['Abs Diff (ms)'] * 1_000_000

            compare_table = compare_pivot[[
                'target_dte_group', 'option_type', 'model',

                'Python', 'C++', 'Speedup', 'Abs Diff (ms)', 'Abs Diff (ns)'
            ]].copy()
            compare_table.rename(columns={
                'Python': 'Avg Python Time (ms)',
                'C++': 'Avg C++ Time (ms)',
                'Speedup': 'Speedup (Py / C++)'
            }, inplace=True)

            compare_table['Avg Python Time (ms)'] = compare_table['Avg Python Time (ms)'].apply(
                lambda x: f'{x:.4f}' if pd.notna(x) else 'NaN')

            compare_table['Avg C++ Time (ms)'] = compare_table['Avg C++ Time (ms)'].apply(

                lambda x: f'{x:.6f}' if pd.notna(x) and x > 1e-4 else (f'{x:.3e}' if pd.notna(x) else 'NaN'))

            compare_table['Speedup (Py / C++)'] = compare_table['Speedup (Py / C++)'].apply(
                lambda x: f'{x:.1f}x' if pd.notna(x) and np.isfinite(x) else ('inf' if x == np.inf else 'NaN'))

            compare_table['Abs Diff (ms)'] = compare_table['Abs Diff (ms)'].apply(
                lambda x: f'{x:.4f}' if pd.notna(x) else 'NaN')

            compare_table['Abs Diff (ns)'] = compare_table['Abs Diff (ns)'].apply(

                lambda x: f'{x:,.1f}' if pd.notna(x) else 'NaN')

            compare_table.sort_values(
                by=['target_dte_group', 'option_type', 'model'], inplace=True)

            compare_filename = os.path.join(
                self.plots_dir, "runtime_comparison_summary_2.csv")
            compare_table.to_csv(compare_filename, index=False)
            logger.info(
                f"Saved runtime comparison table to: {compare_filename}")

            print("\nRuntime Comparison Summary (Sample):")
            print(compare_table[['target_dte_group', 'option_type', 'model',
                                 'Avg Python Time (ms)', 'Avg C++ Time (ms)',

                                 'Speedup (Py / C++)']].head().to_string(index=False))

        except Exception as e:
            logger.error(
                f"Failed to create runtime comparison table: {e}", exc_info=True)

    def generate_all_plots(self):
        """Generates and saves all defined plots."""
        logger.info("--- Generating All Thesis Plots ---")

        self.plot_equity_price_with_dividends()
        self.create_split_table()
        self.plot_rolling_volatility()
        self.plot_tree_runtime_vs_steps()

        self.plot_benchmark_distribution_crr_lr()
        self.plot_benchmark_distribution_bs()
        self.plot_runtime_speedup_barchart()
        self._generate_runtime_summary_table_2()

        self.plot_pricing_error_magnitude_by_dte()  # Changed from _per_option

        self.plot_greeks_distribution()

        self._generate_runtime_summary_table()

        logger.info("--- Finished Generating Plots ---")

    def _generate_runtime_summary_table(self):
        """ Generates the Python vs C++ summary table, ensuring BS2002 is included. """
        if self.combined_runtime_avg_df is None or self.combined_runtime_avg_df.empty:
            logger.warning(
                "Combined runtime data not available, cannot generate summary table.")
            return
        logger.info(
            "Calculating Python vs C++ runtime comparison table (including BS2002)...")
        try:
            compare_pivot = self.combined_runtime_avg_df.pivot_table(
                index=['target_dte_group', 'option_type', 'model'], columns='source', values='calc_time_ms').reset_index()
            if 'Python' not in compare_pivot.columns or 'C++' not in compare_pivot.columns:
                logger.error("Pivot table missing 'Python' or 'C++' columns.")
                return
            compare_pivot.dropna(subset=['Python', 'C++'], inplace=True)
            if compare_pivot.empty:
                logger.warning("No rows with both Py/C++ data after pivot.")
                compare_filename = os.path.join(
                    self.plots_dir, "runtime_comparison_summary.csv")
                pd.DataFrame(columns=['target_dte_group', 'option_type', 'model', 'Avg Python Time (ms)',
                             'Avg C++ Time (ms)', 'Speedup (Py / C++)']).to_csv(compare_filename, index=False)
                logger.info(f"Saved empty runtime summary table.")
                return
            compare_pivot['Speedup (Py / C++)'] = np.where(compare_pivot['C++']
                                                           > 1e-9, compare_pivot['Python'] / compare_pivot['C++'], np.inf)
            compare_pivot.rename(columns={
                                 'Python': 'Avg Python Time (ms)', 'C++': 'Avg C++ Time (ms)'}, inplace=True)
            compare_table = compare_pivot[['target_dte_group', 'option_type', 'model',
                                           'Avg Python Time (ms)', 'Avg C++ Time (ms)', 'Speedup (Py / C++)']].copy()
            compare_table['Avg Python Time (ms)'] = compare_table['Avg Python Time (ms)'].apply(
                lambda x: f'{x:.4f}' if pd.notna(x) else 'NaN')
            compare_table['Avg C++ Time (ms)'] = compare_table['Avg C++ Time (ms)'].apply(
                lambda x: f'{x:.6f}' if pd.notna(x) else 'NaN')
            compare_table['Speedup (Py / C++)'] = compare_table['Speedup (Py / C++)'].apply(
                lambda x: f'{x:.1f}x' if pd.notna(x) and np.isfinite(x) else ('inf' if x == np.inf else 'NaN'))
            compare_filename = os.path.join(
                self.plots_dir, "runtime_comparison_summary.csv")
            compare_table.sort_values(
                by=['target_dte_group', 'option_type', 'model'], inplace=True)
            compare_table.to_csv(compare_filename, index=False)
            logger.info(
                f"Saved runtime comparison table to: {compare_filename}")
            print("\nRuntime Comparison Summary (Sample):")
            print(compare_table.head().to_string(index=False))
        except Exception as e:
            logger.error(
                f"Failed to create runtime comparison table: {e}", exc_info=True)
