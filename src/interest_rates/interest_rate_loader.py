import pandas as pd
import numpy as np
import logging
from datetime import datetime, date, timezone
from typing import Optional
import os

logger = logging.getLogger(__name__)


class InterestRateLoader:
    """
    Loads and provides interpolated risk-free interest rates based on
    historical Treasury yield data.
    """

    def __init__(self):
        """
        Initializes the loader and loads the rate data.

        Args:
            file_path (str): The path to the CSV file containing interest rate data.
                             Expected columns: 'observation_date', 'DGS1MO', 'DGS3MO',
                                               'DGS1', 'DGS10'. Rates assumed to be percentages.
        """
        base_path = os.path.dirname(os.path.abspath(
            __file__))
        self.data_source = os.path.join(
            base_path, "combined_interest_rates.csv")

        self.maturities_years = np.array([
            1/12,  # DGS1MO
            3/12,  # DGS3MO
            1.0,   # DGS1
            10.0   # DGS10
        ])

        self.rate_columns = ['DGS1MO', 'DGS3MO', 'DGS1', 'DGS10']

        self.load_rates()

    def load_rates(self):
        """Loads and preprocesses the interest rate data from the CSV file."""
        logger.info(
            f"Attempting to load interest rates from: {self.data_source}")
        try:

            if not os.path.exists(self.data_source):
                raise FileNotFoundError(
                    f"Interest rate file not found at {self.data_source}")

            df = pd.read_csv(self.data_source)
            print(df)
            logger.info(
                f"Successfully read {len(df)} rows from {self.data_source}.")

            required_cols = ['observation_date'] + self.rate_columns
            if not all(col in df.columns for col in required_cols):
                missing = [
                    col for col in required_cols if col not in df.columns]
                raise ValueError(
                    f"Missing required columns in rate file: {missing}")

            df['observation_date'] = pd.to_datetime(
                df['observation_date'], errors='coerce')

            df.dropna(subset=['observation_date'], inplace=True)
            df.set_index('observation_date', inplace=True)

            for col in self.rate_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if not df[col].isnull().all() and df[col].max() > 1.5:
                    logger.info(
                        f"Converting column '{col}' from percentage to decimal.")
                    df[col] = df[col] / 100.0
                elif df[col].isnull().all():
                    logger.warning(
                        f"Rate column '{col}' contains only NaN values.")

            if df.index.tz is None:
                df = df.tz_localize('UTC')
            else:

                df = df.tz_convert('UTC')

            df.sort_index(inplace=True)

            self.rates_df = df[self.rate_columns].copy()
            logger.info(
                f"Interest rate data loaded and preprocessed. Index range: {self.rates_df.index.min()} to {self.rates_df.index.max()}")

        except FileNotFoundError as e:
            logger.error(f"Error loading interest rates: {e}")
            self.rates_df = None
        except ValueError as e:
            logger.error(f"Error processing interest rate data: {e}")
            self.rates_df = None
        except Exception as e:
            logger.error(
                f"Unexpected error loading interest rates: {e}", exc_info=True)
            self.rates_df = None

    def get_rate(self, calc_date_aware: datetime, time_to_expiry_years: float) -> Optional[float]:
        """
        Gets the interpolated risk-free rate for a given calculation date and time to expiry.
        Uses forward fill to find the rate on or before the calculation date.
        """
        if self.rates_df is None or self.rates_df.empty:
            logger.warning("Interest rate data not loaded. Cannot get rate.")
            return None
        if pd.isna(time_to_expiry_years) or time_to_expiry_years < 0:
            logger.warning(f"Invalid TTE provided: {time_to_expiry_years}")
            return None

        if calc_date_aware.tzinfo is None:
            calc_date_aware = calc_date_aware.replace(tzinfo=timezone.utc)
        elif calc_date_aware.tzinfo != timezone.utc:
            calc_date_aware = calc_date_aware.astimezone(timezone.utc)

        try:
            indexer = self.rates_df.index.get_indexer(
                [calc_date_aware], method='ffill')[0]

            if indexer < 0:
                lookup_date_str = self.rates_df.index.min().strftime(
                    '%Y-%m-%d') if not self.rates_df.empty else "N/A"
                logger.warning(
                    f"Calc date {calc_date_aware.date()} before first rate date ({lookup_date_str}). Cannot get rate.")
                return None

            rates_on_date = self.rates_df.iloc[indexer]
            found_date_str = self.rates_df.index[indexer].strftime(
                '%Y-%m-%d')

            if rates_on_date.isnull().any():
                logger.warning(
                    f"Rate data for lookup date {found_date_str} contains NaN: {rates_on_date.to_dict()}. Cannot interpolate.")
                return None

            available_rates = rates_on_date.values

            # Interpolate/Extrapolate
            if time_to_expiry_years <= self.maturities_years[0]:
                interpolated_rate = available_rates[0]
            elif time_to_expiry_years >= self.maturities_years[-1]:
                interpolated_rate = available_rates[-1]
            else:
                interpolated_rate = np.interp(
                    time_to_expiry_years, self.maturities_years, available_rates)

            if pd.isna(interpolated_rate):
                logger.error(
                    f"Interpolation resulted in NaN for T={time_to_expiry_years:.4f} on lookup date {found_date_str}.")
                return None

            logger.debug(
                f"Rate lookup for {calc_date_aware.date()} T={time_to_expiry_years:.4f}: Used data from {found_date_str}. Interpolated rate: {interpolated_rate:.6f}")

            return float(max(0.0, interpolated_rate))

        except IndexError:
            logger.warning(
                f"Index error during rate lookup for date {calc_date_aware.date()}.")
            return None
        except Exception as e:
            logger.error(
                f"Error getting interest rate for {calc_date_aware.date()} T={time_to_expiry_years:.4f}: {e}", exc_info=True)
            return None
