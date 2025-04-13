import os
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Type
from dotenv import load_dotenv, find_dotenv


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


try:
    from src.config.settings import Settings, OptionType, TimeInterval
    from src.models.option_pricing_models import (
        AmericanOptionPricerBase,
        CRRBinomialAmericanPricer,
        LeisenReimerAmericanPricer,
        BjerksundStensland2002Pricer,
        OptionType  # *** Import OptionType from models file ***
    )
    MODELS_IMPORTED = True
    logger.info(
        "Successfully imported pricing models and OptionType from src.models")
except ImportError as e:
    logger.error(f"FATAL: Could not import Settings or required modules: {e}")
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


def run_runtime_benchmark():
    logger.info("--- Starting Python Option Model Runtime Benchmark ---")

    if not MODELS_IMPORTED:
        logger.error(
            "FATAL: Pricing models or OptionType not imported. Exiting.")
        return

    settings = Settings()
    if not hasattr(settings, 'polygon_api_key'):
        logger.warning(
            "API Key missing or invalid in settings (although not required for this script).")

    output_base_dir = settings.output_base_dir
    benchmark_input_dir = os.path.join(
        output_base_dir, settings.benchmark_input_subdir)
    graph_data_dir = os.path.join(output_base_dir, settings.graph_data_subdir)
    os.makedirs(graph_data_dir, exist_ok=True)
    logger.info(
        f"Reading Benchmark input data from: {os.path.abspath(benchmark_input_dir)}")
    logger.info(
        f"Saving Runtime Benchmark results to: {os.path.abspath(graph_data_dir)}")

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

    logger.info(f"Loaded {len(benchmark_input_df)} benchmark input rows.")
    logger.info(f"Number of runs per model: {settings.benchmark_n_runs}")
    logger.info(f"Binomial steps: {settings.benchmark_n_steps_binomial}")

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
            spot_price = float(input_row['S'])
            strike_price = float(input_row['K'])
            time_to_expiry = float(input_row['T'])
            risk_free_rate = float(input_row['r'])
            hist_vol = float(input_row['sigma'])
            div_yield = float(input_row['q'])
            option_type_str = input_row['option_type']
            option_type = OptionType(option_type_str)
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f" Skipping row {index+1} invalid input data: {e}")
            continue
        if pd.isna(time_to_expiry) or pd.isna(hist_vol) or pd.isna(div_yield) or pd.isna(spot_price) or time_to_expiry <= 0 or hist_vol <= 1e-6:
            logger.warning(
                f" Skipping row {index+1} NaN/Zero/Negative inputs.")
            continue

        logger.debug(
            f"  Inputs: S={spot_price:.2f}, K={strike_price:.2f}, T={time_to_expiry:.4f}, r={risk_free_rate:.4f}, sigma={hist_vol:.4f}, q={div_yield:.4f}")

        for model_name, ModelClass in models_to_run.items():
            model_params = {'S': spot_price, 'K': strike_price, 'T': time_to_expiry,
                            'r': risk_free_rate, 'sigma': hist_vol, 'q': div_yield, 'option_type': option_type}
            if model_name in ["CRR", "LeisenReimer"]:
                model_params['N'] = settings.benchmark_n_steps_binomial
            try:
                model_instance = ModelClass(**model_params)
                run_times = []
                first_result_dict = {}  # Store price + greeks
                for run_num in range(settings.benchmark_n_runs):
                    result_dict = model_instance.calculate_all()
                    calc_time = result_dict.get('calc_time_sec', np.nan)
                    run_times.append(calc_time)
                    if run_num == 0:
                        first_result_dict = result_dict  # Capture first result
                avg_calc_time = np.nanmean(run_times) if run_times else np.nan

                for run_num in range(settings.benchmark_n_runs):

                    current_run_time = run_times[run_num] if run_num < len(
                        run_times) else np.nan

                    model_price = first_result_dict.get('price', np.nan)
                    delta = first_result_dict.get('delta', np.nan)
                    gamma = first_result_dict.get('gamma', np.nan)
                    vega = first_result_dict.get('vega', np.nan)
                    theta = first_result_dict.get('theta', np.nan)
                    rho = first_result_dict.get('rho', np.nan)

                    benchmark_results.append({
                        "underlying_ticker": input_row['underlying_ticker'],
                        "option_ticker": input_row['option_ticker'],
                        "target_dte_group": input_row['target_dte_group'],
                        "option_type": option_type_str,

                        "timestamp": input_row['timestamp'],
                        "model": model_name,
                        "binomial_steps": model_params.get('N'),
                        "calc_time_sec": current_run_time,
                        "run_number": run_num + 1,

                        "calculated_price": model_price,

                        "delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho,
                        "input_S": spot_price, "input_K": strike_price, "input_T": time_to_expiry,
                        "input_r": risk_free_rate, "input_sigma": hist_vol, "input_q": div_yield,
                    })
                logger.debug(
                    f"    {model_name}: Avg time = {avg_calc_time * 1000:.4f} ms")

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

        output_cols = ["underlying_ticker", "option_ticker", "target_dte_group", "option_type", "timestamp", "model", "binomial_steps", "run_number", "calc_time_sec",
                       "calculated_price", "delta", "gamma", "vega", "theta", "rho",
                       "input_S", "input_K", "input_T", "input_r", "input_sigma", "input_q"]

        results_df = results_df[[
            col for col in output_cols if col in results_df.columns]]
        output_filename = os.path.join(
            graph_data_dir, "py_runtime_benchmark_results.csv")
        results_df.to_csv(output_filename, index=False)
        logger.info(
            f"Saved Python runtime benchmark results ({len(results_df)} rows) to: {output_filename}")
    else:
        logger.warning("No benchmark results generated.")


if __name__ == "__main__":
    run_runtime_benchmark()
