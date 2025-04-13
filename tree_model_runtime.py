import os
import pandas as pd
import numpy as np
import logging
import time
from typing import List, Dict, Any, Type
from dotenv import load_dotenv, find_dotenv


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


try:
    from src.config.settings import Settings
    from src.models.option_pricing_models import (
        AmericanOptionPricerBase,
        CRRBinomialAmericanPricer,
        LeisenReimerAmericanPricer,
        OptionType
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

dotenv_path = find_dotenv()
if dotenv_path:
    logger.info(f"Loading .env: {dotenv_path}")
    load_dotenv(dotenv_path)
else:
    logger.warning(".env not found.")


def run_tree_step_benchmark():
    logger.info("--- Starting Tree Model Runtime vs Steps Benchmark ---")

    if not MODELS_IMPORTED:
        logger.error("FATAL: Models not imported. Exiting.")
        return

    settings = Settings()
    if not hasattr(settings, 'polygon_api_key') or not settings.polygon_api_key or settings.polygon_api_key == "YOUR_DEFAULT_KEY_HERE":
        logger.error(
            "FATAL: API Key missing (though not strictly needed for calc, good practice to check settings load).")
        return

    output_base_dir = settings.output_base_dir
    graph_data_dir = os.path.join(
        output_base_dir, settings.graph_data_subdir)
    os.makedirs(graph_data_dir, exist_ok=True)
    logger.info(
        f"Saving benchmark results to: {os.path.abspath(graph_data_dir)}")

    # NOTE: Sample parameter definition -> Just for benchmarkign!
    S0 = 100.0
    K_val = 100.0
    sigma_val = 0.25
    r_val = settings.benchmark_risk_free_rate
    q_val = 0.02
    dte_targets_days = settings.dte_targets

    T_values = {dte: max(dte / 365.25, 1e-9) for dte in dte_targets_days}

    N_steps_list = settings.benchmark_tree_steps_list

    tree_models_to_run: Dict[str, Type[AmericanOptionPricerBase]] = {
        "CRR": CRRBinomialAmericanPricer,
        "LeisenReimer": LeisenReimerAmericanPricer
    }

    benchmark_results = []
    n_runs = settings.benchmark_n_runs  # Number of times to average runtime

    logger.info(f"Benchmarking with N steps: {N_steps_list}")
    logger.info(f"Number of runs per N for averaging: {n_runs}")

    # Main iterative
    for target_dte, T_val_years in T_values.items():
        for option_type in [OptionType.CALL, OptionType.PUT]:
            logger.info(
                f"\nProcessing Group: DTE={target_dte}, Type={option_type.name}, T={T_val_years:.4f}")

            for N_step_val in N_steps_list:
                # Ensure N is valid for the models
                N_actual_crr = N_step_val
                N_actual_lr = N_step_val if N_step_val % 2 != 0 else N_step_val + 1  # LR needs odd N

                for model_name, ModelClass in tree_models_to_run.items():
                    current_n = N_actual_lr if model_name == "LeisenReimer" else N_actual_crr
                    logger.debug(
                        f"  Running {model_name} with N={current_n}...")

                    model_params = {
                        'S': S0, 'K': K_val, 'T': T_val_years, 'r': r_val,
                        'sigma': sigma_val, 'q': q_val, 'option_type': option_type,
                        'N': current_n  # Pass the correct N steps
                    }

                    try:
                        model_instance = ModelClass(**model_params)
                        run_times = []
                        for _ in range(n_runs):

                            start_t = time.perf_counter()
                            model_instance._price_impl()

                            end_t = time.perf_counter()
                            run_times.append(end_t - start_t)

                        avg_calc_time = sum(
                            run_times) / len(run_times) if run_times else np.nan
                        logger.debug(
                            f"    Avg time for {model_name}(N={current_n}): {avg_calc_time * 1000:.4f} ms")

                        benchmark_results.append({
                            "target_dte_group": target_dte,
                            "option_type": option_type.value,
                            "model": model_name,
                            "N_steps": current_n,  # Actual N used in the current run!
                            "avg_calc_time_sec": avg_calc_time,

                            "S": S0, "K": K_val, "T": T_val_years, "r": r_val, "sigma": sigma_val, "q": q_val
                        })

                    except Exception as e:
                        logger.error(
                            f"   ERROR running {model_name} with N={current_n}: {e}")
                        # Append NaN result to indicate failure -> Later on filtered in graph
                        benchmark_results.append({
                            "target_dte_group": target_dte, "option_type": option_type.value,
                            "model": model_name, "N_steps": current_n,
                            "avg_calc_time_sec": np.nan,
                            "S": S0, "K": K_val, "T": T_val_years, "r": r_val, "sigma": sigma_val, "q": q_val
                        })

    logger.info(f"\nTree model benchmarking finished.")
    if benchmark_results:
        results_df = pd.DataFrame(benchmark_results)
        output_filename = os.path.join(
            graph_data_dir, "tree_model_runtime_vs_steps.csv")
        results_df.sort_values(
            by=['target_dte_group', 'option_type', 'model', 'N_steps'], inplace=True)
        results_df.to_csv(output_filename, index=False)
        logger.info(
            f"Saved tree benchmark results ({len(results_df)} rows) to: {output_filename}")
        print("\nSample Results:")
        print(results_df.head())
    else:
        logger.warning("No tree benchmark results generated.")


if __name__ == "__main__":
    run_tree_step_benchmark()
