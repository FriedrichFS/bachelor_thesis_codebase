# Comparative Analysis of American Option Pricing Models Codebase

**Author:** Friedrich Leonard Rösel
**Degree:** Bachelor of Science (B.Sc.)
**Institution:** Frankfurt School of Finance & Management
**Date:** April 2025

---

## Overview

This repository contains the Python and C++ code developed for the Bachelor Thesis titled "Comparative Analysis of American Option Pricing: Evaluating Arbitrage Opportunities and Runtime Complexities of Analytical and Lattice Models".

The project empirically compares three prominent models for pricing American options:

1.  **Bjerksund-Stensland (2002) Analytical Approximation (BS2002)**
2.  **Cox-Ross-Rubinstein (CRR) Binomial Lattice Model**
3.  **Leisen-Reimer (LR) Binomial Lattice Model**

The analysis focuses on:

-   **Computational Performance:** Runtime comparisons between models and implementations (Python vs. C++).
-   **Pricing Accuracy:** Evaluation against historical market data and model-to-model consistency.
-   **Numerical Stability:** Assessment of Greek calculations (Delta, Gamma, Vega, Theta, Rho), particularly Gamma stability.

The system uses historical options and equity data for selected S&P 500 constituents, incorporates realistic discrete dividend adjustments, and utilizes dynamic term-matched Treasury yields for risk-free rates.

## Features

-   Implementations of CRR, LR, and BS2002 models in both Python and C++.
-   Automated data pipeline using Polygon.io API for sourcing equity, option, dividend, and split data.
-   Volatility-based selection of underlying S&P 500 tickers for diverse analysis.
-   Multi-stage option filtering based on DTE, type, activity, and data completeness.
-   Calculation of benchmark inputs considering discrete dividends and dynamically interpolated risk-free rates.
-   Systematic benchmarking framework for runtime, pricing accuracy, and Greek calculations.
-   Generation of tables and figures used in the thesis results chapter.

## Prerequisites

Before running the code, ensure you have the following installed:

1.  **Python:** Version 3.8 or higher is recommended.
2.  **Pip:** Python package installer.
3.  **Python Libraries:** Install required libraries using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
4.  **C++ Compiler:** A modern C++ compiler that supports C++11 or later (e.g., g++, Clang). This is required to compile and run the C++ benchmarks (`benchmark_cpp`).
5.  **Polygon.io API Key:** You need an active Polygon.io subscription with access to Stocks and Options data. Ensure your API key has sufficient allowance to avoid rate limits during extensive data fetching (e.g., "Developer" or higher tiers might be necessary, especially for the initial pipeline run).

## Configuration

1.  **API Key Setup:**

    -   Create a file named `.env` in the root directory of the project.
    -   Add your Polygon.io API key to this file:
        ```dotenv
        POLYGON_API_KEY=YOUR_ACTUAL_POLYGON_API_KEY
        ```
        Replace `YOUR_ACTUAL_POLYGON_API_KEY` with your key.

2.  **Pipeline & Analysis Settings:**
    -   The primary configuration file is `src/config/settings.py`.
    -   This file controls various parameters, including:
        -   `output_base_dir`: The main directory for all generated outputs.
        -   `start_date`, `end_date`: The analysis period for historical data.
        -   `vola_select_n_tickers`, `vola_rolling_window_days`: Parameters for volatility analysis and ticker selection.
        -   `dte_targets`, `option_types`: Target Days-to-Expiration groups and option types (Call/Put) to analyze.
        -   `selector_top_n_candidates`, `analysis_min_volume`, `option_completeness_threshold`: Filters for selecting candidate options.
        -   `benchmark_n_steps_binomial`: The number of steps (N) used for CRR/LR in primary accuracy/Greek comparisons (default: 101).
        -   `benchmark_n_runs`: Number of iterations for runtime measurements.
        -   `benchmark_tree_steps_list`: List of N values used for the specific tree model runtime scaling analysis.
        -   `benchmark_risk_free_rate`: Default risk-free rate if dynamic interpolation fails.
    -   Review and adjust these settings in `src/config/settings.py` before running the pipeline if needed. The defaults align with the thesis methodology.

## Usage: Execution Flow

The analysis is performed through a sequence of scripts. Run them from the project's root directory in the following order:

1.  **Run Data Pipeline (`run_pipeline.py`):**

    -   **Purpose:** Acquires all necessary data (S&P 500 tickers, equity OHLCV, option contracts metadata, option OHLCV, dividends, splits, interest rates), performs volatility analysis, selects underlying tickers, filters options based on DTE/type/activity/completeness, calculates benchmark model inputs (adjusting spot price for discrete dividends, using dynamic rates), and saves raw data and the final benchmark input file.
    -   **Command:**
        ```bash
        python3 run_pipeline.py
        ```
    -   **Key Output:** `pipeline_output_final/benchmark_input_data/benchmark_input_data_discrete_div.csv` (used by subsequent steps), plus various raw data files in `pipeline_output_final/raw_data/`. _Note: This step can take significant time due to extensive data fetching and processing._

2.  **Run Python Model Benchmarking (`run_model_benchmarking.py`):**

    -   **Purpose:** Loads the benchmark input data, runs the Python implementations of BS2002, CRR (N=101), and LR (N=101), calculates option prices and Greeks for each input row, measures execution time over multiple runs, and saves the results.
    -   **Command:**
        ```bash
        python3 run_model_benchmarking.py
        ```
    -   **Key Output:** `py_results_bench-marking.csv` (or similar name, check script output) containing detailed results from Python models.

3.  **Compile and Run C++ Model Benchmarking (`benchmark_cpp`):**

    -   **Purpose:** Performs the same calculations as the Python benchmarking script but using the optimized C++ implementations for performance comparison.
    -   **Compilation (Example using g++):**
        ```bash
        g++ -std=c++11 -O3 -o benchmark_cpp src/benchmarking/benchmark_models.cpp src/models/bjerksund_stensland_2002.cpp src/models/leisen_reimer.cpp src/models/cox_ross_rubinstein.cpp src/models/american_option_pricer_base.cpp -I./src -lm
        # Adjust include paths (-I) and source files as necessary based on your project structure
        ```
    -   **Execution:**
        ```bash
        ./benchmark_cpp
        ```
    -   **Key Output:** A CSV file (e.g., `cpp_results_benchmarking.csv` - check C++ code/output for exact name) containing results from C++ models.

4.  **Run Mispricing Analysis (`run_mispricing_benchmarking.py`):**

    -   **Purpose:** Loads the Python benchmark results, fetches corresponding historical market prices for the options, calculates pricing errors (Model Price - Market Price), merges Python and C++ results, and performs statistical comparisons between model prices.
    -   **Command:**
        ```bash
        python3 run_mispricing_benchmarking.py
        ```
    -   **Key Output:** Files containing pricing errors and statistical test results (check script for specific filenames).

5.  **Run Tree Model Runtime Scaling Analysis (`tree_model_runtime.py`):**

    -   **Purpose:** Conducts a specific benchmark focusing only on the CRR and LR C++ models, measuring their runtime across a wider range of step counts (N) defined in `settings.py` (`benchmark_tree_steps_list`).
    -   **Command:**
        ```bash
        python3 tree_model_runtime.py
        ```
    -   **Key Output:** A CSV file containing runtime data for CRR vs LR across different N values.

6.  **Generate Charts and Tables (`run_charts.py`):**
    -   **Purpose:** Consolidates results from the previous steps and generates the figures (plots) and statistical tables presented in Chapter 6 of the thesis (e.g., pricing error distributions, runtime comparisons, Greek distributions, model comparison statistics).
    -   **Command:**
        ```bash
        python3 run_charts.py
        ```
    -   **Key Output:** Image files (e.g., .png) and potentially CSV/text files containing formatted tables saved in the `graph_data_dir` specified in `settings.py`.

## Output Description

The primary output directory is defined by `output_base_dir` in `settings.py` (default: `pipeline_output_final`). Key outputs include:

-   **`raw_data/`**: Contains intermediate and source data files (metadata, OHLCV, events).
-   **`benchmark_input_data/`**: Holds the crucial `benchmark_input_data_discrete_div.csv` file used for model execution.
-   **`graph_data/`**: Contains generated figures and tables for the thesis results.
-   **Benchmark Results Files**: CSV files generated by `run_model_benchmarking.py`, `./benchmark_cpp`, `run_mispricing_benchmarking.py`, and `tree_model_runtime.py` containing detailed price, Greek, runtime, and error calculations.

## Code Implementation Details

-   **Languages:** Python 3 and C++ (C++11 standard or later).
-   **Core Libraries (Python):** Pandas, NumPy, SciPy, Pydantic, Requests, Matplotlib, Seaborn, python-dotenv.
-   **Base Class:** A common `AmericanOptionPricerBase` structure is used in both Python and C++ to ensure a consistent interface for pricing, Greek calculation, and benchmarking.
-   **Python Snippets:** Key Python model implementation snippets are included in Appendix A of the thesis document. The full code is available in the `src/` directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Friedrich Leonard Rösel - [https://www.linkedin.com/in/friedrich-leonard-rösel-b920991b9/]
