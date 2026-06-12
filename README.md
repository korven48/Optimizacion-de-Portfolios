# Posit Portfolio Optimization Library

This repository contains a portfolio optimization implementation using Posit arithmetic, designed to compare its precision and stability against standard floating-point arithmetic (IEEE 754). It uses `skfolio` as the base for portfolio optimization and a C++ extension for Posit arithmetic.

## Project Structure

- `posit_lib/`: Main Python library.
- `cpp_extension/`: C++ source code for Posit arithmetic and Python wrappers (pybind11).
- `tests/`: Test and comparison scripts (e.g., `ill_conditioned_comparison.py`).
- `examples/`: Usage examples.

## Installation and Usage Guide

Follow these steps to set up the environment and run the experiments.

### Prerequisites

Make sure you have the following installed on your system:

*   **Python 3.8+**
*   **C++ compiler compatible with C++20** (GCC 10+, Clang 10+, MSVC 19.28+)
*   **CMake 3.15+**
*   **Universal Number Library**: The extension assumes the Universal library is available or its headers can be included. By default it looks in `/usr/local/include`.

### 1. Set Up Virtual Environment

It is recommended to use a virtual environment to isolate dependencies.

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

### 2. Install Python Dependencies

Install the required libraries listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Build the C++ Extension (Posit Wrapper)

To use Posit arithmetic, you need to build the native extension. A `build.sh` script is provided to simplify this process.

```bash
# Grant execution permissions to the script if needed
chmod +x build.sh

# Run the build script
./build.sh
```

This script will:
1.  Clean previous builds.
2.  Configure CMake looking for the Universal library in `/usr/local/include` (you can edit `cpp_extension/CMakeLists.txt` if your path is different).
3.  Compile the `posit` module optimized for your architecture.
4.  Install the resulting `.so` file in `posit_lib/`.

### 4. Run Tests and Comparisons

The `tests/` directory contains several scripts to evaluate and compare the numerical precision of different arithmetic formats in portfolio optimization.

#### 4.1 Comparison Engine (`tests/custom_comparison.py`)

This is the core module that runs the comparisons. It provides the `run_comparison()` function which:

1.  Optimizes a portfolio with **Skfolio** (Float64, reference).
2.  Repeats the optimization with the custom solver using each combination of number type and scaling strategy.
3.  Computes quality metrics: L2 Error, risk gap, weight sum, negativity, etc.

**Direct usage** (with example synthetic data):

```bash
python3 tests/custom_comparison.py
```

**Programmatic usage** (importing the function):

```python
from tests.custom_comparison import run_comparison

df = run_comparison(
    X,                                # Returns matrix (n_samples x n_assets)
    asset_names=["Asset1", ...],      # Asset names (optional)
    scaling_strategies=[('std', 1.0)],# Scaling strategies
    number_types=None,                # None = all available types
    solver_params={'tolerance': 1e-6},# Solver parameters
    scale_to_golden_zone=False,       # Scale to Posit golden zone
    export_csv="results.csv",         # Export to CSV (optional)
    print_console=True                # Display results in console
)
```

**Supported number types:**

| IEEE 754 Family        | Posit Family             |
|------------------------|--------------------------|
| Float8_e4m3fn          | Posit8                   |
| Float8_e5m2            | Posit12                  |
| Float16                | Posit16                  |
| BFloat16               | Posit20                  |
| Float32                | Posit24                  |
| Float64                | Posit32                  |
|                        | Posit64                  |

**Available scaling strategies:** `none`, `manual`, `max`, `std`, `frobenius`, `pow2`.

---

#### 4.2 Real Asset Comparison (`tests/real_asset_comparison.py`)

Downloads historical data from Yahoo Finance for a diversified portfolio of 10 real assets and runs the comparison.

**Included assets:** Gold (GLD), Bitcoin (BTC-USD), S&P 500 (SPY), Nasdaq (QQQ), Treasury Bonds (TLT), Real Estate (VNQ), Emerging Markets (EEM), Oil (USO), Corporate Bonds (LQD), US Dollar (UUP).

```bash
python3 tests/real_asset_comparison.py
```

> **Note:** Requires internet connection to download data via `yfinance`. Default data spans from 2018-01-01 to 2026-01-01 with monthly frequency.

---

#### 4.3 Ill-Conditioned Matrix Comparison (`tests/ill_conditioned_comparison.py`)

Generates synthetic data designed to stress numerical stability (matrices with high correlation and tiny numbers) and compares the performance of different number types.

```bash
python3 tests/ill_conditioned_comparison.py
```

---

#### 4.4 Full Grid Search (`tests/grid_search.py`)

Runs an exhaustive hyperparameter search, combining:

*   **4 datasets:** Synthetic high correlation, tiny numbers, real monthly data, and real daily data.
*   **6 tolerances:** `1e-3`, `1e-4`, `1e-5`, `1e-6`, `1e-7`, `1e-8`.
*   **6 scaling strategies:** `none`, `manual(100)`, `max`, `std`, `frobenius`, `pow2`.
*   **2 Golden Zone options:** `True` / `False`.
*   **13 number types:** 6 IEEE 754 + 7 Posit.

```bash
python3 tests/grid_search.py
```

Results are saved to `tests/full_grid_search_results.csv`.

---

#### 4.5 Grid Search Results (`tests/full_grid_search_resultados.csv`)

CSV file with the complete grid search results (~3400 rows). Columns:

| Column                 | Description                                                     |
|------------------------|-----------------------------------------------------------------|
| `Dataset`              | Name of the dataset used                                        |
| `Tolerance`            | Solver convergence tolerance                                    |
| `Golden_Zone`          | Whether scaling to the Posit golden zone was applied            |
| `Scaling_Strategy`     | Scaling strategy applied                                        |
| `Scaling_Factor`       | Multiplicative factor of the strategy                           |
| `Number_Type`          | Number type used (e.g., `Posit16`, `Float32`)                   |
| `Time_s`               | Execution time in seconds                                       |
| `Iterations`           | Number of solver iterations                                     |
| `Error_L2`             | L2 norm of the weight difference vs. Skfolio                    |
| `Risk_Variance`        | Portfolio variance (w^T * Cov * w)                              |
| `Sum_Weights`          | Sum of weights (should be ~1.0)                                 |
| `Negativity_Violation` | Sum of non-negativity violations                                |
| `Max_Abs_Diff`         | Maximum absolute weight difference vs. Skfolio                  |
| `Risk_Gap_Pct`         | Percentage risk gap relative to Skfolio                         |
| `Grad_Zero_Detected`   | Whether the gradient collapsed to zero (underflow)              |
| `Weights_Array`        | Resulting portfolio weight vector                               |