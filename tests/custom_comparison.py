#!/usr/bin/env python3
"""
Custom Data Comparison Script
Allows users to run the Skfolio vs Posit/Float comparison with their own data.
"""

import sys
import os
import shutil
import time
import numpy as np
import pandas as pd

# Add parent directory to path to find posit_lib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from posit_lib.adapters.skfolio_adapter import PositMeanVariance
from posit_lib.float_wrapper import FloatWrapper, Float16Wrapper, Float32Wrapper
from posit_lib import posit

from skfolio.optimization import MeanRisk, ObjectiveFunction

def get_terminal_width():
    return shutil.get_terminal_size((100, 20)).columns

def print_header(text):
    width = get_terminal_width()
    print("=" * width)
    print(text.center(width))
    print("=" * width)

def print_separator():
    print("-" * get_terminal_width())

def run_comparison(X, asset_names=None, scaling_strategies=None, number_types=None, solver_params=None):
    """
    Runs the portfolio optimization comparison with provided data.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_assets)
        Matrix of asset returns.
    asset_names : list of str, optional
        Names of the assets. Defaults to ["Asset 0", "Asset 1", ...].
    scaling_strategies : list of tuples, optional
        List of (type, factor). Defaults to standard comparison set.
    number_types : list of tuples, optional
        List of (name, class). Defaults to standard numeric types.
    solver_params : dict, optional
        Additional parameters for the solver (e.g. {'learning_rate': 0.1}).
    """
    
    # 1. Setup Defaults
    X = np.array(X)
    n_samples, n_assets = X.shape

    if asset_names is None:
        asset_names = [f"Asset {i}" for i in range(n_assets)]
    
    if len(asset_names) != n_assets:
        print(f"Warning: asset_names length ({len(asset_names)}) does not match n_assets ({n_assets}). Truncating or padding.")
        asset_names = asset_names[:n_assets]
        while len(asset_names) < n_assets:
            asset_names.append(f"Asset {len(asset_names)}")

    if scaling_strategies is None:
        scaling_strategies = [
            ('none', 1.0),
            ('manual', 100.0),
            ('auto_max_abs', 1.0),
            ('pow2', 1.0)
        ]

    if number_types is None:
        number_types = [
            ("Float16", Float16Wrapper),
            ("Float32", Float32Wrapper),
            ("Float64", FloatWrapper),
            ("Posit8", posit.Posit8),
            ("Posit16", posit.Posit16),
            ("Posit32", posit.Posit32),
            ("Posit64", posit.Posit64)
        ]
        
    if solver_params is None:
        solver_params = {}

    print_header("CUSTOM DATA COMPARISON: SKFOLIO vs POSIT vs FLOAT")
    print(f"Data Shape: {n_samples} samples x {n_assets} assets")
    print_separator()

    # 2. Run Skfolio Baseline
    print("\n[Skfolio Official] MeanRisk(objective_function='MINIMIZE_RISK')...")
    start = time.time()
    try:
        model_sk = MeanRisk(risk_aversion=1.0, objective_function=ObjectiveFunction.MINIMIZE_RISK) 
        model_sk.fit(X)
        weights_sk = model_sk.weights_
        time_sk = time.time() - start
        print(f"  Time: {time_sk:.4f}s")
    except Exception as e:
        print(f"  Skfolio Failed: {e}")
        weights_sk = np.zeros(n_assets)
        time_sk = 0.0

    # 3. Run Strategies
    for scale_type, scale_factor in scaling_strategies:
        print("\n" + "=" * 100)
        print(f"SCALING STRATEGY: {scale_type, scale_factor}")
        print("=" * 100)
        
        results = {}

        for name, number_type in number_types:
            start = time.time()
            try:
                model = PositMeanVariance(
                    risk_aversion=1.0, 
                    objective_function='MINIMIZE_RISK', 
                    number_type=number_type,
                    scaling_type=scale_type,
                    scaling_factor=scale_factor,
                    **solver_params
                )
                model.fit(X)
                
                weights = model.weights_
                iterations = getattr(model, 'n_iter_', 'N/A')
                elapsed = time.time() - start
                
                results[name] = {
                    'weights': weights,
                    'time': elapsed,
                    'iters': iterations
                }
            except Exception as e:
                # print(f"  FAILED {name}: {e}")
                results[name] = None

        # 4. Display Results Table
        print("\n")
        print_header(f"RESULTS {scale_type, scale_factor}")

        headers = ["Asset", "Skfolio"] + [name for name, _ in number_types]
        col_width = 12
        asset_width = 15

        row_fmt = f"| {{:<{asset_width}}} | {{:>{col_width}}}" + f" | {{:>{col_width}}}" * len(number_types) + " |"
        sample_line = row_fmt.format(*headers)
        separator = "-" * len(sample_line)

        print(separator)
        print(row_fmt.format(*headers))
        print(separator)

        for i in range(n_assets):
            row_vals = [asset_names[i], f"{weights_sk[i]:.6f}"]
            for name, _ in number_types:
                res = results.get(name)
                if res and res['weights'] is not None:
                    row_vals.append(f"{res['weights'][i]:.6f}")
                else:
                    row_vals.append("ERROR")
            print(row_fmt.format(*row_vals))

        print(separator)

        # Global Metrics
        print("\nPERFORMANCE METRICS:")
        print(separator)
        metric_fmt = f"| {{:<{asset_width}}} | {{:>{col_width}}}" + f" | {{:>{col_width}}}" * len(number_types) + " |"

        # Time
        time_vals = ["Time (s)", f"{time_sk:.4f}"]
        for name, _ in number_types:
            res = results.get(name)
            time_vals.append(f"{res['time']:.4f}" if res else "-")
        print(metric_fmt.format(*time_vals))

        # Iterations
        iter_vals = ["Iterations", "N/A"]
        for name, _ in number_types:
            res = results.get(name)
            iter_vals.append(f"{res['iters']}" if res else "-")
        print(metric_fmt.format(*iter_vals))

        # L2 Error
        l2_vals = ["L2 Error (vs Sk)", "0.00e+00"]
        for name, _ in number_types:
            res = results.get(name)
            if res and res['weights'] is not None:
                diff = np.linalg.norm(weights_sk - res['weights'])
                l2_vals.append(f"{diff:.2e}")
            else:
                l2_vals.append("-")
        print(metric_fmt.format(*l2_vals))
        
        # Portfolio Risk (w.T * Cov * w)
        cov = np.cov(X, rowvar=False)
        risk_sk = weights_sk @ cov @ weights_sk
        risk_vals = ["Risk (Var)", f"{risk_sk:.6f}"]

        for name, _ in number_types:
            res = results.get(name)
            if res and res['weights'] is not None:
                w = res['weights']
                risk = w @ cov @ w
                risk_vals.append(f"{risk:.6f}")
            else:
                risk_vals.append("-")
        print(metric_fmt.format(*risk_vals))

        # --- Reliability Checks ---
        print(separator)
        
        # 1. Sum of Weights (Validity: Should be 1.0)
        sum_vals = ["Sum Weights", f"{np.sum(weights_sk):.6f}"]
        for name, _ in number_types:
            res = results.get(name)
            if res and res['weights'] is not None:
                sum_w = np.sum(res['weights'])
                sum_vals.append(f"{sum_w:.6f}")
            else:
                sum_vals.append("-")
        print(metric_fmt.format(*sum_vals))

        # 2. Constraint Violation (Negativity: Should be 0.0)
        neg_vals = ["Negativity", f"{np.sum(np.abs(np.minimum(weights_sk, 0))):.2e}"]
        for name, _ in number_types:
            res = results.get(name)
            if res and res['weights'] is not None:
                neg_w = np.sum(np.abs(np.minimum(res['weights'], 0)))
                neg_vals.append(f"{neg_w:.2e}")
            else:
                neg_vals.append("-")
        print(metric_fmt.format(*neg_vals))

        # 3. Risk Gap % (Efficiency: (Risk_Posit - Risk_Sk) / Risk_Sk)
        gap_vals = ["Risk Gap %", "0.00%"]
        for name, _ in number_types:
            res = results.get(name)
            if res and res['weights'] is not None:
                w = res['weights']
                risk_p = w @ cov @ w
                if risk_sk > 0:
                    gap = (risk_p - risk_sk) / risk_sk * 100
                    gap_vals.append(f"{gap:+.4f}%")
                else:
                    gap_vals.append("N/A")
            else:
                gap_vals.append("-")
        print(metric_fmt.format(*gap_vals))

        # 4. Max Abs Diff (Similarity)
        mad_vals = ["Max Abs Diff", "0.00e+00"]
        for name, _ in number_types:
            res = results.get(name)
            if res and res['weights'] is not None:
                mad = np.max(np.abs(weights_sk - res['weights']))
                mad_vals.append(f"{mad:.2e}")
            else:
                mad_vals.append("-")
        print(metric_fmt.format(*mad_vals))
        
        print(separator)
        print("\n")

if __name__ == "__main__":
    print("Running Custom Comparison with Synthetic Data...")
    
    # Generate random synthetic data: 100 days, 5 assets
    np.random.seed(42)
    n_samples, n_assets = 100, 5
    # Random returns roughly Gaussian around 0
    X_dummy = np.random.randn(n_samples, n_assets) * 0.01 + 0.0005
    
    asset_ids = [f"Stock {chr(65+i)}" for i in range(n_assets)]
    
    custom_scaling = [
        ('none', 1.0),
        ('manual', 1000.0)
    ]
    
    # Run comparison
    run_comparison(X_dummy, asset_names=asset_ids, scaling_strategies=custom_scaling)
