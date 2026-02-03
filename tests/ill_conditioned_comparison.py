#!/usr/bin/env python3
"""
Stress Test: Ill-Conditioned Matrices Comparison
Generates synthetic data to stress test Posit vs Float optimization.
"""

import sys
import os
import numpy as np
import scipy.linalg

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from examples.custom_comparison import run_comparison

def generate_highly_correlated(n_samples, n_assets, rho=0.999):
    """
    Generates returns with constant correlation rho.
    Covariance matrix will be close to singular.
    """
    print(f"  Generating High Correlation (rho={rho})...")
    # 1. Create Correlation Matrix R
    R = np.full((n_assets, n_assets), rho)
    np.fill_diagonal(R, 1.0)
    
    # 2. Cholesky Decomposition: R = L @ L.T
    # If rho is too high, R might be not positive definite for float precision,
    # so we might need to be careful or just use SVD.
    # For synthetic generation, we can use:
    # X = Z @ L.T
    
    # Simple way:
    # Shared factor model: r_i = sqrt(rho)*F + sqrt(1-rho)*epsilon_i
    # This guarantees expected correlation rho.
    
    F = np.random.randn(n_samples, 1)
    E = np.random.randn(n_samples, n_assets)
    
    X = np.sqrt(rho) * F + np.sqrt(1 - rho) * E
    
    # Scale to typical returns magnitude
    X = X * 0.01 + 0.0005
    return X, [f"Corr {i}" for i in range(n_assets)]

def generate_scale_imbalance(n_samples, n_assets, power_range=6):
    """
    Generates returns where assets have vastly different variances.
    Scales range from 10^(-power/2) to 10^(power/2).
    """
    print(f"  Generating Scale Imbalance (10^-{power_range//2} to 10^{power_range//2})...")
    X = np.random.randn(n_samples, n_assets)
    
    # Log-spaced scales
    scales = np.logspace(-power_range/2, power_range/2, n_assets)
    
    # Apply scales
    X = X * scales
    
    # Base return shift
    X = X + 0.0005
    
    assets = [f"Scale 1e{int(np.log10(s))}" for s in scales]
    return X, assets

def generate_rank_deficient(n_samples, n_assets):
    """
    Generates checking case where N > T.
    The sample covariance matrix will be singular (rank deficient).
    """
    print(f"  Generating Rank Deficient (T={n_samples} < N={n_assets})...")
    X = np.random.randn(n_samples, n_assets) * 0.01 + 0.0005
    return X, [f"Asset {i}" for i in range(n_assets)]

def generate_tiny_scaling_battle(n_samples, n_assets, condition_number=100.0):
    print(f"  Generating Tiny Data (Mean ~ 5e-6, Cov ~ 1e-8)...")
    np.random.seed(42)
    
    # 1. Generate Ill-conditioned Covariance
    H = np.random.randn(n_assets, n_assets)
    Q, _ = np.linalg.qr(H)
    eigenvalues = np.logspace(0, -np.log10(condition_number), n_assets)
    S = np.diag(eigenvalues)
    cov_base = Q @ S @ Q.T
    
    # Scale to match scaling_battle_16bit
    cov_tiny = cov_base * 1e-8
    
    # 2. Generate Mu
    mu_tiny = (np.random.randn(n_assets) * 0.01 + 0.05) * 1e-4
    
    # 3. Generate X from N(mu, cov)
    # We need n_samples.
    X = np.random.multivariate_normal(mu_tiny, cov_tiny, n_samples)
    
    return X, [f"Tiny {i}" for i in range(n_assets)]

if __name__ == "__main__":
    
    # 1. High Correlation
    # print("\n>>> TEST 1: HIGH CORRELATION (Multicollinearity) <<<")
    # X_corr, assets_corr = generate_highly_correlated(100, 5, rho=0.9999)
    # run_comparison(X_corr, asset_names=assets_corr, scaling_strategies=[('none', 1.0), ('auto_max_abs', 1.0)])
    
    # 2. Scale Imbalance
    # print("\n>>> TEST 2: SCALE IMBALANCE (Stiff Problem) <<<")
    # X_scale, assets_scale = generate_scale_imbalance(100, 5, power_range=8) # 1e-4 to 1e4
    # run_comparison(X_scale, asset_names=assets_scale, scaling_strategies=[('none', 1.0), ('auto_max_abs', 1.0)])
    
    # 3. Rank Deficient
    # print("\n>>> TEST 3: RANK DEFICIENT (More Assets than Samples) <<<")
    # X_rank, assets_rank = generate_rank_deficient(50, 20)
    # run_comparison(X_rank, asset_names=assets_rank[0:5], scaling_strategies=[('none', 1.0)]) 
    # Note: passing truncated names list just for display if run_comparison handles it, 
    # but run_comparison uses X.shape[1].
    # Let's verify run_comparison logic: "if len(asset_names) != n_assets... truncating or padding".
    # So we should pass correct length or let it pad.
    # run_comparison(X_rank, asset_names=[f"A{i}" for i in range(20)], scaling_strategies=[('none', 1.0)])

    # 4. Scaling Battle (Tiny Numbers)
    print("\n>>> TEST 4: TINY NUMBERS (Replicating Scaling Battle) <<<")
    # Derived from experiments/scaling_battle_16bit.py
    # Cov ~ 1e-8, Mu ~ 1e-4. Causes underflow/overflow in fp16.
    


    X_tiny, assets_tiny = generate_tiny_scaling_battle(100, 10, condition_number=100.0)
    # We expect 'none' strategy to FAIL on Float16 here
    # We boost learning_rate to 10000.0 because gradients will be ~1e-8.
    run_comparison(
        X_tiny, 
        asset_names=assets_tiny, 
        scaling_strategies=[
            ('none', 1.0),
            ('max', 1.0),
            ('std', 1.0),
            ('frobenius', 1.0),
            ('pow2', 1.0)
        ],
        solver_params={'learning_rate': 10000.0}
    )
