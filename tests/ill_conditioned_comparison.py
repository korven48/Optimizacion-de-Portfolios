#!/usr/bin/env python3
"""
Prueba de Estrés: Comparación de Matrices Mal Condicionadas
Genera datos sintéticos para probar la optimización Posit vs Float bajo estrés.
"""

import sys
import os
import numpy as np
import scipy.linalg

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom_comparison import run_comparison

def generate_highly_correlated(n_samples, n_assets, rho=0.999):
    """
    Genera retornos con correlación constante rho.
    La matriz de covarianza será cercana a singular.
    """
    print(f"  Generando Alta Correlación (rho={rho})...")
    # 1. Crear Matriz de Correlación R
    R = np.full((n_assets, n_assets), rho)
    np.fill_diagonal(R, 1.0)
    
    # 2. Descomposición de Cholesky: R = L @ L.T
    # Si rho es muy alto, R podría no ser definida positiva para precisión float,
    # así que debemos tener cuidado o usar SVD.
    # Para generación sintética, podemos usar:
    # X = Z @ L.T
    
    # Forma simple:
    # Modelo de factor compartido: r_i = sqrt(rho)*F + sqrt(1-rho)*epsilon_i
    # Esto garantiza correlación esperada rho.
    
    F = np.random.randn(n_samples, 1)
    E = np.random.randn(n_samples, n_assets)
    
    X = np.sqrt(rho) * F + np.sqrt(1 - rho) * E
    
    # Escalar a magnitud típica de retornos
    X = X * 0.01 + 0.0005
    return X, [f"Corr {i}" for i in range(n_assets)]

def generate_scale_imbalance(n_samples, n_assets, power_range=6):
    """
    Genera retornos donde los activos tienen varianzas vastamente diferentes.
    Las escalas van de 10^(-power/2) a 10^(power/2).
    """
    print(f"  Generando Desbalance de Escala (10^-{power_range//2} a 10^{power_range//2})...")
    X = np.random.randn(n_samples, n_assets)
    
    # Escalas espaciadas logarítmicamente
    scales = np.logspace(-power_range/2, power_range/2, n_assets)
    
    # Aplicar escalas
    X = X * scales
    
    # Desplazamiento base de retorno
    X = X + 0.0005
    
    assets = [f"Scale 1e{int(np.log10(s))}" for s in scales]
    return X, assets

def generate_rank_deficient(n_samples, n_assets):
    """
    Genera caso de prueba donde N > T.
    La matriz de covarianza muestral será singular (rango deficiente).
    """
    print(f"  Generando Rango Deficiente (T={n_samples} < N={n_assets})...")
    X = np.random.randn(n_samples, n_assets) * 0.01 + 0.0005
    return X, [f"Asset {i}" for i in range(n_assets)]

def generate_tiny_scaling_battle(n_samples, n_assets, condition_number=100.0):
    print(f"  Generando Datos Diminutos (Media ~ 5e-6, Cov ~ 1e-8)...")
    np.random.seed(42)
    
    # 1. Generar Covarianza Mal Condicionada
    H = np.random.randn(n_assets, n_assets)
    Q, _ = np.linalg.qr(H)
    eigenvalues = np.logspace(0, -np.log10(condition_number), n_assets)
    S = np.diag(eigenvalues)
    cov_base = Q @ S @ Q.T
    
    # Escalar para coincidir con scaling_battle_16bit
    cov_tiny = cov_base * 1e-8
    
    # 2. Generar Mu
    mu_tiny = (np.random.randn(n_assets) * 0.01 + 0.05) * 1e-4
    
    # 3. Generar X de N(mu, cov)
    # Necesitamos n_samples.
    X = np.random.multivariate_normal(mu_tiny, cov_tiny, n_samples)
    
    return X, [f"Tiny {i}" for i in range(n_assets)]

if __name__ == "__main__":
    
    # 1. Alta Correlación
    # print("\n>>> PRUEBA 1: ALTA CORRELACIÓN (Multicolinealidad) <<<")
    # X_corr, assets_corr = generate_highly_correlated(100, 5, rho=0.9999)
    # run_comparison(X_corr, asset_names=assets_corr, scaling_strategies=[('none', 1.0), ('auto_max_abs', 1.0)])
    
    # 2. Desbalance de Escala
    # print("\n>>> PRUEBA 2: DESBALANCE DE ESCALA (Problema Rígido) <<<")
    # X_scale, assets_scale = generate_scale_imbalance(100, 5, power_range=8) # 1e-4 a 1e4
    # run_comparison(X_scale, asset_names=assets_scale, scaling_strategies=[('none', 1.0), ('auto_max_abs', 1.0)])
    
    # 3. Rango Deficiente
    # print("\n>>> PRUEBA 3: RANGO DEFICIENTE (Más Activos que Muestras) <<<")
    # X_rank, assets_rank = generate_rank_deficient(50, 20)
    # run_comparison(X_rank, asset_names=assets_rank[0:5], scaling_strategies=[('none', 1.0)]) 

    # 4. Batalla de Escalado (Números Diminutos)
    # Aqui fallaba float16, no consigo replicarlo
    # Aumentamos learning_rate a 10000.0 porque gradientes serán ~1e-8.
    print("\n>>> PRUEBA 4: NÚMEROS DIMINUTOS <<<")
    X_tiny, assets_tiny = generate_tiny_scaling_battle(100, 10, condition_number=100.0)
    # print(X_tiny)
    run_comparison(
        X_tiny, 
        asset_names=assets_tiny, 
        scaling_strategies=[
            ('none', 1.0)
            # ('max', 1.0),
            # ('std', 1.0),
            # ('frobenius', 1.0),
            # ('pow2', 1.0)
        ],
        solver_params={
            'learning_rate': 10000.0
        }
    )
