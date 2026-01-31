#!/usr/bin/env python3
"""
Scaling Battle 16-bit: Posit16 vs Float16 (Half Precision)
"""

import sys
import os
import numpy as np
import time
import warnings

# Añadir directorio padre al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from posit_lib import posit
from posit_lib.float_wrapper import FloatWrapper, Float16Wrapper
from posit_lib.solver import PGDSolver

def generate_ill_conditioned_matrix(n, condition_number):
    """Genera matriz de covarianza con número de condición específico."""
    H = np.random.randn(n, n)
    Q, R = np.linalg.qr(H)
    eigenvalues = np.logspace(0, -np.log10(condition_number), n)
    S = np.diag(eigenvalues)
    cov_matrix = Q @ S @ Q.T
    return cov_matrix

def apply_scaling(cov, mu, strategy):
    """
    Aplica una estrategia de escalado a la matriz de covarianza y retornos.
    Retorna: (cov_scaled, mu_scaled, scale_factor_applied)
    """
    if strategy == 'none':
        return cov, mu, 1.0
        
    vals = np.abs(mu)
    if np.max(vals) == 0: return cov, mu, 1.0

    S = 1.0
    
    if strategy == 'max':
        # S = 1 / max(|mu|)
        S = 1.0 / np.max(vals)
        
    elif strategy == 'std':
        # S = 1 / std(mu)
        sigma = np.std(vals)
        if sigma > 0:
            S = 1.0 / sigma
            
    elif strategy == 'frobenius':
        # S = 1 / ||mu||_2
        norm = np.linalg.norm(vals)
        if norm > 0:
            S = 1.0 / norm
            
    elif strategy == 'pow2':
        # S = 2^k más cercano a 1/mean(vals)
        avg = np.mean(vals)
        if avg > 0:
            target_scale = 1.0 / avg
            k = np.round(np.log2(target_scale))
            S = 2.0 ** k
            
    mu_scaled = mu * S
    cov_scaled = cov * (S * S)
    
    return cov_scaled, mu_scaled, S

def run_scaling_battle_16bit():
    print("==========================================================")
    print("  SCALING BATTLE 16-BIT: Posit16 vs Float16")
    print("==========================================================")
    
    N_ASSETS = 10
    COND_NUM = 1e2 
    
    # Generar datos base muy pequeños para desafiar a los 16 bits
    np.random.seed(42)
    cov_raw = generate_ill_conditioned_matrix(N_ASSETS, COND_NUM)
    cov_raw = cov_raw * 1e-8 
    mu_raw = (np.random.randn(N_ASSETS) * 0.01 + 0.05) * 1e-4
    
    print(f"Datos Originales: Mean(Mu)={np.mean(np.abs(mu_raw)):.2e}")
    
    # Baseline (Float64 Reference)
    baseline_solver = PGDSolver(FloatWrapper)
    mu_ref_scaled = mu_raw * 1e4
    cov_ref_scaled = cov_raw * 1e8
    w_ref, _ = baseline_solver.solve(
        'MINIMIZE_RISK', cov_ref_scaled.tolist(), mu_ref_scaled.tolist(), max_iterations=5000, tolerance=1e-13
    )
    w_ref_np = np.array(w_ref, dtype=np.float64)
    print(f"Riesgo Ref: {w_ref_np @ cov_raw @ w_ref_np:.8e}")
    
    STRATEGIES = ['none', 'max', 'std', 'frobenius', 'pow2']
    
    print("\nIniciando Batalla: Posit16 vs Float16...")
    print(f"{'Strategy':<10} | {'Scale':<9} | {'P16 Err':<9} | {'F16 Err':<9} | {'P16 It':<6} | {'F16 It':<6}")
    print("-" * 75)
    
    best_strategy = "None"
    best_error = float('inf')
    
    errors_log = []

    for strategy in STRATEGIES:
        try:
            cov_in, mu_in, S = apply_scaling(cov_raw, mu_raw, strategy)
            
            # Ajustar Learning Rate
            if strategy == 'none':
                 avg_mag = np.mean(np.abs(mu_raw))
                 # Heurística para unscaled
                 lr_adj = 0.001 / (avg_mag**2) if avg_mag > 0 else 0.1
                 lr_adj = min(lr_adj, 1e9) 
            else:
                 # Scaled std LR
                 lr_adj = 1e-3
            
            # Run Posit16
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    solver_p = PGDSolver(posit.Posit16)
                    w_p, it_p = solver_p.solve('MINIMIZE_RISK', cov_in.tolist(), mu_in.tolist(), max_iterations=1000, learning_rate=lr_adj, tolerance=1e-5)
                    w_p_np = np.array([float(x) for x in w_p])
                    err_p = np.linalg.norm(w_p_np - w_ref_np)
                    if w:
                        for warning in w:
                            msg = f"{strategy} (Posit16 Warning): {warning.message}"
                            if msg not in errors_log:
                                errors_log.append(msg)
            except Exception as e:
                err_p = float('nan')
                it_p = -1
                msg = f"{strategy} (Posit16 Error): {str(e)}"
                if msg not in errors_log:
                    errors_log.append(msg)
            
            # Run Float16
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    solver_f = PGDSolver(Float16Wrapper)
                    w_f, it_f = solver_f.solve('MINIMIZE_RISK', cov_in.tolist(), mu_in.tolist(), max_iterations=1000, learning_rate=lr_adj, tolerance=1e-5)
                    w_f_np = np.array([float(x) for x in w_f])
                    err_f = np.linalg.norm(w_f_np - w_ref_np)
                    if w:
                        for warning in w:
                            msg = f"{strategy} (Float16 Warning): {warning.message}"
                            if msg not in errors_log:
                                errors_log.append(msg)
            except Exception as e:
                err_f = float('nan')
                it_f = -1
                msg = f"{strategy} (Float16 Error): {str(e)}"
                if msg not in errors_log:
                    errors_log.append(msg)

            print(f"{strategy:<10} | {S:.1e}   | {err_p:.1e}   | {err_f:.1e}   | {it_p:<6} | {it_f:<6}")
            
            if not np.isnan(err_p) and err_p < best_error:
                best_error = err_p
                best_strategy = strategy
                
        except Exception as e:
            print(f"{strategy:<10} | ERROR: {e}")
            errors_log.append(f"{strategy} (Setup Error): {str(e)}")

    print("-" * 75)
    print(f"Ganador 16-bit: {best_strategy.upper()}")
    
    if errors_log:
        print("\n" + "="*50)
        print("  ERRORES ENCONTRADOS")
        print("="*50)
        for err in errors_log:
            print(f" - {err}")
        print("="*50)

if __name__ == "__main__":
    run_scaling_battle_16bit()
