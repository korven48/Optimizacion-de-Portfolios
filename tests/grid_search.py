#!/usr/bin/env python3

import sys
import os
import pandas as pd
import numpy as np

# Asegurar que el directorio raíz y el de tests están en el path
# Asumiendo que el script está en <root>/scripts/reproduce_grid_search.py o en la raíz
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(CURRENT_DIR) == 'scripts':
    ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
else:
    ROOT_DIR = CURRENT_DIR

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'tests'))

# Importaciones de los módulos de soporte
try:
    from ill_conditioned_comparison import generate_highly_correlated, generate_tiny_scaling_battle
    from real_asset_comparison import load_real_data
    from custom_comparison import run_comparison
except ImportError:
    from tests.ill_conditioned_comparison import generate_highly_correlated, generate_tiny_scaling_battle
    from tests.real_asset_comparison import load_real_data
    from tests.custom_comparison import run_comparison

# Importar los wrappers de tipos de datos
from posit_lib.float_wrapper import (
    Float64Wrapper, Float32Wrapper, Float16Wrapper, 
    BFloat16Wrapper, Float8_e4m3fn_Wrapper, Float8_e5m2_Wrapper
)
from posit_lib import posit

def main():
    # El archivo de salida
    results_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "full_grid_search_resultados.csv"))
    
    # Si el archivo ya existe, lo borramos para empezar de cero (o lo renombramos)
    if os.path.exists(results_file):
        os.remove(results_file)

    print("="*80)
    print("REPRODUCCIÓN DE GRID SEARCH MASIVO - TFG")
    print("="*80)
    print(f"Los resultados se guardarán en: {results_file}")
    print("="*80)

    # 1. PARÁMETROS DEL GRID SEARCH
    
    tolerances = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    
    scaling_strategies = [
        ('none', 1.0),
        ('manual', 100.0),
        ('max', 1.0),
        ('std', 1.0),
        ('frobenius', 1.0),
        ('pow2', 1.0)
    ]
    
    golden_zone_options = [False, True]

    # Lista completa de tipos numéricos a evaluar
    number_types = [
        ("Float16", Float16Wrapper),
        ("BFloat16", BFloat16Wrapper),
        ("Float32", Float32Wrapper),
        ("Float64", Float64Wrapper),
        ("Posit8", posit.Posit8),
        ("Posit12", posit.Posit12),
        ("Posit16", posit.Posit16),
        ("Posit20", posit.Posit20),
        ("Posit24", posit.Posit24),
        ("Posit32", posit.Posit32),
        ("Posit64", posit.Posit64),
        ("Float8_e4m3fn", Float8_e4m3fn_Wrapper),
        ("Float8_e5m2", Float8_e5m2_Wrapper)
    ]

    # 2. CARGA DE DATASETS
    
    print("\nGenerando/Descargando datasets...")
    datasets = []
    
    # A. Sintético: Alta Correlación
    X_corr, assets_corr = generate_highly_correlated(100, 5, rho=0.9999)
    datasets.append({
        'name': 'Synthetic: Alta Correlacion (rho=0.9999)',
        'X': X_corr,
        'assets': assets_corr
    })
    
    # B. Sintético: Números Diminutos
    X_tiny, assets_tiny = generate_tiny_scaling_battle(100, 10, condition_number=100.0)
    datasets.append({
        'name': 'Synthetic: Numeros Diminutos (Underflow)',
        'X': X_tiny,
        'assets': assets_tiny
    })
    
    # C. Real: Mensual (Yahoo Finance)
    try:
        X_mensual, assets_mensual = load_real_data(start="2018-01-01", end="2026-01-01", interval="1mo")
        datasets.append({
            'name': 'Real Assets: Mensual (1mo)',
            'X': X_mensual,
            'assets': assets_mensual
        })
    except Exception as e:
        print(f"Error cargando datos mensuales: {e}")
        
    # D. Real: Diario (Yahoo Finance)
    try:
        X_diario, assets_diario = load_real_data(start="2018-01-01", end="2026-01-01", interval="1d")
        datasets.append({
            'name': 'Real Assets: Diario (1d)',
            'X': X_diario,
            'assets': assets_diario
        })
    except Exception as e:
        print(f"Error cargando datos diarios: {e}")

    # 3. EJECUCIÓN DEL GRID SEARCH
    
    total_combinations = len(datasets) * len(tolerances) * len(golden_zone_options)
    current_iteration = 0
    
    print(f"\nIniciando {total_combinations} pasadas de optimización...\n")
    
    for ds in datasets:
        for tol in tolerances:
            for gz in golden_zone_options:
                current_iteration += 1
                
                # Configuración específica para el dataset de números diminutos
                current_lr = 10000.0 if "Diminutos" in ds['name'] else 0.1
                
                solver_params = {
                    'tolerance': tol,
                    'learning_rate': current_lr,
                    'max_iterations': 10000,
                    'momentum': 0.9,
                    'objective_function': 'MINIMIZE_RISK'
                }
                
                print(f"[{current_iteration}/{total_combinations}] Dataset: {ds['name']} | Tol: {tol} | GZ: {gz}")
                
                # Ejecutar comparación para este bloque de parámetros
                run_comparison(
                    ds['X'], 
                    asset_names=ds['assets'], 
                    scaling_strategies=scaling_strategies, 
                    number_types=number_types,
                    solver_params=solver_params,
                    scale_to_golden_zone=gz,
                    dataset_name=ds['name'], 
                    export_csv=results_file, 
                    print_console=False
                )

    print("\n" + "="*80)
    print("PROCESO COMPLETADO")
    print(f"Resultados generados en: {results_file}")
    print("="*80)

if __name__ == "__main__":
    main()
