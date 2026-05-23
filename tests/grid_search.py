#!/usr/bin/env python3
"""
Este script realiza una búsqueda en rejilla, evaluando diferentes
estrategias de escalado, tolerancias y tipos de precisión numérica
(incluyendo aritmética Posit y wrappers de punto flotante).
"""

import sys
import os
import pandas as pd
import numpy as np

# Configurar el directorio raíz del proyecto en el sys.path para importaciones robustas
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tests.ill_conditioned_comparison import generate_highly_correlated, generate_tiny_scaling_battle
from tests.real_asset_comparison import load_real_data
from tests.custom_comparison import run_comparison

from posit_lib.float_wrapper import (
    Float64Wrapper, Float32Wrapper, Float16Wrapper, 
    BFloat16Wrapper, Float8_e4m3fn_Wrapper, Float8_e5m2_Wrapper
)
from posit_lib import posit

def main():
    # Establecer semilla aleatoria para asegurar la replicabilidad de los datos sintéticos
    np.random.seed(42)
    
    results_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "full_grid_search_resultados.csv"))
    
    if os.path.exists(results_file):
        os.remove(results_file)

    print("=" * 80)
    print("REPRODUCCIÓN DE GRID SEARCH MASIVO - TRABAJO DE FIN DE GRADO")
    print("=" * 80)
    print(f"Los resultados se guardarán en:\n  {results_file}")
    print("=" * 80)

    # PARÁMETROS DEL GRID SEARCH
    
    # Tolerancias de parada para el solver PGD
    tolerances = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    
    # Estrategias de escalado de datos de retorno
    scaling_strategies = [
        ('none', 1.0),
        ('manual', 100.0),
        ('max', 1.0),
        ('std', 1.0),
        ('frobenius', 1.0),
        ('pow2', 1.0)
    ]
    
    # Opciones para el escalado dinámico a la zona dorada de representación
    golden_zone_options = [False, True]

    # Lista completa de formatos numéricos a evaluar para comparar precisión
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

    # CARGA Y GENERACIÓN DE DATASETS
    
    print("\nGenerando y cargando conjuntos de datos...")
    datasets = []
    
    # Datos Sintéticos con Alta Correlación (Multicolinealidad extrema)
    X_corr, assets_corr = generate_highly_correlated(100, 5, rho=0.9999)
    datasets.append({
        'name': 'Synthetic: Alta Correlacion (rho=0.9999)',
        'X': X_corr,
        'assets': assets_corr
    })
    
    # Datos Sintéticos Diminutos (Riesgo alto de Underflow en formatos de baja precisión)
    X_tiny, assets_tiny = generate_tiny_scaling_battle(100, 10, condition_number=100.0)
    datasets.append({
        'name': 'Synthetic: Numeros Diminutos (Underflow)',
        'X': X_tiny,
        'assets': assets_tiny
    })
    
    # Datos Reales Históricos: Intervalo Mensual
    try:
        X_mensual, assets_mensual = load_real_data(start="2018-01-01", end="2026-01-01", interval="1mo")
        datasets.append({
            'name': 'Real Assets: Mensual (1mo)',
            'X': X_mensual,
            'assets': assets_mensual
        })
    except Exception as e:
        print(f"Advertencia: No se pudieron cargar los datos reales mensuales: {e}")
        
    # Datos Reales Históricos: Intervalo Diario
    try:
        X_diario, assets_diario = load_real_data(start="2018-01-01", end="2026-01-01", interval="1d")
        datasets.append({
            'name': 'Real Assets: Diario (1d)',
            'X': X_diario,
            'assets': assets_diario
        })
    except Exception as e:
        print(f"Advertencia: No se pudieron cargar los datos reales diarios: {e}")

    # PROCESAMIENTO Y EJECUCIÓN
    
    total_combinations = len(datasets) * len(tolerances) * len(golden_zone_options)
    current_iteration = 0
    
    print(f"\nIniciando búsqueda en rejilla ({total_combinations} combinaciones de control)...\n")
    
    for ds in datasets:
        for tol in tolerances:
            for gz in golden_zone_options:
                current_iteration += 1

                current_lr = 0.1
                
                solver_params = {
                    'tolerance': tol,
                    'learning_rate': current_lr,
                    'max_iterations': 10000,
                    'momentum': 0.9,
                    'objective_function': 'MINIMIZE_RISK'
                }
                
                print(f"[{current_iteration}/{total_combinations}] Procesando: {ds['name']} | Tol: {tol} | Golden Zone: {gz}")
                
                # Ejecutar comparación del solver para todas las estrategias y tipos numéricos
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

    print("\n" + "=" * 80)
    print("BÚSQUEDA EN REJILLA COMPLETADA CON ÉXITO")
    print(f"Todos los resultados consolidados en:\n  {results_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
