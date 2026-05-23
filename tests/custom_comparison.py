#!/usr/bin/env python3
"""
Script de Comparación de Datos Personalizados
Permite a los usuarios ejecutar la comparación Skfolio vs Posit/Float con sus propios datos.
"""

import sys
import os
import shutil
import time
import numpy as np
import pandas as pd

# Agregar directorio padre al path para encontrar posit_lib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

from posit_lib.adapters.skfolio_adapter import PositMeanVariance
from posit_lib.float_wrapper import Float64Wrapper, Float16Wrapper, Float32Wrapper, BFloat16Wrapper, Float8_e4m3fn_Wrapper, Float8_e5m2_Wrapper
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

def print_comparison_results(results_df, asset_names, n_samples):
    """Muestra los resultados en consola tal y como se hacía originalmente."""
    n_assets = len(asset_names)
    print_header("COMPARACIÓN DE DATOS PERSONALIZADOS: SKFOLIO vs POSIT vs FLOAT")
    print(f"Forma de Datos: {n_samples} muestras x {n_assets} activos")
    print_separator()
    
    sk_row = results_df[results_df['Number_Type'] == 'Skfolio (Float64)'].iloc[0]
    weights_sk = sk_row['Weights_Array']
    time_sk = sk_row['Time_s']
    
    print("\n[Skfolio Oficial] MeanRisk(objective_function='MINIMIZE_RISK')...")
    print(f"  Tiempo: {time_sk:.4f}s")
    
    # Extraer pares (Estrategia, Factor) únicos sin N/A
    strats = results_df[results_df['Scaling_Strategy'] != 'N/A'][['Scaling_Strategy', 'Scaling_Factor']].drop_duplicates()
    
    # Extraer tipos numericos preservando orden original
    number_types = []
    for nt in results_df['Number_Type']:
        if nt != 'Skfolio (Float64)' and nt not in number_types:
            number_types.append(nt)

    for _, strat in strats.iterrows():
        scale_type = strat['Scaling_Strategy']
        scale_factor = strat['Scaling_Factor']
        
        print("\n" + "=" * 100)
        print(f"ESTRATEGIA DE ESCALADO: {(scale_type, scale_factor)}")
        print("=" * 100)
        # Re-imprimimos los mensajes de "Calculando..." para mantener el output 100% identico,
        # aunque en este caso ya se calculó todo antes.
        for nt in number_types:
            print(f"Calculando estadísticas con {nt} (esto puede tardar)...")
        
        print("\n")
        print_header(f"RESULTADOS {(scale_type, scale_factor)}")
        
        headers = ["Asset", "Skfolio"] + number_types
        col_width = 12
        asset_width = 16

        row_fmt = f"| {{:<{asset_width}}} | {{:>{col_width}}}" + f" | {{:>{col_width}}}" * len(number_types) + " |"
        sample_line = row_fmt.format(*headers)
        separator = "-" * len(sample_line)

        print(separator)
        print(row_fmt.format(*headers))
        print(separator)
        
        strat_df = results_df[(results_df['Scaling_Strategy'] == scale_type) & (results_df['Scaling_Factor'] == scale_factor)]
        
        for i in range(n_assets):
            row_vals = [asset_names[i], f"{weights_sk[i]:.6f}"]
            for nt in number_types:
                res = strat_df[strat_df['Number_Type'] == nt]
                if not res.empty and res.iloc[0]['Weights_Array'] is not None:
                    row_vals.append(f"{res.iloc[0]['Weights_Array'][i]:.6f}")
                else:
                    row_vals.append("ERROR")
            print(row_fmt.format(*row_vals))

        print(separator)

        # Métricas Globales
        print("\nMETRICAS DE RENDIMIENTO:")
        print(separator)
        metric_fmt = f"| {{:<{asset_width}}} | {{:>{col_width}}}" + f" | {{:>{col_width}}}" * len(number_types) + " |"

        # Tiempo
        time_vals = ["Time (s)", f"{time_sk:.4f}"]
        for nt in number_types:
            res = strat_df[strat_df['Number_Type'] == nt]
            val = res.iloc[0]['Time_s'] if not res.empty and pd.notna(res.iloc[0]['Time_s']) else None
            time_vals.append(f"{val:.4f}" if val is not None else "-")
        print(metric_fmt.format(*time_vals))

        # Iteraciones
        iter_vals = ["Iterations", "N/A"]
        for nt in number_types:
            res = strat_df[strat_df['Number_Type'] == nt]
            val = res.iloc[0]['Iterations'] if not res.empty and pd.notna(res.iloc[0]['Iterations']) else None
            iter_vals.append(f"{int(val)}" if val is not None else "-")
        print(metric_fmt.format(*iter_vals))

        # Error L2
        l2_vals = ["Error L2 (vs Sk)", "0.00e+00"]
        for nt in number_types:
            res = strat_df[strat_df['Number_Type'] == nt]
            val = res.iloc[0]['Error_L2'] if not res.empty and pd.notna(res.iloc[0]['Error_L2']) else None
            l2_vals.append(f"{val:.2e}" if val is not None else "-")
        print(metric_fmt.format(*l2_vals))
        
        # Riesgo del Portafolio
        risk_vals = ["Riesgo (Var)", f"{sk_row['Risk_Variance']:.9f}"]
        for nt in number_types:
            res = strat_df[strat_df['Number_Type'] == nt]
            if not res.empty and pd.notna(res.iloc[0]['Risk_Variance']):
                risk = res.iloc[0]['Risk_Variance']
                grad_zero = res.iloc[0]['Grad_Zero_Detected']
                if risk == 0.0:
                    risk_vals.append(f"{risk:0} (!)")
                elif grad_zero: 
                    risk_vals.append(f"{risk:.5f} [G0]")
                else:
                    risk_vals.append(f"{risk:.9f}")
            else:
                risk_vals.append("-")
        print(metric_fmt.format(*risk_vals))

        print(separator)
        
        # 1. Suma de Pesos
        sum_vals = ["Suma Pesos", f"{sk_row['Sum_Weights']:.6f}"]
        for nt in number_types:
            res = strat_df[strat_df['Number_Type'] == nt]
            val = res.iloc[0]['Sum_Weights'] if not res.empty and pd.notna(res.iloc[0]['Sum_Weights']) else None
            sum_vals.append(f"{val:.6f}" if val is not None else "-")
        print(metric_fmt.format(*sum_vals))

        # 2. Negatividad
        neg_vals = ["Negatividad", f"{sk_row['Negativity_Violation']:.2e}"]
        for nt in number_types:
            res = strat_df[strat_df['Number_Type'] == nt]
            val = res.iloc[0]['Negativity_Violation'] if not res.empty and pd.notna(res.iloc[0]['Negativity_Violation']) else None
            neg_vals.append(f"{val:.2e}" if val is not None else "-")
        print(metric_fmt.format(*neg_vals))

        # 3. Brecha de Riesgo %
        gap_vals = ["Brecha Riesgo %", "0.00%"]
        for nt in number_types:
            res = strat_df[strat_df['Number_Type'] == nt]
            val = res.iloc[0]['Risk_Gap_Pct'] if not res.empty and pd.notna(res.iloc[0]['Risk_Gap_Pct']) else None
            gap_vals.append(f"{val:+.4f}%" if val is not None else "-")
        print(metric_fmt.format(*gap_vals))

        # 4. Dif Abs Max
        mad_vals = ["Dif Abs Max", "0.00e+00"]
        for nt in number_types:
            res = strat_df[strat_df['Number_Type'] == nt]
            val = res.iloc[0]['Max_Abs_Diff'] if not res.empty and pd.notna(res.iloc[0]['Max_Abs_Diff']) else None
            mad_vals.append(f"{val:.2e}" if val is not None else "-")
        print(metric_fmt.format(*mad_vals))
        
        print(separator)
        print("Leyenda:")
        print("  (!)  Posible Underflow detectado (Riesgo == 0.0)")
        print("  [G0] Gradiente colapsó a Cero durante la optimización (Underflow)")
        print("\n")


def export_results_to_csv(results_df, filepath):
    """Vuelca el DataFrame al disco. Usa 'append' silenciosamente si ya existe."""
    # Convertimos el array de pesos explícitamente a string antes del csv
    # para evitar problemas de formato raros, garantizando que "[0.1, 0.2]" se guarde literal.
    df_out = results_df.copy()
    df_out['Weights_Array'] = df_out['Weights_Array'].apply(lambda x: str(x) if x is not None else "")
    
    file_exists = os.path.isfile(filepath)
    df_out.to_csv(filepath, mode='a' if file_exists else 'w', header=not file_exists, index=False)


def run_comparison(X, asset_names=None, scaling_strategies=None, number_types=None, solver_params=None, 
                   scale_to_golden_zone=False, dataset_name="Experimento Generico", 
                   export_csv=None, print_console=True):
    """
    Ejecuta la comparación de optimización de portafolios con los datos proporcionados.
    Retorna un pandas.DataFrame con todas las métricas.
    """
    
    # Configuración de valores por defecto
    X = np.array(X)
    n_samples, n_assets = X.shape

    if asset_names is None:
        asset_names = [f"Asset {i}" for i in range(n_assets)]
    
    if len(asset_names) != n_assets:
        if print_console:
            print(f"Advertencia: longitud de asset_names ({len(asset_names)}) no coincide con n_assets ({n_assets}). Truncando o rellenando.")
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
            ("Float8_e4m3fn", Float8_e4m3fn_Wrapper),
            ("Float8_e5m2", Float8_e5m2_Wrapper),
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
            ("Posit64", posit.Posit64)
        ]
        
    if solver_params is None:
        solver_params = {}

    all_metrics = []

    # Matriz Covarianza Base
    cov = np.cov(X, rowvar=False)

    # Ejecutar Base Skfolio
    start = time.time()
    try:
        model_sk = MeanRisk(risk_aversion=1.0, objective_function=ObjectiveFunction.MINIMIZE_RISK) 
        model_sk.fit(X)
        weights_sk = model_sk.weights_
        time_sk = time.time() - start
        risk_sk = weights_sk @ cov @ weights_sk
    except Exception as e:
        if print_console:
            print(f"  Skfolio Falló: {e}")
        weights_sk = np.zeros(n_assets)
        time_sk = 0.0
        risk_sk = 0.0

    all_metrics.append({
        'Dataset': dataset_name,
        'Tolerance': solver_params.get('tolerance', 1e-6) if solver_params else 1e-6,
        'Golden_Zone': scale_to_golden_zone,
        'Scaling_Strategy': 'N/A',
        'Scaling_Factor': 1.0,
        'Number_Type': 'Skfolio (Float64)',
        'Time_s': time_sk,
        'Iterations': None,
        'Error_L2': 0.0,
        'Risk_Variance': risk_sk,
        'Sum_Weights': np.sum(weights_sk),
        'Negativity_Violation': np.sum(np.abs(np.minimum(weights_sk, 0))),
        'Max_Abs_Diff': 0.0,
        'Risk_Gap_Pct': 0.0,
        'Grad_Zero_Detected': False,
        'Weights_Array': weights_sk.tolist()
    })

    # Ejecutar Estrategias Ad Hoc
    for scale_type, scale_factor in scaling_strategies:
        for name, number_type in number_types:
            start = time.time()
            try:
                model = PositMeanVariance(
                    number_type=number_type,
                    scaling_type=scale_type,
                    scaling_factor=scale_factor,
                    scale_to_golden_zone=scale_to_golden_zone,
                    **solver_params
                )
                
                # Callback para detectar gradiente cero
                model._grad_zero_detected = False
                def monitor_grad(w, g, i):
                    if not model._grad_zero_detected:
                        if all(val == 0.0 for val in g):
                            model._grad_zero_detected = True
                
                model.monitor_callback = monitor_grad
                model.fit(X)
                
                weights = model.weights_
                iterations = model.n_iter_
                elapsed = time.time() - start
                
                # Compute metrics
                diff = np.linalg.norm(weights_sk - weights)
                risk = weights @ cov @ weights
                sum_w = np.sum(weights)
                neg_w = np.sum(np.abs(np.minimum(weights, 0)))
                mad = np.max(np.abs(weights_sk - weights))
                
                if risk_sk > 0:
                    gap = (risk - risk_sk) / risk_sk * 100
                else:
                    gap = float('nan')
                    
                all_metrics.append({
                    'Dataset': dataset_name,
                    'Tolerance': solver_params.get('tolerance', 1e-6) if solver_params else 1e-6,
                    'Golden_Zone': scale_to_golden_zone,
                    'Scaling_Strategy': scale_type,
                    'Scaling_Factor': scale_factor,
                    'Number_Type': name,
                    'Time_s': elapsed,
                    'Iterations': iterations,
                    'Error_L2': diff,
                    'Risk_Variance': risk,
                    'Sum_Weights': sum_w,
                    'Negativity_Violation': neg_w,
                    'Max_Abs_Diff': mad,
                    'Risk_Gap_Pct': gap,
                    'Grad_Zero_Detected': model._grad_zero_detected,
                    'Weights_Array': weights.tolist()
                })
                
            except Exception as e:
                all_metrics.append({
                    'Dataset': dataset_name,
                    'Tolerance': solver_params.get('tolerance', 1e-6) if solver_params else 1e-6,
                    'Golden_Zone': scale_to_golden_zone,
                    'Scaling_Strategy': scale_type,
                    'Scaling_Factor': scale_factor,
                    'Number_Type': name,
                    'Time_s': None,
                    'Iterations': None,
                    'Error_L2': None,
                    'Risk_Variance': None,
                    'Sum_Weights': None,
                    'Negativity_Violation': None,
                    'Max_Abs_Diff': None,
                    'Risk_Gap_Pct': None,
                    'Grad_Zero_Detected': False,
                    'Weights_Array': None
                })

    # Construir DataFrame
    results_df = pd.DataFrame(all_metrics)

    # Llamar visualizadores de acuerdo a las peticiones
    if print_console:
        print_comparison_results(results_df, asset_names, n_samples)
        
    if export_csv:
        export_results_to_csv(results_df, export_csv)

    return results_df


if __name__ == "__main__":
    # Ejemplo de uso    
    np.random.seed(42)
    n_samples, n_assets = 100, 5
    X_dummy = np.random.randn(n_samples, n_assets) * 0.01 + 0.0005
    asset_ids = [f"Stock {chr(65+i)}" for i in range(n_assets)]
    
    custom_scaling = [
        ('none', 1.0),
        ('manual', 1000.0)
    ]
    
    df = run_comparison(X_dummy, asset_names=asset_ids, scaling_strategies=custom_scaling, solver_params={"objective_function": "MINIMIZE_RISK"}, print_console=True)

