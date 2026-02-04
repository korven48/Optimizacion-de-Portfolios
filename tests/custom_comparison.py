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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Probablemente deba hacer otra cosa

from posit_lib.adapters.skfolio_adapter import PositMeanVariance
from posit_lib.float_wrapper import Float64Wrapper, Float16Wrapper, Float32Wrapper
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
    Ejecuta la comparación de optimización de portafolios con los datos proporcionados.

    Parámetros
    ----------
    X : array-like, forma (n_samples, n_assets)
        Matriz de retornos de activos.
    asset_names : lista de str, opcional
        Nombres de los activos. Por defecto ["Asset 0", "Asset 1", ...].
    scaling_strategies : lista de tuplas, opcional
        Lista de (tipo, factor). Por defecto conjunto de comparación estándar.
    number_types : lista de tuplas, opcional
        Lista de (nombre, clase). Por defecto tipos numéricos estándar.
    solver_params : dict, opcional
        Parámetros adicionales para el solver (ej. {'learning_rate': 0.1}). No incluye number_type, scaling_type y scaling_factor.
    """
    
    # 1. Configuración de valores por defecto
    X = np.array(X)
    n_samples, n_assets = X.shape

    if asset_names is None:
        asset_names = [f"Asset {i}" for i in range(n_assets)]
    
    if len(asset_names) != n_assets:
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
            ("Float16", Float16Wrapper),
            ("Float32", Float32Wrapper),
            ("Float64", Float64Wrapper),
            ("Posit8", posit.Posit8),
            ("Posit16", posit.Posit16),
            ("Posit32", posit.Posit32),
            ("Posit64", posit.Posit64)
        ]
        
    if solver_params is None:
        solver_params = {}

    print_header("COMPARACIÓN DE DATOS PERSONALIZADOS: SKFOLIO vs POSIT vs FLOAT")
    print(f"Forma de Datos: {n_samples} muestras x {n_assets} activos")
    print_separator()

    # 2. Ejecutar Base Skfolio
    print("\n[Skfolio Oficial] MeanRisk(objective_function='MINIMIZE_RISK')...")
    start = time.time()
    try:
        model_sk = MeanRisk(risk_aversion=1.0, objective_function=ObjectiveFunction.MINIMIZE_RISK) 
        model_sk.fit(X)
        weights_sk = model_sk.weights_
        time_sk = time.time() - start
        print(f"  Tiempo: {time_sk:.4f}s")
    except Exception as e:
        print(f"  Skfolio Falló: {e}")
        weights_sk = np.zeros(n_assets)
        time_sk = 0.0

    # 3. Ejecutar Estrategias
    for scale_type, scale_factor in scaling_strategies:
        print("\n" + "=" * 100)
        print(f"ESTRATEGIA DE ESCALADO: {scale_type, scale_factor}")
        print("=" * 100)
        
        results = {}

        for name, number_type in number_types:
            start = time.time()
            try:
                model = PositMeanVariance(
                    number_type=number_type,
                    scaling_type=scale_type,
                    scaling_factor=scale_factor,
                    **solver_params
                )
                
                # Callback para detectar gradiente cero
                model._grad_zero_detected = False
                def monitor_grad(w, g, i):
                    # Si la norma es 0 o todos son 0
                    if not model._grad_zero_detected:
                        # Si el gradiente es todo ceros (salvo quizás flotantes minúsculos no representables)
                        # Verificamos si todos los elementos son exactamente 0.0
                        if all(val == 0.0 for val in g):
                            model._grad_zero_detected = True
                
                model.monitor_callback = monitor_grad
                
                model.fit(X)
                
                weights = model.weights_
                iterations = model.n_iter_
                elapsed = time.time() - start
                
                results[name] = {
                    'weights': weights,
                    'time': elapsed,
                    'iters': iterations,
                    'grad_zero': model._grad_zero_detected
                }
            except Exception as e:
                # print(f"  FAILED {name}: {e}")
                results[name] = None

        # 4. Mostrar Tabla de Resultados
        print("\n")
        print_header(f"RESULTADOS {scale_type, scale_factor}")

        headers = ["Asset", "Skfolio"] + [name for name, _ in number_types]
        col_width = 12
        asset_width = 16

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

        # Métricas Globales
        print("\nMETRICAS DE RENDIMIENTO:")
        print(separator)
        metric_fmt = f"| {{:<{asset_width}}} | {{:>{col_width}}}" + f" | {{:>{col_width}}}" * len(number_types) + " |"

        # Tiempo
        time_vals = ["Time (s)", f"{time_sk:.4f}"]
        for name, _ in number_types:
            res = results.get(name)
            time_vals.append(f"{res['time']:.4f}" if res else "-")
        print(metric_fmt.format(*time_vals))

        # Iteraciones
        iter_vals = ["Iterations", "N/A"]
        for name, _ in number_types:
            res = results.get(name)
            iter_vals.append(f"{res['iters']}" if res else "-")
        print(metric_fmt.format(*iter_vals))

        # Error L2
        l2_vals = ["Error L2 (vs Sk)", "0.00e+00"]
        for name, _ in number_types:
            res = results.get(name)
            if res and res['weights'] is not None:
                diff = np.linalg.norm(weights_sk - res['weights'])
                l2_vals.append(f"{diff:.2e}")
            else:
                l2_vals.append("-")
        print(metric_fmt.format(*l2_vals))
        
        # Riesgo del Portafolio (w.T * Cov * w)
        cov = np.cov(X, rowvar=False)
        risk_sk = weights_sk @ cov @ weights_sk
        risk_vals = ["Riesgo (Var)", f"{risk_sk:.9f}"]

        for name, _ in number_types:
            res = results.get(name)
            if res and res['weights'] is not None:
                w = res['weights']
                risk = w @ cov @ w
                
                # Detectar Underflow (Riesgo Exactamente 0)
                if risk == 0.0:
                    risk_vals.append(f"{risk:0} (!)")
                elif res.get('grad_zero', False): 
                    # Añadir flag si hubo gradiente cero
                    risk_vals.append(f"{risk:.5f} [G0]")
                else:
                    risk_vals.append(f"{risk:.9f}")
            else:
                risk_vals.append("-")
        print(metric_fmt.format(*risk_vals))

        # --- Chequeos de Confiabilidad ---
        print(separator)
        
        # 1. Suma de Pesos (Validez: Debe ser 1.0)
        sum_vals = ["Suma Pesos", f"{np.sum(weights_sk):.6f}"]
        for name, _ in number_types:
            res = results.get(name)
            if res and res['weights'] is not None:
                sum_w = np.sum(res['weights'])
                sum_vals.append(f"{sum_w:.6f}")
            else:
                sum_vals.append("-")
        print(metric_fmt.format(*sum_vals))

        # 2. Violación de Restricciones (Negatividad: Debe ser 0.0)
        neg_vals = ["Negatividad", f"{np.sum(np.abs(np.minimum(weights_sk, 0))):.2e}"]
        for name, _ in number_types:
            res = results.get(name)
            if res and res['weights'] is not None:
                neg_w = np.sum(np.abs(np.minimum(res['weights'], 0)))
                neg_vals.append(f"{neg_w:.2e}")
            else:
                neg_vals.append("-")
        print(metric_fmt.format(*neg_vals))

        # 3. Brecha de Riesgo % (Eficiencia: (Riesgo_Posit - Riesgo_Sk) / Riesgo_Sk)
        gap_vals = ["Brecha Riesgo %", "0.00%"]
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

        # 4. Dif Abs Max (Similitud)
        mad_vals = ["Dif Abs Max", "0.00e+00"]
        for name, _ in number_types:
            res = results.get(name)
            if res and res['weights'] is not None:
                mad = np.max(np.abs(weights_sk - res['weights']))
                mad_vals.append(f"{mad:.2e}")
            else:
                mad_vals.append("-")
        print(metric_fmt.format(*mad_vals))
        
        print(separator)
        print("Leyenda:")
        print("  (!)  Posible Underflow detectado (Riesgo == 0.0)")
        print("  [G0] Gradiente colapsó a Cero durante la optimización (Underflow)")
        print("\n")

if __name__ == "__main__":
    print("Ejecutando Comparación Personalizada con Datos Sintéticos...")
    
    # Generar datos sintéticos aleatorios: 100 días, 5 activos
    np.random.seed(42)
    n_samples, n_assets = 100, 5
    # Retornos aleatorios aproximadamente Gaussianos alrededor de 0
    X_dummy = np.random.randn(n_samples, n_assets) * 0.01 + 0.0005
    
    asset_ids = [f"Stock {chr(65+i)}" for i in range(n_assets)]
    
    custom_scaling = [
        ('none', 1.0),
        ('manual', 1000.0)
    ]
    
    # Ejecutar comparación
    run_comparison(X_dummy, asset_names=asset_ids, scaling_strategies=custom_scaling, solver_params={"objective_function": "MINIMIZE_RISK"})
