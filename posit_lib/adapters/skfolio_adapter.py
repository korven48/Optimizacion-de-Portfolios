#!/usr/bin/env python3
"""
Adaptador Skfolio para el Solver Posit

Este módulo proporciona una clase `PositMeanVariance` que imita la API de
`skfolio.optimization.MeanVariance`, permitiendo a los usuarios usar patrones
estándar de estilo scikit-learn mientras ejecutan la optimización completamente en posits.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import posit_lib.posit as posit
from posit_lib.solver import PGDSolver

class PositMeanVariance:
    """
    Estimador de Optimización Media-Varianza usando Aritmética Posit Pura.
    
    Parámetros
    ----------
    risk_aversion : float, default=1.0
        Coeficiente de aversión al riesgo. Valores más altos minimizan la varianza más agresivamente.
    
    objective_function : str, default='MINIMIZE_RISK'
        Función objetivo a optimizar. Opciones: 'MINIMIZE_RISK', 'MAXIMIZE_RETURN', 'MAXIMIZE_UTILITY', 'MAXIMIZE_RATIO'.
        
    number_type : class, default=None
        Clase del tipo numérico a usar para los cálculos (ej. posit.Posit64). 
        Si es None, se usa posit.Posit64 por defecto.
        
    Atributos
    ----------
    weights_ : ndarray de forma (n_assets,)
        Pesos del portafolio óptimo.
    """
    def __init__(self, 
                 risk_aversion=1.0, 
                 objective_function='MINIMIZE_RISK', 
                 number_type=None, 
                 scaling_type='none', 
                 scaling_factor=1.0,
                 max_iterations=1000,
                 tolerance=1e-6,
                 momentum=0.9,
                 learning_rate=0.1):
        """
        Args:
            risk_aversion: Parámetro de aversión al riesgo (gamma).
            objective_function: 'MINIMIZE_RISK', 'MAXIMIZE_RETURN', 'MAXIMIZE_UTILITY', 'MAXIMIZE_RATIO'
            number_type: Clase numérica a usar (ej. posit.Posit64, FloatWrapper). Por defecto usa Posit64.
            scaling_type: 'none', 'manual', 'auto_max_abs'. Estrategia de escalado de datos.
            scaling_factor: Factor de escalado manual (usado si scaling_type='manual').
            max_iterations: Número máximo de iteraciones para el solver.
            tolerance: Tolerancia de convergencia.
            momentum: Factor de momentum para el solver.
            learning_rate: Tasa de aprendizaje base.
        """
        self.risk_aversion = risk_aversion
        self.objective_function = objective_function
        self.number_type = number_type if number_type is not None else posit.Posit64
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor
        
        # Solver params
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.momentum = momentum
        self.learning_rate = learning_rate
        
        self.weights_ = None
        self._solver = None
        
    def fit(self, X, y=None):
        """
        Ajusta el modelo usando aritmética Posit o Float.
        """
        # 1. Convertir entrada a array numpy si es necesario
        X = np.array(X)
        
        # 2. Lógica de Escalado
        scale = 1.0
        if self.scaling_type == 'manual':
            scale = self.scaling_factor
        elif self.scaling_type == 'auto_max_abs':
            max_val = np.max(np.abs(X))
            if max_val > 0:
                scale = 1.0 / max_val
        
        X_scaled = X * scale
        
        n_samples, n_assets = X_scaled.shape
        
        # Calcular Mu y Cov
        # Si number_type es FloatWrapper, usamos numpy por velocidad.
        # Si es Posit, usamos la implementación en posit_lib.statistics para pureza.
        
        if self.number_type.__name__ == "FloatWrapper":
            mu = np.mean(X_scaled, axis=0)
            cov = np.cov(X_scaled, rowvar=False)
            # Convertir a listas para consistencia con lo que espera el solver si usara Posits
            # (aunque el solver maneja arrays numpy si son floats)
        else:
            print(f"Calculando estadísticas con {self.number_type.__name__} (esto puede tardar)...")
            from posit_lib.statistics import compute_mean, compute_covariance
            
            # Calcular media
            mu_posit = compute_mean(X_scaled, self.number_type)
            # Convertir a lista de floats para pasar a kwargs si es necesario, 
            # OJO: El solver espera listas de objetos number_type o floats?
            # El solver convierte internamente: self.number_type(val).
            # Si pasamos objetos Posit, el constructor de PositWrapper debe aceptar PositWrapper (copy constructor).
            # Asumamos que el wrapper tiene constructor de copia o que pasamos floats.
            
            # Para máxima precisión, deberíamos pasar los objetos Posit directamente al solver.
            # Pero PGDSolver.__init__ no recibe datos. solve() recibe kwargs.
            # solve() usa: expected_returns = [self.number_type(x) for x in kwargs['expected_returns']]
            # Si x ya es number_type, esto debe funcionar.
            
            mu = mu_posit
            
            # Calcular covarianza
            if self.objective_function in ['MINIMIZE_RISK', 'MAXIMIZE_UTILITY', 'MAXIMIZE_RATIO']:
                cov_posit = compute_covariance(X_scaled, mu_posit, self.number_type)
                cov = cov_posit
            else:
                cov = None # No se necesita
        
        # Inicializar Solver Genérico con el tipo inyectado
        solver = PGDSolver(self.number_type)
        
        # Ajuste Automático de Learning Rate por Escala
        # Si X aumenta k, Grad ~ k^2 * w. Necesitamos reducir lr en 1/k^2
        adjusted_lr = self.learning_rate / (scale ** 2)
        
        # Despachar llamadas específicas a solve()
        # NOTA: PGDSolver.solve ahora toma argumentos explícitos, no kwargs para todo.
        
        cov_arg = None
        mu_arg = None
        
        if self.objective_function == 'MINIMIZE_RISK':
            if isinstance(cov, np.ndarray):
                cov_arg = cov.tolist()
            else:
                cov_arg = cov
                
        elif self.objective_function == 'MAXIMIZE_RETURN':
            if isinstance(mu, np.ndarray):
                mu_arg = mu.tolist()
            else:
                mu_arg = mu
            
        elif self.objective_function in ['MAXIMIZE_UTILITY', 'MAXIMIZE_RATIO']:
             if isinstance(cov, np.ndarray): cov_arg = cov.tolist() 
             else: cov_arg = cov
             
             if isinstance(mu, np.ndarray): mu_arg = mu.tolist()
             else: mu_arg = mu

        
        weights_p, iterations = solver.solve(
            objective_type=self.objective_function,
            cov_matrix=cov_arg,
            expected_returns=mu_arg,
            risk_aversion=self.risk_aversion,
            max_iterations=self.max_iterations,
            learning_rate=adjusted_lr,
            tolerance=self.tolerance,
            momentum=self.momentum,
            callback=getattr(self, 'monitor_callback', None)
        )
        
        # 4. Almacenar resultados (convertir de nuevo a float para compatibilidad)
        self.weights_ = np.array([float(w) for w in weights_p])
        self.n_iter_ = iterations
        self.effective_scale_ = scale
        
        return self
    
    def predict(self, X):
        """
        Predice el retorno del portafolio para los datos dados.
        
        Parámetros
        ----------
        X : array-like de forma (n_samples, n_assets)
        
        Retorna
        -------
        y_pred : ndarray de forma (n_samples,)
            Retornos del portafolio.
        """
        if self.weights_ is None:
            raise ValueError("Modelo no ajustado aún. Llame a 'fit' primero.")
            
        return np.dot(X, self.weights_)
        
    def score(self, X, y=None):
        """
        Retorna el Ratio de Sharpe del portafolio.
        """
        returns = self.predict(X)
        return np.mean(returns) / np.std(returns)

# Ejemplo de Uso si se ejecuta directamente
if __name__ == "__main__":
    print("Prueba del Adaptador PositMeanVariance")
    print("-" * 30)
    
    # Datos Sintéticos (100 días, 4 activos)
    np.random.seed(42)
    returns = np.random.randn(100, 4) * 0.01 + 0.0005
    
    # Inicializar Modelo
    model = PositMeanVariance(risk_aversion=1.0)
    
    # Ajustar
    print("Ajustando modelo...")
    model.fit(returns)
    
    print("Pesos Óptimos:", model.weights_)
    print("Suma de pesos:", np.sum(model.weights_))
