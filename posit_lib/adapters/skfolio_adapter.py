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
    """
    def __init__(self, 
                 risk_aversion=1.0, 
                 objective_function='MINIMIZE_RISK', 
                 number_type=None, 
                 scaling_type='none', 
                 scaling_factor=1.0,
                 max_iterations=10000,
                 tolerance=1e-6,
                 momentum=0.9,
                 learning_rate=0.1):
        """
        Args:
            risk_aversion: Parámetro de aversión al riesgo (gamma).
            objective_function: 'MINIMIZE_RISK', 'MAXIMIZE_RETURN', 'MAXIMIZE_UTILITY', 'MAXIMIZE_RATIO'
            number_type: Clase numérica a usar (ej. posit.Posit64, Float64Wrapper). Por defecto usa Posit64.
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
        
    def _compute_scaling_factor(self, X):
        """
        Calcula el factor de escala según self.scaling_type y los datos X.
        """
        if self.scaling_type == 'none':
            return 1.0
            
        if self.scaling_type == 'manual':
            return self.scaling_factor
            
        vals = np.abs(X)
        if np.max(vals) == 0:
            return 1.0
            
        if self.scaling_type in ['max', 'auto_max_abs']:
            return 1.0 / np.max(vals)
            
        elif self.scaling_type == 'std':
            sigma = np.std(vals)
            return 1.0 / sigma if sigma > 0 else 1.0
            
        elif self.scaling_type == 'frobenius':
            # Norma Frobenius de la matriz X
            norm = np.linalg.norm(vals) 
            return 1.0 / norm if norm > 0 else 1.0
            
        elif self.scaling_type == 'pow2':
            # Potencia de 2 más cercana a 1/mean(vals)
            # Útil para escalar binariamente sin perder precisión en mantisa (excepto exponente)
            avg = np.mean(vals)
            if avg > 0:
                target = 1.0 / avg
                k = np.round(np.log2(target))
                return 2.0 ** k
            return 1.0
            
        return 1.0

    def _adjust_solver_params(self, scale):
        """
        Ajusta los parámetros del solver (learning_rate, risk_aversion) según la escala y función objetivo.
        """
        adjusted_lr = self.learning_rate
        adjusted_gamma = self.risk_aversion

        if self.objective_function == 'MINIMIZE_RISK':
            # Solo término cuadrático: wT (S^2 Cov) w
            # Gradiente escala por S^2
            adjusted_lr = self.learning_rate / (scale ** 2)
            
        elif self.objective_function == 'MAXIMIZE_RETURN':
            # Solo término lineal: (S mu)T w
            # Gradiente escala por S
            adjusted_lr = self.learning_rate / scale
            
        elif self.objective_function == 'MAXIMIZE_UTILITY':
            # Maximize: (S mu)T w - (gamma/2) wT (S^2 Cov) w
            # Para equivalencia, factorizamos S: S * [ muT w - (gamma*S / 2) wT Cov w ]
            # Entonces la "Gamma efectiva" debe ser gamma / S para recuperar proporción original.
            
            adjusted_gamma = self.risk_aversion / scale
            
            # Con esta gamma ajustada, Grad total escala por S.
            # LR debe contrarrestar S.
            adjusted_lr = self.learning_rate / scale
            
        elif self.objective_function == 'MAXIMIZE_RATIO':
            # Ratio invariante? Asumimos LR lineal conservador.
            adjusted_lr = self.learning_rate / scale
            
        return adjusted_lr, adjusted_gamma

    def compute_mean_and_covariance(self, X):
        """
        Calcula la media y la covarianza de los datos. 
        Si number_type es float64, usamos numpy por velocidad.
        Si es Posit, usamos la implementación en posit_lib.statistics para pureza.
        """
        mu = None
        cov = None
        if self.number_type.__name__ in ["FloatWrapper", "Float64Wrapper"]:
            mu = np.mean(X, axis=0).tolist()
            cov = np.cov(X, rowvar=False).tolist()
        else:
            print(f"Calculando estadísticas con {self.number_type.__name__} (esto puede tardar)...")
            from posit_lib.statistics import compute_mean, compute_covariance
            
            # Calcular media
            mu = compute_mean(X, self.number_type)
            
            # Calcular covarianza
            cov = compute_covariance(X, mu, self.number_type)
        return mu, cov

    def fit(self, X, y=None):
        """
        Ajusta el modelo usando aritmética Posit o Float.
        """
        # 1. Convertir entrada a array numpy si es necesario
        X = np.array(X)
        
        # 2. Lógica de Escalado 
        scale = self._compute_scaling_factor(X)
        
        X_scaled = X * scale
        
        n_samples, n_assets = X_scaled.shape
        
        # Calcular Mu y Cov
        mu, cov = self.compute_mean_and_covariance(X_scaled)
        
        # Inicializar Solver Genérico con el tipo inyectado
        solver = PGDSolver(self.number_type)
        
        # Ajuste de Parámetros según Escala y Función Objetivo
        adjusted_lr, adjusted_gamma = self._adjust_solver_params(scale) # TODO: Realmente no sé como de util es
        
        weights_p, iterations = solver.solve(
            objective_type=self.objective_function,
            cov_matrix=cov,
            expected_returns=mu,
            risk_aversion=adjusted_gamma,
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
        
        Devuelve
        -------
        y_pred : ndarray de forma (n_samples,)
            Retornos del portafolio.
        """
        if self.weights_ is None:
            raise ValueError("Modelo no ajustado aún. Llame a 'fit' primero.")
            
        return np.dot(X, self.weights_)
        
    def score(self, X, y=None):
        """
        Devuelve el Ratio de Sharpe del portafolio.
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
