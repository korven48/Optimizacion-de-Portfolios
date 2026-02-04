#!/usr/bin/env python3
"""
Solver Genérico Robusto para Optimización de Portafolios

Este módulo implementa una clase PGDSolver reutilizable que utiliza
Descenso de Gradiente Proyectado (PGD) para resolver problemas de optimización.
Es genérico y puede trabajar con tipos Posit o Floats estándar (vía Float64Wrapper).
"""

import sys
import os

# Importar la extensión C++ desde el paquete actual
from . import posit as posit
from .float_wrapper import Float64Wrapper
import math

class PGDSolver:
    """
    Solver de Descenso de Gradiente Proyectado para Optimización de Portafolios.
    Implementación genérica que puede ejecutarse en tipos Posit o Float64Wrapper.
    """
    def __init__(self, number_type):
        """
        Inicializa el solver con un tipo numérico específico.
        
        Args:
            number_type: Un posit.PositXX o FloatXXWrapper.
        """
        self.number_type = number_type
        self.zero = self.number_type(0.0)
        self.one = self.number_type(1.0)

    def _dot_product_kahan(self, v1, v2):
        """
        Calcula el producto punto usando Suma de Kahan para minimizar el error numérico.
        Algoritmo:
        1. Mantiene una variable 'c' que acumula los errores de redondeo de bajo orden.
        2. En cada paso, intenta sumar el error acumulado a la entrada actual antes de sumar al total.
        3. Esto recupera bits de precisión que normalmente se perderían al sumar un número pequeño a uno grande.
        """
        sum_val = self.zero
        c = self.zero # Compensación de error
        
        for a, b in zip(v1, v2):
            prod = a * b
            y = prod - c       # Restar el error de la suma anterior
            t = sum_val + y    # Suma estándar con la corrección
            c = (t - sum_val) - y # Calcular el nuevo error de redondeo
            sum_val = t
            
        return sum_val

    def _dot_product_standard(self, v1, v2):
        """
        Implementación estándar del producto punto (para referencia/comparación).
        Acumula el error de redondeo si las magnitudes varían mucho.
        """
        result = self.zero
        for a, b in zip(v1, v2):
            result = result + (a * b)
        return result

    def _matrix_vector_product(self, matrix, vector):
        """Calcula el producto matriz-vector (matrix @ vector) usando Kahan."""
        result = []
        for row in matrix:
            # Usar Kahan para mayor precisión en Posit16
            result.append(self._dot_product_kahan(row, vector))
        return result

    def _projection_simplex(self, weights):
        """
        Proyecta los pesos sobre el simplex de probabilidad (sum(w) = 1, w >= 0).
        Algoritmo: Proyección basada en ordenamiento.
        """
        n = len(weights)
        
        # 1. Ordenar pesos en orden descendente
        sorted_weights = sorted(weights, reverse=True)
        
        # 2. Encontrar rho
        tmpsum = self.zero
        rho = -1
        
        # Iteramos de 0 a n-1 actualizando la suma parcial (tmpsum) en cada paso.
        # La condición u + (1 - tmpsum) / (i+1) > 0 determina si el peso u
        # puede ser parte de la solución activa (rho) sin violar las restricciones.
        
        for i in range(n):
            u = sorted_weights[i]
            tmpsum = tmpsum + u
            
            # Calcular (tmpsum - 1) / (i + 1)
            # Nota: i es int, necesitamos convertir a number_type
            divisor = self.number_type(float(i + 1))
            val = u + (self.one - tmpsum) / divisor
            
            if val > self.zero:
                rho = i
        
        # 3. Calcular lambda
        # lambda = (1 - sum(sorted_weights[:rho+1])) / (rho + 1)
        sum_rho = self.zero
        for i in range(rho + 1):
            sum_rho = sum_rho + sorted_weights[i]
            
        divisor_rho = self.number_type(float(rho + 1))
        lambda_val = (self.one - sum_rho) / divisor_rho
        
        # 4. Calcular resultado: w = max(v + lambda, 0)
        result = [self.zero] * n
        for i in range(n):
            val = weights[i] + lambda_val
            if val > self.zero:
                result[i] = val
            else:
                result[i] = self.zero
                
        return result

    def _compute_gradient(self, w, objective_type):
        """
        Calcula el gradiente de la función objetivo.
        """
        grad = []
        
        if objective_type == 'MINIMIZE_RISK':
            # Gradiente de w^T * Cov * w es 2 * Cov * w
            # G = 2 * (Cov . w)
            cov_w = self._matrix_vector_product(self._cov_p, w)
            two = self.number_type(2.0)
            for val in cov_w:
                grad.append(two * val)
                
        elif objective_type == 'MAXIMIZE_RETURN':
            grad = [val for val in self._mu_p]
            
        elif objective_type == 'MAXIMIZE_UTILITY':
            # U = w^T * mu - (gamma/2) * w^T * Cov * w
            # Grad = mu - gamma * Cov * w
            cov_w = self._matrix_vector_product(self._cov_p, w)
            for i in range(self._n_assets):
                term2 = self._gamma_p * cov_w[i]
                grad.append(self._mu_p[i] - term2)
                
        elif objective_type == 'MAXIMIZE_RATIO':
            
            for i in range(n):
                term1 = (R * Sw[i]) / V3
                term2 = mu_p[i] / V
                grad.append(term1 - term2)
        else:
            raise ValueError(f"Tipo de objetivo desconocido: {objective_type}")
            
        return grad

    def _setup_problem_data(self, cov_matrix, expected_returns, risk_aversion, objective_type):
        """
        Pre-procesa los datos de entrada (cov_matrix, expected_returns, risk_aversion)
        convirtiéndolos a number_type y almacenándolos internamente.
        """
        self._n_assets = 0
        self._cov_p = None
        self._mu_p = None
        self._gamma_p = self.number_type(risk_aversion)

        if cov_matrix is not None:
            self._n_assets = len(cov_matrix)
            self._cov_p = [[self.number_type(x) for x in row] for row in cov_matrix]
            
        if expected_returns is not None:
            if self._n_assets == 0: self._n_assets = len(expected_returns)
            self._mu_p = [self.number_type(x) for x in expected_returns]
            
        if self._n_assets == 0:
            raise ValueError("Se debe proporcionar 'cov_matrix' o 'expected_returns' para determinar el tamaño del problema.")

        # Pre-cálculo para MAXIMIZE_RETURN (gradiente constante)
        if objective_type == 'MAXIMIZE_RETURN':
            if self._mu_p is None:
                raise ValueError("Para 'MAXIMIZE_RETURN', 'expected_returns' debe ser proporcionado.")
            self._grad_const_p = [-val for val in self._mu_p]

    def solve(self, 
              objective_type='MINIMIZE_RISK', 
              cov_matrix=None, 
              expected_returns=None, 
              risk_aversion=1.0, 
              max_iterations=1000, 
              learning_rate=0.1, 
              tolerance=1e-6, 
              momentum=0.9, 
              callback=None):
        """
        Resuelve el problema de optimización.
        
        Args:
            objective_type (str): Tipo de función objetivo.
            cov_matrix (list): Matriz de covarianza.
            expected_returns (list): Retornos esperados.
            risk_aversion (float): Aversión al riesgo.
            max_iterations (int): Número máximo de iteraciones.
            learning_rate (float): Tasa de aprendizaje.
            tolerance (float): Tolerancia para convergencia.
            momentum (float): Factor de momentum (0.0 a 1.0).
            callback (callable): Función llamada en cada iteración: callback(weights, gradient, iteration).
        
        Returns:
            tuple: (weights, iterations)
        """
        
        # Pre-procesamiento de datos para el solver
        self._setup_problem_data(cov_matrix, expected_returns, risk_aversion, objective_type)
        
        # Inicializar pesos (1/N)
        initial_weight = self.number_type(1.0 / self._n_assets)
        w = [initial_weight for _ in range(self._n_assets)]
        
        # Inicializar velocidad para Momentum
        velocity = [self.zero for _ in range(self._n_assets)]
        
        lr = self.number_type(learning_rate)
        tol = self.number_type(tolerance)
        mu = self.number_type(momentum)
        
        for i in range(max_iterations):
            grad = self._compute_gradient(w, objective_type)
            
            # Check for zero gradient (Underflow prevention or Stationary point)
            # If gradient is exactly zero, we cannot improve further using gradient descent.
            
            if callback:
                w_float = [float(val) for val in w]
                grad_float = [float(val) for val in grad]
                callback(w_float, grad_float, i)

            is_zero_gradient = all(g == self.zero for g in grad)
            if is_zero_gradient:
                return w, i + 1
            
            # Actualización con Momentum
            # v_{t+1} = mu * v_t + grad
            # w_{t+1} = w_t - lr * v_{t+1}
            # Nota: Si es maximización, sumamos grad. Si es minimización, restamos grad.
            # Para simplificar, calculamos el "paso" completo.
            
            w_step = []
            for j in range(self._n_assets):
                # Dirección del gradiente
                g = grad[j]
                if objective_type not in ['MAXIMIZE_RETURN', 'MAXIMIZE_UTILITY', 'MAXIMIZE_RATIO']:
                    g = -g # Descenso
                
                # Actualizar velocidad
                # v_new = mu * v_old + g
                # Ojo: Standard momentum suele ser v = mu*v - lr*grad.
                # Aquí 'g' ya tiene el signo correcto para "avanzar".
                # Vamos a usar: v = mu * v + lr * g
                
                velocity[j] = mu * velocity[j] + lr * g
                
                # Actualizar peso
                w_step.append(w[j] + velocity[j])
            
            # Proyección al Simplex
            w_new = self._projection_simplex(w_step)
            
            # Criterio de parada
            diff_sq_sum = self.number_type(0.0)
            for j in range(self._n_assets):
                diff = w_new[j] - w[j]
                diff_sq_sum += diff * diff
                
            if float(diff_sq_sum) < float(tol * tol):
                return w_new, i + 1
                
            w = w_new
            
        return w, max_iterations
