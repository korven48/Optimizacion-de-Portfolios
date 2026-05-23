#!/usr/bin/env python3
"""
Solver para Optimización de Portafolios

Este módulo implementa una clase PGDSolver reutilizable que utiliza
Descenso de Gradiente Proyectado (PGD) para resolver problemas de optimización.
Es genérico y puede trabajar con tipos Posit o Floats estándar (vía Float64Wrapper).
"""

from . import posit as posit
from .float_wrapper import Float64Wrapper

class PGDSolver:
    def __init__(self, number_type):
        self.number_type = number_type
        self.zero = self.number_type(0.0)
        self.one = self.number_type(1.0)

    def _dot_product(self, v1, v2):
        """
        Producto punto que usa el quire (acumulación exacta) cuando el tipo
        numérico es posit, o la acumulación estándar en caso contrario (floats).
        """
        if hasattr(self.zero, 'dot_product_quire'):
            return self.zero.dot_product_quire(v1, v2)
        result = self.zero
        for a, b in zip(v1, v2):
            result = result + (a * b)
        return result

    # def _dot_product_standard(self, v1, v2):
    #     result = self.zero
    #     for a, b in zip(v1, v2):
    #         result = result + (a * b)
    #     return result

    def _matrix_vector_product(self, matrix, vector):
        """Calcula el producto matriz-vector (matrix @ vector).
        Usa el quire para tipos posit (acumulación exacta) o la suma estándar para floats."""
        result = []
        for row in matrix:
            result.append(self._dot_product(row, vector))
        return result

    def _projection_simplex(self, weights, target_sum=None):
        """
        Proyecta los pesos sobre el simplex escalado (sum(w) = target_sum, w >= 0).
        Algoritmo: Proyección basada en ordenamiento.
        """
        target_sum_val = self.one if target_sum is None else target_sum
        n = len(weights)
        
        # Ordena pesos en orden descendente
        sorted_weights = sorted(weights, reverse=True)
        
        # Encuentra rho
        tmpsum = self.zero
        rho = -1
        
        # Itera de 0 a n-1 actualizando la suma parcial (tmpsum) en cada paso.
        # La condición u + (1 - tmpsum) / (i+1) > 0 determina si el peso u
        # puede ser parte de la solución activa (rho) sin violar las restricciones.
        
        for i in range(n):
            u = sorted_weights[i]
            tmpsum = tmpsum + u

            divisor = self.number_type(float(i + 1))
            val = u + (target_sum_val - tmpsum) / divisor
            
            if val > self.zero:
                rho = i
        
        # Calcula lambda = (1 - sum(sorted_weights[:rho+1])) / (rho + 1)
        sum_rho = self.zero
        for i in range(rho + 1):
            sum_rho = sum_rho + sorted_weights[i]
            
        divisor_rho = self.number_type(float(rho + 1))
        lambda_val = (target_sum_val - sum_rho) / divisor_rho
        
        # Calcula los pesos finales: w = max(v + lambda, 0)
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
            cov_w = self._matrix_vector_product(self._cov_p, w)
            two = self.number_type(2.0)
            for val in cov_w:
                grad.append(two * val)
                
        elif objective_type == 'MAXIMIZE_RETURN':
            grad = [val for val in self._mu_p]
            
        elif objective_type == 'MAXIMIZE_UTILITY':
            # Grad = mu - gamma * Cov * w
            cov_w = self._matrix_vector_product(self._cov_p, w)
            for i in range(self._n_assets):
                term2 = self._gamma_p * cov_w[i]
                grad.append(self._mu_p[i] - term2)
                
        elif objective_type == 'MAXIMIZE_RATIO':
            pass # No implementado
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
              callback=None,
              scale_to_golden_zone=False):
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
        
        # Inicializar pesos para que la suma sea 1 o N según scale_to_golden_zone
        if scale_to_golden_zone:
            initial_weight = self.one
            t_sum = self.number_type(float(self._n_assets))
        else:
            initial_weight = self.number_type(1.0 / self._n_assets)
            t_sum = self.one
            
        w = [initial_weight for _ in range(self._n_assets)]
        
        # Inicializa velocidad para Momentum
        velocity = [self.zero for _ in range(self._n_assets)]
        
        lr = self.number_type(learning_rate)
        if scale_to_golden_zone:
            tol = self.number_type(tolerance * float(self._n_assets))
        else:
            tol = self.number_type(tolerance)
            
        mu = self.number_type(momentum)
        
        for i in range(max_iterations):
            grad = self._compute_gradient(w, objective_type)
            
            # Si hay un callback, se llama antes de la actualización
            if callback:
                w_float = [float(val) for val in w]
                grad_float = [float(val) for val in grad]
                callback(w_float, grad_float, i)

            # Si el gradiente es cero, se detiene el algoritmo
            is_zero_gradient = all(g == self.zero for g in grad)
            if is_zero_gradient:
                if scale_to_golden_zone:
                    dev = self.number_type(float(self._n_assets))
                    w = [val / dev for val in w]
                return w, i + 1
            
            # Actualización con Momentum
            w_step = []
            for j in range(self._n_assets):
                g = grad[j]
                if objective_type not in ['MAXIMIZE_RETURN', 'MAXIMIZE_UTILITY', 'MAXIMIZE_RATIO']:
                    g = -g # Descenso
                
                # Actualiza velocidad
                velocity[j] = mu * velocity[j] + lr * g
                
                # Actualiza peso
                w_step.append(w[j] + velocity[j])
            
            # Proyección al Simplex
            w_new = self._projection_simplex(w_step, target_sum=t_sum)
            
            # Criterio de parada
            diff_sq_sum = self.number_type(0.0)
            for j in range(self._n_assets):
                diff = w_new[j] - w[j]
                diff_sq_sum += diff * diff
                
            if float(diff_sq_sum) < float(tol * tol):
                if scale_to_golden_zone:
                    dev = self.number_type(float(self._n_assets))
                    w_new = [val / dev for val in w_new]
                return w_new, i + 1
                
            w = w_new
            
        if scale_to_golden_zone:
            dev = self.number_type(float(self._n_assets))
            w = [val / dev for val in w]
            
        return w, max_iterations
