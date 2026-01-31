import numpy as np

def compute_mean(X, number_type):
    """
    Calcula la media de cada columna de X usando aritmética del number_type dado.
    
    Args:
        X: Array-like de forma (n_samples, n_features).
        number_type: Clase numérica a usar (ej. Posit64, FloatWrapper).
        
    Returns:
        Lista de objetos number_type con las medias.
    """
    X = np.array(X)
    n_samples, n_features = X.shape
    
    means = []
    
    # Convertir n_samples a number_type para la división
    n_samples_p = number_type(n_samples)
    
    for j in range(n_features):
        sum_val = number_type(0.0)
        col = X[:, j]
        
        for val in col:
            sum_val += number_type(float(val))
            
        mean_val = sum_val / n_samples_p
        means.append(mean_val)
        
    return means

def compute_covariance(X, mu, number_type):
    """
    Calcula la matriz de covarianza de X usando aritmética del number_type dado.
    
    Args:
        X: Array-like de forma (n_samples, n_features).
        mu: Lista de medias (objetos number_type) calculada previamente.
        number_type: Clase numérica a usar.
        
    Returns:
        Lista de listas (matriz) de objetos number_type con la covarianza.
    """
    X = np.array(X)
    n_samples, n_features = X.shape
    
    if n_samples < 2:
        raise ValueError("Se requieren al menos 2 muestras para calcular la covarianza.")
        
    # Factor de normalización 1/(N-1)
    norm_factor = number_type(1.0) / number_type(n_samples - 1)
    
    # Pre-convertir X a number_type para evitar conversiones repetidas en el bucle interno?
    # Sería mucha memoria si X es grande. Mejor convertir al vuelo o por columnas.
    # Vamos a hacerlo elemento a elemento para ser estrictos con la aritmética, 
    # aunque sea lento.
    
    cov_matrix = [[number_type(0.0) for _ in range(n_features)] for _ in range(n_features)]
    
    # Centrar datos primero (X - mu)
    # Esto requiere almacenar X_centered en memoria como objetos Posit.
    X_centered = []
    for i in range(n_samples):
        row = []
        for j in range(n_features):
            val = number_type(float(X[i, j]))
            centered = val - mu[j]
            row.append(centered)
        X_centered.append(row)
        
    # Calcular Covarianza: Sum((X_centered_j) * (X_centered_k)) / (N-1)
    # Aprovechamos simetría
    for j in range(n_features):
        for k in range(j, n_features):
            sum_prod = number_type(0.0)
            for i in range(n_samples):
                # sum += X_centered[i][j] * X_centered[i][k]
                prod = X_centered[i][j] * X_centered[i][k]
                sum_prod += prod
            
            cov_val = sum_prod * norm_factor
            cov_matrix[j][k] = cov_val
            if j != k:
                cov_matrix[k][j] = cov_val
                
    return cov_matrix
