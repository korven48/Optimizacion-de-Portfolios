# Posit Portfolio Optimization Library

Este repositorio contiene una implementación de optimización de carteras usando la aritmética Posit, diseñado para comparar su precisión y estabilidad frente a la aritmética de punto flotante estándar (IEEE 754). Utiliza `skfolio` como base para la optimización de carteras y una extensión en C++ para la aritmética Posit.

## Estructura del Proyecto

- `posit_lib/`: Librería principal de Python.
- `cpp_extension/`: Código fuente C++ para la aritmética Posit y wrappers de Python (pybind11).
- `tests/`: Scripts de prueba y comparación (ej. `ill_conditioned_comparison.py`).
- `examples/`: Ejemplos de uso.

## Guía de Instalación y Uso

Sigue estos pasos para configurar el entorno y ejecutar los experimentos.

### Prerrequisitos

Asegúrate de tener instalado lo siguiente en tu sistema:

*   **Python 3.8+**
*   **Compilador C++ compatible con C++20** (GCC 10+, Clang 10+, MSVC 19.28+)
*   **CMake 3.15+**
*   **Universal Number Library**: La extensión asume que la librería Universal está disponible o sus headers pueden ser incluidos. Por defecto busca en `/usr/local/include`.

### 1. Configurar Entorno Virtual

Se recomienda usar un entorno virtual para aislar las dependencias.

```bash
# Crear entorno virtual
python3 -m venv .venv

# Activar entorno virtual
source .venv/bin/activate
```

### 2. Instalar Dependencias de Python

Instala las librerías necesarias listadas en `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Compilar la Extensión C++ (Posit Wrapper)

Para usar la aritmética Posit, necesitas compilar la extensión nativa. Se proporciona un script `build.sh` para facilitar este proceso.

```bash
# Dar permisos de ejecución al script si es necesario
chmod +x build.sh

# Ejecutar el script de construcción
./build.sh
```

Este script:
1.  Limpiará construcciones previas.
2.  Configurará CMake buscando la librería Universal en `/usr/local/include` (puedes editar `cpp_extension/CMakeLists.txt` si tu ruta es diferente).
3.  Compilará el módulo `posit` optimizado para tu arquitectura.
4.  Instalará el archivo `.so` resultante en `posit_lib/`.

### 4. Ejecutar Tests y Comparaciones

El directorio `tests/` contiene varios scripts para evaluar y comparar la precision numerica de distintos formatos aritmeticos en la optimizacion de portafolios.

#### 4.1 Motor de Comparacion (`tests/custom_comparison.py`)

Este es el modulo central que ejecuta las comparaciones. Proporciona la funcion `run_comparison()` que:

1.  Optimiza un portafolio con **Skfolio** (Float64, referencia).
2.  Repite la optimizacion con el solver propio usando cada combinacion de tipo numerico y estrategia de escalado.
3.  Calcula metricas de calidad: Error L2, brecha de riesgo, suma de pesos, negatividad, etc.

**Uso directo** (con datos sinteticos de ejemplo):

```bash
python3 tests/custom_comparison.py
```

**Uso programatico** (importando la funcion):

```python
from tests.custom_comparison import run_comparison

df = run_comparison(
    X,                                # Matriz de retornos (n_samples x n_assets)
    asset_names=["Activo1", ...],     # Nombres de activos (opcional)
    scaling_strategies=[('std', 1.0)],# Estrategias de escalado
    number_types=None,                # None = todos los tipos disponibles
    solver_params={'tolerance': 1e-6},# Parametros del solver
    scale_to_golden_zone=False,       # Escalar a zona dorada Posit
    export_csv="resultados.csv",      # Exportar a CSV (opcional)
    print_console=True                # Mostrar resultados por consola
)
```

**Tipos numericos soportados:**

| Familia IEEE 754       | Familia Posit            |
|------------------------|--------------------------|
| Float8_e4m3fn          | Posit8                   |
| Float8_e5m2            | Posit12                  |
| Float16                | Posit16                  |
| BFloat16               | Posit20                  |
| Float32                | Posit24                  |
| Float64                | Posit32                  |
|                        | Posit64                  |

**Estrategias de escalado disponibles:** `none`, `manual`, `max`, `std`, `frobenius`, `pow2`.

---

#### 4.2 Comparacion con Activos Reales (`tests/real_asset_comparison.py`)

Descarga datos historicos de Yahoo Finance para un portafolio diversificado de 10 activos reales y ejecuta la comparacion.

**Activos incluidos:** Oro (GLD), Bitcoin (BTC-USD), S&P 500 (SPY), Nasdaq (QQQ), Bonos Tesoro (TLT), Inmobiliario (VNQ), Mercados Emergentes (EEM), Petroleo (USO), Bonos Corporativos (LQD), Dolar (UUP).

```bash
python3 tests/real_asset_comparison.py
```

> **Nota:** Requiere conexion a internet para descargar datos via `yfinance`. Los datos por defecto abarcan de 2018-01-01 a 2026-01-01 con frecuencia mensual.

---

#### 4.3 Comparacion con Matrices Mal Condicionadas (`tests/ill_conditioned_comparison.py`)

Genera datos sinteticos disenados para estresar la estabilidad numerica (matrices con alta correlacion y numeros diminutos) y compara el rendimiento de los distintos tipos numericos.

```bash
python3 tests/ill_conditioned_comparison.py
```

---

#### 4.4 Grid Search Completo (`tests/grid_search.py`)

Ejecuta una busqueda exhaustiva de hiperparametros, combinando:

*   **4 datasets:** Alta correlacion sintetica, numeros diminutos, datos reales mensuales y datos reales diarios.
*   **6 tolerancias:** `1e-3`, `1e-4`, `1e-5`, `1e-6`, `1e-7`, `1e-8`.
*   **6 estrategias de escalado:** `none`, `manual(100)`, `max`, `std`, `frobenius`, `pow2`.
*   **2 opciones de Golden Zone:** `True` / `False`.
*   **13 tipos numericos:** 6 IEEE 754 + 7 Posit.

```bash
python3 tests/grid_search.py
```

Los resultados se guardan en `tests/full_grid_search_resultados.csv`.

---

#### 4.5 Resultados del Grid Search (`tests/full_grid_search_resultados.csv`)

Archivo CSV con los resultados completos del grid search (~3400 filas). Columnas:

| Columna                | Descripcion                                                     |
|------------------------|-----------------------------------------------------------------|
| `Dataset`              | Nombre del dataset utilizado                                    |
| `Tolerance`            | Tolerancia de convergencia del solver                           |
| `Golden_Zone`          | Si se escalo a la zona dorada del formato Posit                 |
| `Scaling_Strategy`     | Estrategia de escalado aplicada                                 |
| `Scaling_Factor`       | Factor multiplicativo de la estrategia                          |
| `Number_Type`          | Tipo numerico utilizado (ej. `Posit16`, `Float32`)              |
| `Time_s`               | Tiempo de ejecucion en segundos                                 |
| `Iterations`           | Numero de iteraciones del solver                                |
| `Error_L2`             | Norma L2 de la diferencia de pesos vs. Skfolio                  |
| `Risk_Variance`        | Varianza del portafolio resultante (w^T * Cov * w)              |
| `Sum_Weights`          | Suma de los pesos (debe ser ~1.0)                               |
| `Negativity_Violation` | Suma de violaciones de no-negatividad                           |
| `Max_Abs_Diff`         | Diferencia absoluta maxima de un peso vs. Skfolio               |
| `Risk_Gap_Pct`         | Brecha porcentual de riesgo respecto a Skfolio                  |
| `Grad_Zero_Detected`   | Si el gradiente colapso a cero (underflow)                      |
| `Weights_Array`        | Vector de pesos del portafolio resultante                       |