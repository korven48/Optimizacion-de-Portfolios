cmake_minimum_required(VERSION 3.15)
project(posit_wrapper)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Buscar Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Buscar pybind11
find_package(pybind11 REQUIRED)

# Ruta a la biblioteca Universal
set(UNIVERSAL_DIR "/usr/local/include" CACHE PATH "Path to universal library")
set(UNIVERSAL_LOCAL_DIR "/home/lucas/Desktop/universal/include" CACHE PATH "Path to local universal library")

# Directorios de inclusión
include_directories(${UNIVERSAL_DIR})
include_directories(${UNIVERSAL_LOCAL_DIR})

# Crear módulo de Python
pybind11_add_module(posit src/posit_wrapper.cpp)

# Configurar directorio de salida
set_target_properties(posit PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Opciones de compilación para optimización
target_compile_options(posit PRIVATE
    -O3
    -march=native
    -ffast-math
)
