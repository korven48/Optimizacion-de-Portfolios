cmake_minimum_required(VERSION 3.15)
project(posit_wrapper)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find pybind11
find_package(pybind11 REQUIRED)

# Universal library path
set(UNIVERSAL_DIR "/usr/local/include" CACHE PATH "Path to universal library")

# Include directories
include_directories(${UNIVERSAL_DIR})

# Create Python module
pybind11_add_module(posit src/posit_wrapper.cpp)

# Set output directory
set_target_properties(posit PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Compiler flags for optimization
target_compile_options(posit PRIVATE
    -O3
    -march=native
    -ffast-math
)
