#!/bin/bash

# Script de compilación para la biblioteca Posit

# Limpiar compilaciones anteriores
rm -rf build
mkdir -p build
cd build

# Configurar CMake
cmake ../cpp_extension \
    -DPython3_EXECUTABLE=$(which python3) \
    -DCMAKE_BUILD_TYPE=Release

# Compilar
cmake --build . --config Release -j$(nproc)

# Instalar
echo "Installing extension to posit_lib/..."
mv ../cpp_extension/posit*.so ../posit_lib/posit.so

cd ..
echo "Build complete! Extension installed as posit_lib.posit"
