#!/bin/bash

# Build script for Posit Library

# 1. Clean previous builds
rm -rf build
mkdir -p build
cd build

# 2. Configure CMake
# We point to the cpp_extension directory
cmake ../cpp_extension \
    -DPython3_EXECUTABLE=$(which python3) \
    -DCMAKE_BUILD_TYPE=Release

# 3. Build
cmake --build . --config Release -j$(nproc)

# 4. Install (Copy .so to the python package)
echo "Installing extension to posit_lib/..."
cp ../cpp_extension/posit*.so ../posit_lib/posit.so

cd ..
echo "Build complete! Extension installed as posit_lib.posit"
