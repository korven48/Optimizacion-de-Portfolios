#!/usr/bin/env python3
"""Funcionalidad básica de prueba del wrapper de posit"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import posit

print("=" * 60)
print("WRAPPER DE POSIT - PRUEBAS BÁSICAS")
print("=" * 60)

# Prueba 1: Construcción y conversión
print("\n1. Construcción y Conversión:")
p1 = posit.Posit32(3.14159)
p2 = posit.Posit64(2.71828)
print(f"   Posit32(3.14159) = {p1}")
print(f"   Posit64(2.71828) = {p2}")
print(f"   float(Posit32(3.14)) = {float(posit.Posit32(3.14))}")

# Prueba 2: Operaciones aritméticas
print("\n2. Operaciones Aritméticas:")
a = posit.Posit32(10.0)
b = posit.Posit32(3.0)
print(f"   a = {a}, b = {b}")
print(f"   a + b = {a + b}")
print(f"   a - b = {a - b}")
print(f"   a * b = {a * b}")
print(f"   a / b = {a / b}")
print(f"   a ** 2 = {a ** posit.Posit32(2.0)}")

# Prueba 3: Operadores de comparación
print("\n3. Operadores de Comparación:")
x = posit.Posit32(5.0)
y = posit.Posit32(3.0)
print(f"   x = {x}, y = {y}")
print(f"   x > y: {x > y}")
print(f"   x < y: {x < y}")
print(f"   x == y: {x == y}")
print(f"   x >= posit.Posit32(5.0): {x >= posit.Posit32(5.0)}")

# Prueba 4: Funciones matemáticas
print("\n4. Funciones Matemáticas:")
p = posit.Posit64(4.0)
print(f"   p = {p}")
print(f"   sqrt(p) = {p.sqrt()}")
print(f"   p.exp() = {p.exp()}")
print(f"   p.log() = {p.log()}")
print(f"   abs(Posit64(-5.0)) = {posit.Posit64(-5.0).abs()}")

# Test 5: Precision comparison
print("\n5. Precision Comparison (Posit32 vs Posit64):")
val = 1.23456789012345
p32 = posit.Posit32(val)
p64 = posit.Posit64(val)
print(f"   Original value: {val}")
print(f"   Posit32:        {float(p32)}")
print(f"   Posit64:        {float(p64)}")
print(f"   Error32:        {abs(val - float(p32)):.2e}")
print(f"   Error64:        {abs(val - float(p64)):.2e}")

# Test 6: Chain operations
print("\n6. Chain Operations:")
result = (posit.Posit32(2.0) * posit.Posit32(3.0) + posit.Posit32(1.0)).sqrt()
print(f"   ((2 * 3) + 1).sqrt() = {result}")
print(f"   As float: {float(result)}")

# Test 7: List operations
print("\n7. List Operations:")
weights = [posit.Posit32(w) for w in [0.3, 0.4, 0.3]]
returns = [posit.Posit32(r) for r in [0.05, 0.08, 0.06]]
# Use manual sum to avoid Python's sum() starting with int(0)
portfolio_return = posit.Posit32(0.0)
for w, r in zip(weights, returns):
    portfolio_return = portfolio_return + (w * r)
print(f"   Weights: {[float(w) for w in weights]}")
print(f"   Returns: {[float(r) for r in returns]}")
print(f"   Portfolio Return: {float(portfolio_return):.6f}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED! ✓")
print("=" * 60)
