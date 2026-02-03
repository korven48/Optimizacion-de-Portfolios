#!/usr/bin/env python3
import sys
import os
import yfinance as yf
import pandas as pd
import numpy as np

# Añadir directorio padre al path para encontrar posit_lib y ejemplos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from examples.custom_comparison import run_comparison

def load_real_data():
    assets_map = {
        'GLD': 'Oro',
        'BTC-USD': 'Bitcoin',
        'SPY': 'S&P 500',
        'QQQ': 'Nasdaq',
        'TLT': 'Bonos Tesoro',
        'VNQ': 'Inmobiliario',
        'EEM': 'Emergentes',
        'USO': 'Petróleo',
        'LQD': 'Bonos Corp',
        'UUP': 'Dólar'
    }
    tickers = list(assets_map.keys())
    
    print(f"Descargando datos para {len(tickers)} activos...")
    raw_data = yf.download(tickers, start="2018-01-01", end="2026-01-01", progress=False, auto_adjust=True, interval="1mo")
    
    if 'Close' in raw_data:
        data = raw_data['Close']
    elif 'Adj Close' in raw_data:
        data = raw_data['Adj Close']
    else:
        data = raw_data
    
    returns = data.pct_change().dropna()
    X = returns.values
    
    # Limpiar y mapear nombres de tickers
    sorted_tickers = returns.columns.tolist()
    clean_tickers = []
    for t in sorted_tickers:
        if isinstance(t, tuple):
            clean_tickers.append(t[0])
        else:
            clean_tickers.append(t)
            
    assets = [assets_map.get(t, t) for t in clean_tickers]
    
    return X, assets

if __name__ == "__main__":
    try:
        X, assets = load_real_data()
        
        # Estrategias originales
        strategies = [
            ('none', 1.0)
        ]
        
        run_comparison(X, asset_names=assets, scaling_strategies=strategies)
        
    except Exception as e:
        print(f"Error fatal: {e}")
