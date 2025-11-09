"""
Quick Test Script
=================
Tests the curve fitting optimization with synthetic data.
"""

import numpy as np
from curve_fitting import ParametricCurveFitter

def test_optimization():
    """Test the optimization with synthetic data."""
    print("Testing Parametric Curve Fitting Optimization")
    print("="*60)
    
    # Create synthetic data using known parameters
    theta_true = np.deg2rad(25)
    M_true = 0.02
    X_true = 50
    
    print(f"\nTrue Parameters:")
    print(f"  θ = {np.rad2deg(theta_true):.4f}° ({theta_true:.6f} radians)")
    print(f"  M = {M_true:.6f}")
    print(f"  X = {X_true:.6f}")
    
    # Generate test data
    fitter = ParametricCurveFitter('test_data.csv')
    
    # Create synthetic data
    t_values = np.linspace(6.1, 59.9, 27)
    x_values = []
    y_values = []
    
    for t in t_values:
        x = t * np.cos(theta_true) - np.exp(M_true * abs(t)) * np.sin(0.3 * t) * np.sin(theta_true) + X_true
        y = 42 + t * np.sin(theta_true) + np.exp(M_true * abs(t)) * np.sin(0.3 * t) * np.cos(theta_true)
        x_values.append(x)
        y_values.append(y)
    
    # Save to CSV for testing
    import pandas as pd
    test_data = pd.DataFrame({'x': x_values, 'y': y_values})
    test_data.to_csv('test_data.csv', index=False)
    print(f"\nGenerated {len(test_data)} test data points")
    
    # Run optimization
    print("\nRunning optimization...")
    fitter = ParametricCurveFitter('test_data.csv')
    fitter.load_data()
    
    # Use differential evolution for faster testing
    results = fitter.optimize(strategy='differential_evolution')
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"\nOptimized Parameters:")
    print(f"  θ = {np.rad2deg(results['theta']):.4f}° ({results['theta']:.6f} radians)")
    print(f"  M = {results['M']:.6f}")
    print(f"  X = {results['X']:.6f}")
    print(f"\nL1 Distance Error: {results['error']:.6f}")
    
    print("\nParameter Errors:")
    print(f"  θ error: {abs(results['theta'] - theta_true):.6f} radians ({abs(np.rad2deg(results['theta'] - theta_true)):.4f}°)")
    print(f"  M error: {abs(results['M'] - M_true):.6f}")
    print(f"  X error: {abs(results['X'] - X_true):.6f}")
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)
    
    # Cleanup
    import os
    if os.path.exists('test_data.csv'):
        os.remove('test_data.csv')
    
    return results

if __name__ == "__main__":
    test_optimization()

