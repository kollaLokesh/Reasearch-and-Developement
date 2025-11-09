"""
Example Usage Script
====================
Demonstrates how to use the parametric curve fitting optimization.
"""

from curve_fitting import ParametricCurveFitter
import numpy as np

def example_usage():
    """Example of using the curve fitting optimization."""
    
    print("="*70)
    print("PARAMETRIC CURVE FITTING - EXAMPLE USAGE")
    print("="*70)
    
    # Initialize the fitter
    fitter = ParametricCurveFitter('xy_data.csv')
    
    # Load data
    print("\n1. Loading data...")
    data_loaded = fitter.load_data()
    
    if not data_loaded:
        print("   Note: Using synthetic data since xy_data.csv not found.")
        print("   Place your xy_data.csv file in the same directory.")
    
    # Run optimization with a specific strategy
    print("\n2. Running optimization...")
    print("   Strategy: Differential Evolution (most robust)")
    results = fitter.optimize(strategy='differential_evolution')
    
    # Display results
    print("\n3. Optimization Results:")
    print(f"   θ = {np.rad2deg(results['theta']):.6f}° ({results['theta']:.6f} radians)")
    print(f"   M = {results['M']:.6f}")
    print(f"   X = {results['X']:.6f}")
    print(f"   L1 Distance Error = {results['error']:.6f}")
    
    # Generate equations
    print("\n4. Generated Equations:")
    print("\n   LaTeX Format:")
    print(f"   {fitter.get_latex_equation()}")
    print("\n   Desmos Format:")
    print(f"   {fitter.get_desmos_equation()}")
    
    # Visualize
    print("\n5. Generating visualization...")
    fitter.visualize(save_path='curve_fitting_results.png')
    
    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)
    
    return fitter, results


if __name__ == "__main__":
    fitter, results = example_usage()

