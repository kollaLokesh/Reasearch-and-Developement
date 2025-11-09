"""
Quick optimization runner - uses only Differential Evolution for speed
"""
import sys
import io

# Set UTF-8 encoding for stdout
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from curve_fitting import ParametricCurveFitter
import numpy as np

def main():
    print("="*60)
    print("PARAMETRIC CURVE FITTING OPTIMIZATION")
    print("="*60)
    print("\nLoading data...")
    
    # Initialize fitter
    fitter = ParametricCurveFitter('xy_data.csv')
    
    # Load data
    fitter.load_data()
    print(f"Loaded {len(fitter.data)} data points")
    
    print("\nRunning optimization with all strategies...")
    print("This may take several minutes...")
    
    # Run optimization with all strategies for best results
    results = fitter.optimize(strategy='all')
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"\nOptimized Parameters:")
    print(f"  theta = {np.rad2deg(results['theta']):.6f} deg ({results['theta']:.6f} radians)")
    print(f"  M = {results['M']:.6f}")
    print(f"  X = {results['X']:.6f}")
    print(f"\nL1 Distance Error: {results['error']:.6f}")
    
    print("\n" + "="*60)
    print("LATEX EQUATION")
    print("="*60)
    print(fitter.get_latex_equation())
    
    print("\n" + "="*60)
    print("DESMOS EQUATION")
    print("="*60)
    print(fitter.get_desmos_equation())
    
    print("\nGenerating visualization...")
    fitter.visualize()
    
    print("\nDone!")
    
    return fitter, results

if __name__ == "__main__":
    fitter, results = main()

