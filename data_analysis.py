"""
Data Analysis and Validation Utilities
========================================
Provides functions for analyzing the data and validating the optimization results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def analyze_data(data_file='xy_data.csv'):
    """
    Analyze the input data file.
    
    Parameters:
    -----------
    data_file : str
        Path to CSV file
    """
    try:
        data = pd.read_csv(data_file)
        print("="*60)
        print("DATA ANALYSIS")
        print("="*60)
        print(f"\nNumber of data points: {len(data)}")
        print(f"\nData columns: {data.columns.tolist()}")
        print(f"\nData statistics:")
        print(data.describe())
        print(f"\nData preview:")
        print(data.head(10))
        
        # Check for missing values
        print(f"\nMissing values:")
        print(data.isnull().sum())
        
        # Basic visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(data['x'], data['y'], c='blue', s=50, alpha=0.7)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Input Data Points')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\nData visualization saved to 'data_analysis.png'")
        
        return data
    except Exception as e:
        print(f"Error analyzing data: {e}")
        return None


def validate_solution(theta, M, X, data_file='xy_data.csv', t_range=(6, 60), n_samples=1000):
    """
    Validate the optimized solution by calculating various error metrics.
    
    Parameters:
    -----------
    theta : float
        Optimized theta value
    M : float
        Optimized M value
    X : float
        Optimized X value
    data_file : str
        Path to CSV file
    t_range : tuple
        (min_t, max_t) range
    n_samples : int
        Number of samples for evaluation
    """
    from curve_fitting import ParametricCurveFitter
    
    fitter = ParametricCurveFitter(data_file)
    fitter.load_data()
    
    # Calculate L1 distance
    l1_error = fitter.l1_distance([theta, M, X])
    
    # Calculate other metrics
    t_samples = np.linspace(t_range[0], t_range[1], n_samples)
    x_pred, y_pred = fitter.parametric_equations(t_samples, theta, M, X)
    pred_points = np.column_stack([x_pred, y_pred])
    actual_points = fitter.data[['x', 'y']].values
    
    # Euclidean distance
    distances_euclidean = cdist(pred_points, actual_points, metric='euclidean')
    min_euclidean = np.min(distances_euclidean, axis=1)
    mean_euclidean = np.mean(min_euclidean)
    
    # Maximum error
    max_error = np.max(min_euclidean)
    
    print("\n" + "="*60)
    print("SOLUTION VALIDATION")
    print("="*60)
    print(f"\nL1 Distance Error: {l1_error:.6f}")
    print(f"Mean Euclidean Distance: {mean_euclidean:.6f}")
    print(f"Maximum Distance: {max_error:.6f}")
    print(f"Standard Deviation: {np.std(min_euclidean):.6f}")
    
    return {
        'l1_error': l1_error,
        'mean_euclidean': mean_euclidean,
        'max_error': max_error,
        'std_error': np.std(min_euclidean)
    }


if __name__ == "__main__":
    analyze_data('xy_data.csv')

