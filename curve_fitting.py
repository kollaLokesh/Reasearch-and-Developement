"""
Parametric Curve Fitting Optimization
=====================================
Finds optimal parameters θ, M, X for the parametric equations:
    x = t * cos(θ) - e^(M|t|) * sin(0.3t) * sin(θ) + X
    y = 42 + t * sin(θ) + e^(M|t|) * sin(0.3t) * cos(θ)

Constraints:
    θ: 0° < θ < 50° (in radians: 0 < θ < 0.8727)
    M: -0.05 < M < 0.05
    X: 0 < X < 100
    t: 6 < t < 60
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize, basinhopping
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')


class ParametricCurveFitter:
    """
    Optimizes parameters for parametric curve fitting using multiple strategies.
    """
    
    def __init__(self, data_file='xy_data.csv'):
        """
        Initialize the curve fitter.
        
        Parameters:
        -----------
        data_file : str
            Path to CSV file containing (x, y) data points
        """
        self.data_file = data_file
        self.data = None
        self.theta_opt = None
        self.M_opt = None
        self.X_opt = None
        self.best_error = None
        
        # Parameter bounds
        self.bounds = [
            (0.001, np.deg2rad(50) - 0.001),  # theta: 0 < θ < 50° (in radians)
            (-0.05 + 1e-6, 0.05 - 1e-6),      # M: -0.05 < M < 0.05
            (0.001, 100 - 0.001)              # X: 0 < X < 100
        ]
        
        # Time range for evaluation
        self.t_min = 6
        self.t_max = 60
        self.n_samples = 1000  # Number of uniform samples for L1 distance
        
    def load_data(self):
        """Load data from CSV file."""
        try:
            self.data = pd.read_csv(self.data_file)
            print(f"Loaded {len(self.data)} data points from {self.data_file}")
            print(f"Data columns: {self.data.columns.tolist()}")
            return True
        except FileNotFoundError:
            print(f"Warning: {self.data_file} not found. Using synthetic data for demonstration.")
            self._generate_synthetic_data()
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def _generate_synthetic_data(self):
        """Generate synthetic data for testing (if CSV not available)."""
        # Using example parameters for synthetic data
        theta_true = np.deg2rad(25)
        M_true = 0.02
        X_true = 50
        
        t_values = np.linspace(6.1, 59.9, 27)
        x_values = []
        y_values = []
        
        for t in t_values:
            x = t * np.cos(theta_true) - np.exp(M_true * abs(t)) * np.sin(0.3 * t) * np.sin(theta_true) + X_true
            y = 42 + t * np.sin(theta_true) + np.exp(M_true * abs(t)) * np.sin(0.3 * t) * np.cos(theta_true)
            x_values.append(x)
            y_values.append(y)
        
        self.data = pd.DataFrame({'x': x_values, 'y': y_values})
        print(f"Generated {len(self.data)} synthetic data points")
    
    def parametric_equations(self, t, theta, M, X):
        """
        Evaluate parametric equations for given parameters.
        
        Parameters:
        -----------
        t : array-like
            Time values
        theta : float
            Rotation angle (radians)
        M : float
            Exponential scaling parameter
        X : float
            X-axis offset
        
        Returns:
        --------
        x, y : tuple of arrays
            Parametric curve coordinates
        """
        t = np.array(t)
        x = t * np.cos(theta) - np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.sin(theta) + X
        y = 42 + t * np.sin(theta) + np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.cos(theta)
        return x, y
    
    def l1_distance(self, params):
        """
        Calculate L1 distance between uniformly sampled predicted curve and actual data.
        
        Parameters:
        -----------
        params : array-like
            [theta, M, X] parameter values
        
        Returns:
        --------
        l1_dist : float
            L1 distance metric
        """
        theta, M, X = params
        
        # Uniformly sample t values
        t_samples = np.linspace(self.t_min, self.t_max, self.n_samples)
        
        # Generate predicted curve points
        x_pred, y_pred = self.parametric_equations(t_samples, theta, M, X)
        pred_points = np.column_stack([x_pred, y_pred])
        
        # Get actual data points
        actual_points = self.data[['x', 'y']].values
        
        # Calculate minimum L1 distance from each predicted point to nearest actual point
        # and vice versa (bidirectional distance)
        distances_pred_to_actual = cdist(pred_points, actual_points, metric='cityblock')
        distances_actual_to_pred = cdist(actual_points, pred_points, metric='cityblock')
        
        # L1 distance: sum of minimum distances
        min_distances_pred = np.min(distances_pred_to_actual, axis=1)
        min_distances_actual = np.min(distances_actual_to_pred, axis=1)
        
        # Average L1 distance
        l1_dist = np.mean(min_distances_pred) + np.mean(min_distances_actual)
        
        return l1_dist
    
    def objective_function(self, params):
        """
        Objective function for optimization (minimize L1 distance).
        
        Parameters:
        -----------
        params : array-like
            [theta, M, X] parameter values
        
        Returns:
        --------
        error : float
            L1 distance error
        """
        # Ensure parameters are within bounds
        theta, M, X = params
        theta = np.clip(theta, self.bounds[0][0], self.bounds[0][1])
        M = np.clip(M, self.bounds[1][0], self.bounds[1][1])
        X = np.clip(X, self.bounds[2][0], self.bounds[2][1])
        
        error = self.l1_distance([theta, M, X])
        return error
    
    def optimize_differential_evolution(self, maxiter=500, seed=42):
        """
        Optimize using Differential Evolution (global optimization).
        
        Parameters:
        -----------
        maxiter : int
            Maximum iterations
        seed : int
            Random seed for reproducibility
        
        Returns:
        --------
        result : OptimizeResult
            Optimization result
        """
        print("\n" + "="*60)
        print("Optimization Strategy 1: Differential Evolution")
        print("="*60)
        print("This method is robust to local minima and suitable for")
        print("non-convex optimization problems with bounded parameters.")
        
        result = differential_evolution(
            self.objective_function,
            bounds=self.bounds,
            maxiter=maxiter,
            seed=seed,
            strategy='best1bin',
            popsize=15,
            mutation=(0.5, 1),
            recombination=0.7,
            polish=True,  # Local optimization refinement
            atol=1e-8,
            tol=1e-8
        )
        
        return result
    
    def optimize_basinhopping(self, initial_guess, niter=200):
        """
        Optimize using Basin Hopping (global optimization with local search).
        
        Parameters:
        -----------
        initial_guess : array-like
            Initial parameter guess [theta, M, X]
        niter : int
            Number of basin hopping iterations
        
        Returns:
        --------
        result : OptimizeResult
            Optimization result
        """
        print("\n" + "="*60)
        print("Optimization Strategy 2: Basin Hopping")
        print("="*60)
        print("This method combines global and local optimization.")
        print("Good for escaping local minima.")
        
        # Local minimizer options
        minimizer_kwargs = {
            'method': 'L-BFGS-B',
            'bounds': self.bounds
        }
        
        result = basinhopping(
            self.objective_function,
            initial_guess,
            niter=niter,
            minimizer_kwargs=minimizer_kwargs,
            T=1.0,
            stepsize=0.1
        )
        
        return result
    
    def optimize_multistart(self, n_starts=20, method='L-BFGS-B'):
        """
        Multi-start local optimization.
        
        Parameters:
        -----------
        n_starts : int
            Number of random starting points
        method : str
            Optimization method
        
        Returns:
        --------
        best_result : OptimizeResult
            Best optimization result
        """
        print("\n" + "="*60)
        print(f"Optimization Strategy 3: Multi-Start ({method})")
        print("="*60)
        print(f"Trying {n_starts} random starting points.")
        
        best_error = np.inf
        best_result = None
        
        np.random.seed(42)
        for i in range(n_starts):
            # Random initialization within bounds
            initial_guess = [
                np.random.uniform(self.bounds[0][0], self.bounds[0][1]),
                np.random.uniform(self.bounds[1][0], self.bounds[1][1]),
                np.random.uniform(self.bounds[2][0], self.bounds[2][1])
            ]
            
            try:
                result = minimize(
                    self.objective_function,
                    initial_guess,
                    method=method,
                    bounds=self.bounds,
                    options={'maxiter': 1000}
                )
                
                if result.success and result.fun < best_error:
                    best_error = result.fun
                    best_result = result
                    print(f"  Start {i+1}: Found better solution (error={best_error:.6f})")
            except Exception as e:
                print(f"  Start {i+1}: Failed - {e}")
        
        return best_result
    
    def optimize(self, strategy='all'):
        """
        Run optimization using specified strategy or all strategies.
        
        Parameters:
        -----------
        strategy : str
            'differential_evolution', 'basinhopping', 'multistart', or 'all'
        
        Returns:
        --------
        best_params : dict
            Best parameters found
        """
        if self.data is None:
            self.load_data()
        
        results = []
        
        if strategy in ['differential_evolution', 'all']:
            result_de = self.optimize_differential_evolution()
            results.append(('Differential Evolution', result_de))
            print(f"\nResult: theta={np.rad2deg(result_de.x[0]):.4f} deg, "
                  f"M={result_de.x[1]:.6f}, X={result_de.x[2]:.6f}")
            print(f"L1 Distance: {result_de.fun:.6f}")
        
        if strategy in ['basinhopping', 'all']:
            # Use DE result as initial guess if available
            if results:
                initial_guess = results[0][1].x
            else:
                initial_guess = [
                    np.deg2rad(25),
                    0.01,
                    50
                ]
            result_bh = self.optimize_basinhopping(initial_guess)
            results.append(('Basin Hopping', result_bh))
            print(f"\nResult: theta={np.rad2deg(result_bh.x[0]):.4f} deg, "
                  f"M={result_bh.x[1]:.6f}, X={result_bh.x[2]:.6f}")
            print(f"L1 Distance: {result_bh.fun:.6f}")
        
        if strategy in ['multistart', 'all']:
            result_ms = self.optimize_multistart()
            if result_ms:
                results.append(('Multi-Start', result_ms))
                print(f"\nResult: theta={np.rad2deg(result_ms.x[0]):.4f} deg, "
                      f"M={result_ms.x[1]:.6f}, X={result_ms.x[2]:.6f}")
                print(f"L1 Distance: {result_ms.fun:.6f}")
        
        # Find best result
        if results:
            best_result = min(results, key=lambda x: x[1].fun)
            method_name, result = best_result
            
            self.theta_opt = result.x[0]
            self.M_opt = result.x[1]
            self.X_opt = result.x[2]
            self.best_error = result.fun
            
            print("\n" + "="*60)
            print("BEST SOLUTION FOUND")
            print("="*60)
            print(f"Method: {method_name}")
            print(f"theta = {np.rad2deg(self.theta_opt):.6f} deg ({self.theta_opt:.6f} radians)")
            print(f"M = {self.M_opt:.6f}")
            print(f"X = {self.X_opt:.6f}")
            print(f"L1 Distance: {self.best_error:.6f}")
            
            return {
                'theta': self.theta_opt,
                'M': self.M_opt,
                'X': self.X_opt,
                'error': self.best_error,
                'method': method_name
            }
        else:
            raise ValueError("No optimization results available")
    
    def visualize(self, save_path='curve_fitting_results.png'):
        """
        Visualize the fitted curve against actual data.
        
        Parameters:
        -----------
        save_path : str
            Path to save the visualization
        """
        if self.theta_opt is None:
            raise ValueError("Must run optimization first")
        
        # Generate fitted curve
        t_fitted = np.linspace(self.t_min, self.t_max, 1000)
        x_fitted, y_fitted = self.parametric_equations(t_fitted, self.theta_opt, self.M_opt, self.X_opt)
        
        # Actual data points
        x_actual = self.data['x'].values
        y_actual = self.data['y'].values
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot 1: Curve comparison
        ax1 = axes[0]
        ax1.plot(x_fitted, y_fitted, 'b-', linewidth=2, label='Fitted Curve', alpha=0.7)
        ax1.scatter(x_actual, y_actual, c='red', s=100, marker='o', 
                   edgecolors='black', linewidth=1.5, label='Actual Data', zorder=5)
        ax1.set_xlabel('X', fontsize=12)
        ax1.set_ylabel('Y', fontsize=12)
        ax1.set_title('Fitted Parametric Curve vs Actual Data', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Plot 2: Residuals
        ax2 = axes[1]
        # Find closest points on fitted curve for each actual data point
        residuals = []
        for x_a, y_a in zip(x_actual, y_actual):
            distances = np.sqrt((x_fitted - x_a)**2 + (y_fitted - y_a)**2)
            min_idx = np.argmin(distances)
            residual = distances[min_idx]
            residuals.append(residual)
        
        ax2.scatter(range(len(residuals)), residuals, c='green', s=80, alpha=0.7)
        ax2.axhline(y=np.mean(residuals), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(residuals):.4f}')
        ax2.set_xlabel('Data Point Index', fontsize=12)
        ax2.set_ylabel('Residual Distance', fontsize=12)
        ax2.set_title('Residual Analysis', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to {save_path}")
        plt.show()
    
    def get_latex_equation(self):
        """
        Generate LaTeX formatted equations with optimized parameters.
        
        Returns:
        --------
        latex_str : str
            LaTeX formatted equation string
        """
        if self.theta_opt is None:
            raise ValueError("Must run optimization first")
        
        theta_str = f"{self.theta_opt:.6f}"
        M_str = f"{self.M_opt:.6f}"
        X_str = f"{self.X_opt:.6f}"
        
        latex_str = (
            f"\\left(t\\cdot\\cos({theta_str})-e^{{{M_str}\\left|t\\right|}}\\cdot"
            f"\\sin(0.3t)\\cdot\\sin({theta_str})+{X_str},\\ "
            f"42+t\\cdot\\sin({theta_str})+e^{{{M_str}\\left|t\\right|}}\\cdot"
            f"\\sin(0.3t)\\cdot\\cos({theta_str})\\right)"
        )
        
        return latex_str
    
    def get_desmos_equation(self):
        """
        Generate Desmos calculator format equation.
        
        Returns:
        --------
        desmos_str : str
            Desmos formatted equation string
        """
        if self.theta_opt is None:
            raise ValueError("Must run optimization first")
        
        theta_str = f"{self.theta_opt:.6f}"
        M_str = f"{self.M_opt:.6f}"
        X_str = f"{self.X_opt:.6f}"
        
        desmos_str = (
            f"(t*cos({theta_str})-e^({M_str}*abs(t))*sin(0.3t)*sin({theta_str})+{X_str},"
            f"42+t*sin({theta_str})+e^({M_str}*abs(t))*sin(0.3t)*cos({theta_str}))"
        )
        
        return desmos_str


def main():
    """Main execution function."""
    print("="*60)
    print("PARAMETRIC CURVE FITTING OPTIMIZATION")
    print("="*60)
    print("\nProblem:")
    print("  x = t * cos(theta) - e^(M|t|) * sin(0.3t) * sin(theta) + X")
    print("  y = 42 + t * sin(theta) + e^(M|t|) * sin(0.3t) * cos(theta)")
    print("\nConstraints:")
    print("  0 deg < theta < 50 deg")
    print("  -0.05 < M < 0.05")
    print("  0 < X < 100")
    print("  6 < t < 60")
    
    # Initialize fitter
    fitter = ParametricCurveFitter('xy_data.csv')
    
    # Load data
    fitter.load_data()
    
    # Run optimization
    results = fitter.optimize(strategy='all')
    
    # Visualize results
    fitter.visualize()
    
    # Print final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
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
    
    return fitter, results


if __name__ == "__main__":
    fitter, results = main()

