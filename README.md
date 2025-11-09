# Parametric Curve Fitting Optimization

## Problem Statement

Find the unknown parameters **θ, M, X** that best fit the given data points for the parametric equations:

**x-equation:**
```
x = t * cos(θ) - e^(M|t|) * sin(0.3t) * sin(θ) + X
```

**y-equation:**
```
y = 42 + t * sin(θ) + e^(M|t|) * sin(0.3t) * cos(θ)
```

### Constraints
- **θ (theta)**: 0° < θ < 50° (in radians: 0 < θ < 0.8727)
- **M**: -0.05 < M < 0.05
- **X**: 0 < X < 100
- **t (time)**: 6 < t < 60

### Evaluation Metric
**L1 Distance Accuracy** (100 points maximum): L1 distance between uniformly sampled points on expected vs predicted curve.

---

## Solution Approach

### 1. Problem Analysis

This is a **non-linear parametric curve fitting problem** with:
- **Non-convex objective function** due to exponential and trigonometric terms
- **Bounded parameter space** with box constraints
- **Multi-modal optimization landscape** (multiple local minima possible)
- **Challenge**: The exponential term `e^(M|t|)` creates non-linear behavior that makes optimization sensitive to parameter initialization

### 2. Optimization Strategy Selection

We employ **multiple optimization strategies** to ensure robustness:

#### Strategy 1: Differential Evolution (Primary Method)
- **Why**: Excellent for global optimization of non-convex, bounded problems
- **Advantages**:
  - Population-based metaheuristic avoids local minima
  - Handles discontinuous and non-differentiable objective functions
  - Works well with bounded constraints
- **Implementation**: Uses `scipy.optimize.differential_evolution` with:
  - Strategy: `best1bin` (robust mutation strategy)
  - Population size: 15
  - Mutation factor: (0.5, 1) - adaptive
  - Crossover probability: 0.7
  - Polish: Enabled (local refinement after global search)

#### Strategy 2: Basin Hopping
- **Why**: Combines global exploration with local refinement
- **Advantages**:
  - Escapes local minima through random jumps
  - Uses local optimizer (L-BFGS-B) for refinement
  - Good for rough energy landscapes
- **Implementation**: Uses `scipy.optimize.basinhopping` with:
  - Initial temperature: 1.0
  - Step size: 0.1
  - Local minimizer: L-BFGS-B with bounds

#### Strategy 3: Multi-Start Local Optimization
- **Why**: Redundant validation using multiple random starting points
- **Advantages**:
  - Provides confidence through consistency
  - Can discover alternative local minima
- **Implementation**: 20 random starts with L-BFGS-B optimizer

### 3. Objective Function: L1 Distance

The L1 (Manhattan) distance metric is calculated as:

1. **Uniform Sampling**: Generate N uniformly distributed t-values in [6, 60]
2. **Predicted Curve**: Compute (x, y) points using parametric equations
3. **Bidirectional Distance**: 
   - For each predicted point → find minimum L1 distance to any actual data point
   - For each actual data point → find minimum L1 distance to any predicted point
4. **Aggregation**: Average of both directional distances

**Mathematical Formulation:**
```
L1_distance = mean(min(L1(pred_i, actual_j))) + mean(min(L1(actual_i, pred_j)))
```

This bidirectional approach ensures the fitted curve both:
- Passes close to actual data points
- Covers the entire data distribution

### 4. Handling the Exponential Term

The term `e^(M|t|)` presents challenges:
- **Absolute value**: Creates non-differentiability at t=0 (but t > 6, so avoided)
- **Exponential growth**: Sensitive to M parameter
- **Solution**: 
  - Use bounded optimization methods that handle discontinuities
  - M constrained to [-0.05, 0.05] keeps exponential moderate
  - Gradient-free methods (DE, Basin Hopping) handle non-smoothness

### 5. Parameter Initialization

Initial guesses are derived from:
- **θ**: Mid-range value (~25° = 0.436 radians)
- **M**: Small positive value (~0.01) to avoid extreme exponential behavior
- **X**: Mid-range value (~50) based on typical data ranges

Global optimization methods (DE, Basin Hopping) are less sensitive to initialization.

---

## Implementation Details

### Code Structure

```
.
├── curve_fitting.py      # Main optimization module
├── data_analysis.py      # Data validation utilities
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

### Key Classes and Functions

#### `ParametricCurveFitter`
- `load_data()`: Loads CSV data or generates synthetic data for testing
- `parametric_equations(t, theta, M, X)`: Evaluates parametric equations
- `l1_distance(params)`: Computes L1 distance metric
- `optimize()`: Runs optimization using selected strategy
- `visualize()`: Creates comparison plots
- `get_latex_equation()`: Generates LaTeX formatted equations
- `get_desmos_equation()`: Generates Desmos calculator format

### Usage

```python
from curve_fitting import ParametricCurveFitter

# Initialize fitter
fitter = ParametricCurveFitter('xy_data.csv')

# Load data
fitter.load_data()

# Run optimization (tries all strategies)
results = fitter.optimize(strategy='all')

# Visualize results
fitter.visualize()

# Get LaTeX equation
latex_eq = fitter.get_latex_equation()

# Get Desmos equation
desmos_eq = fitter.get_desmos_equation()
```

### Command Line Execution

```bash
# Install dependencies
pip install -r requirements.txt

# Run optimization
python curve_fitting.py

# Analyze data
python data_analysis.py
```

---

## Results

### Optimized Parameters

After running all optimization strategies, the best solution found:

```
θ = [OPTIMIZED_VALUE] degrees ([OPTIMIZED_VALUE] radians)
M = [OPTIMIZED_VALUE]
X = [OPTIMIZED_VALUE]
L1 Distance Error: [OPTIMIZED_VALUE]
```

*Note: Actual values will be computed when `xy_data.csv` is provided and the script is executed.*

### LaTeX Equation Format

```
\left(t\cdot\cos([θ])-e^{[M]\left|t\right|}\cdot\sin(0.3t)\cdot\sin([θ])+[X],\ 
42+t\cdot\sin([θ])+e^{[M]\left|t\right|}\cdot\sin(0.3t)\cdot\cos([θ])\right)
```

### Desmos Calculator Format

```
(t*cos([θ])-e^([M]*abs(t))*sin(0.3t)*sin([θ])+[X],
42+t*sin([θ])+e^([M]*abs(t))*sin(0.3t)*cos([θ]))
```

---

## Mathematical Justification

### Why L1 Distance?

1. **Robustness**: Less sensitive to outliers than L2 (Euclidean) distance
2. **Evaluation Criteria**: Directly matches assignment requirements
3. **Bidirectional**: Ensures both coverage and accuracy

### Why Multiple Optimization Strategies?

1. **Robustness**: Different algorithms may find different local minima
2. **Validation**: Consistency across methods increases confidence
3. **Global Optima**: Differential Evolution ensures global exploration
4. **Refinement**: Local methods polish the solution

### Convergence Analysis

The optimization process:
1. **Exploration Phase**: Global methods explore parameter space
2. **Exploitation Phase**: Local methods refine best solutions
3. **Validation**: Cross-checking ensures solution quality

### Sensitivity Analysis

Parameters show different sensitivities:
- **θ**: Affects rotation; moderate sensitivity
- **M**: Affects exponential term; high sensitivity due to exponential nature
- **X**: Affects horizontal offset; low sensitivity

---

## Reproducibility

### Environment Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Random Seed

All optimization methods use `seed=42` for reproducibility.

### Data Format

Expected CSV format:
```csv
x,y
x1,y1
x2,y2
...
```

---

## Validation and Testing

### Error Metrics

The solution provides multiple error metrics:
- **L1 Distance**: Primary evaluation metric
- **Mean Euclidean Distance**: Additional validation
- **Maximum Error**: Worst-case performance
- **Standard Deviation**: Solution consistency

### Visualization

Generated plots:
1. **Curve Comparison**: Fitted curve vs actual data points
2. **Residual Analysis**: Distance errors for each data point

---

## Code Quality Features

✅ **Modular Design**: Separated concerns (optimization, analysis, visualization)  
✅ **Documentation**: Comprehensive docstrings and comments  
✅ **Error Handling**: Graceful handling of missing data  
✅ **Reproducibility**: Fixed random seeds  
✅ **Extensibility**: Easy to add new optimization strategies  
✅ **Visualization**: Clear plots for result interpretation  

---

## Future Improvements

1. **Parallel Processing**: Run multiple optimization strategies in parallel
2. **Bayesian Optimization**: Use GP-based methods for parameter exploration
3. **Adaptive Sampling**: Increase sampling density in regions with high error
4. **Uncertainty Quantification**: Estimate parameter confidence intervals
5. **Cross-Validation**: Validate solution on held-out data

---

## References

- Scipy Optimization Documentation: https://docs.scipy.org/doc/scipy/reference/optimize.html
- Differential Evolution Algorithm: Storn & Price (1997)
- Basin Hopping Algorithm: Wales & Doye (1997)

---

## License

This code is provided for educational and research purposes.

---

## Quick Start

### Step 1: Place your data file
Place your `xy_data.csv` file in the project directory with columns: `x, y`

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run optimization
```bash
python curve_fitting.py
```

This will:
- Load your data
- Run multiple optimization strategies
- Display optimized parameters
- Generate visualization plots
- Output LaTeX and Desmos equations

### Step 4: Verify results
```bash
python test_optimization.py  # Test with synthetic data
python data_analysis.py        # Analyze your data
```

---

## Author

KOLLA LOKEH

