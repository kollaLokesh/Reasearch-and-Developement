# Optimization Results Summary

## Optimized Parameters

**Date:** Optimization completed successfully

**Method:** Differential Evolution

### Parameter Values:
- **theta (θ)**: 30.000318 degrees (0.523604 radians)
- **M**: 0.030000
- **X**: 55.000424

### Performance Metrics:
- **L1 Distance Error**: 0.045200

---

## LaTeX Equation Format

```
\left(t\cdot\cos(0.523604)-e^{0.030000\left|t\right|}\cdot\sin(0.3t)\cdot\sin(0.523604)+55.000424,\ 42+t\cdot\sin(0.523604)+e^{0.030000\left|t\right|}\cdot\sin(0.3t)\cdot\cos(0.523604)\right)
```

## Desmos Calculator Format

```
(t*cos(0.523604)-e^(0.030000*abs(t))*sin(0.3t)*sin(0.523604)+55.000424,42+t*sin(0.523604)+e^(0.030000*abs(t))*sin(0.3t)*cos(0.523604))
```

---

## Submission Format (as requested)

### Example format from assignment:
```
\left(t*\cos(0.826)-e^{0.0742}\left|t\right|\cdot\sin(0.3t)\sin(0.826)\ +11.5793,42+\
t*\sin(0.826)+e^{0.0742\left|t\right|}\cdot\sin(0.3t)\cos(0.826)\right)
```

### Our optimized parameters in the same format:
```
\left(t*\cos(0.523604)-e^{0.030000\left|t\right|}\cdot\sin(0.3t)\sin(0.523604)+55.000424,42+\
t*\sin(0.523604)+e^{0.030000\left|t\right|}\cdot\sin(0.3t)\cos(0.523604)\right)
```

---

## Parameter Validation

### Constraints Check:
- ✅ **theta**: 0° < 30.000318° < 50° ✓
- ✅ **M**: -0.05 < 0.030000 < 0.05 ✓
- ✅ **X**: 0 < 55.000424 < 100 ✓

### Data Summary:
- **Total data points**: 1500
- **X range**: 59.66 - 109.23
- **Y range**: 46.03 - 69.69
- **Evaluation**: Uniform sampling (1000 points) in t range [6, 60]

---

## Visualization

A visualization comparing the fitted curve to actual data points has been saved as:
- `curve_fitting_results.png`

The plot shows:
1. **Fitted parametric curve** (blue line)
2. **Actual data points** (red dots)
3. **Residual analysis** (distance errors)

---

## Notes

- The optimization used Differential Evolution, a robust global optimization method
- The L1 distance metric (0.0452) indicates excellent fit quality
- Parameters are within all specified constraints
- The solution can be further refined by running all optimization strategies (takes longer but may improve results)

---

## Next Steps (Optional)

To potentially improve the solution further, you can run:
```bash
python curve_fitting.py
```

This will run all three optimization strategies (Differential Evolution, Basin Hopping, Multi-Start) and select the best result. However, this may take 10-20 minutes.

The current solution is already excellent and ready for submission!

