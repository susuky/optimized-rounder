
# Optimized Rounder
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An efficient optimizer for finding optimal thresholds in ordinal classification problems. This package uses Optuna for efficient threshold search with support for cross-validation and multiple evaluation metrics.

## Installation

```bash
pip install optimized-rounder
```

## Features

- **Threshold Optimization**: Find optimal thresholds to convert continuous predictions to discrete classes
- **Multiple Metrics**: Support for various evaluation metrics including quadratic kappa, linear kappa, RMSE, accuracy, and F1 scores
- **Cross-Validation**: Built-in support for K-fold and stratified cross-validation
- **Efficient Search**: Uses Optuna for efficient Bayesian optimization of thresholds
- **Comprehensive Evaluation**: Evaluate models using multiple metrics simultaneously

## Quick Start

```python
from oprounder import OptimizedRounder
import numpy as np
from sklearn.metrics import cohen_kappa_score

# Generate synthetic data
np.random.seed(42)
n_classes = 4
n_samples = 1000
y_true = np.random.randint(0, n_classes, size=n_samples)
output = y_true + np.random.normal(0, 0.9, size=n_samples) # dummy model output

# Initialize and fit the optimizer
rounder = OptimizedRounder(n_classes=n_classes, n_trials=100)
rounder.fit(output, y_true)

# Get the optimal thresholds
print(f'Optimal thresholds: {rounder.thresholds}')

# Make predictions
y_pred = rounder.predict(output)
kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
print(f'Optimal Quadratic kappa: {kappa:.4f}')
y_pred_default = rounder.apply_thresholds(output, rounder.default_thresholds) # [0.5, 1.5, 2.5, 3.5]
kappa_default = cohen_kappa_score(y_true, y_pred_default, weights='quadratic')
print(f'Default Quadratic kappa: {kappa_default:.4f}')
```

## Advanced Usage

### With Cross-Validation

```python
# Use 5-fold stratified cross-validation
rounder = OptimizedRounder(
    n_classes=4,
    n_trials=200,
    cv=5,
    stratified=True,
    metric='quadratic_kappa',
    verbose=True
)

rounder.fit(output, y_true)
print(f'CV Results: {rounder.cv_results_}')
```

### Using Different Metrics

```python
# Optimize for F1 weighted score
rounder = OptimizedRounder(
    n_classes=4,
    n_trials=200,
    metric='f1_weighted'
)

rounder.fit(output, y_true)

# Comprehensive evaluation
output_val = y_true + np.random.normal(0, 0.8, size=n_samples)
metrics = rounder.evaluate(output_val, y_true)
for metric_name, value in metrics.items():
    print(f"{metric_name}: {value:.4f}")
```

## API Reference

### OptimizedRounder

```python
OptimizedRounder(
    n_classes=None,
    n_trials=200,
    cv=None,
    stratified=True,
    metric='quadratic_kappa',
    verbose=False,
    random_state=42
)
```

#### Parameters

- **n_classes**: Number of target classes (0, 1, 2, ..., n_classes-1)
- **n_trials**: Number of optimization trials for Optuna
- **cv**: Number of cross-validation folds or a CV splitter object
- **stratified**: Whether to use stratified CV (only when cv is an integer)
- **metric**: Metric to optimize ('quadratic_kappa', 'linear_kappa', 'rmse', 'accuracy', 'f1_macro', 'f1_weighted', 'f1_micro')
- **verbose**: Whether to display Optuna's optimization progress
- **random_state**: Random seed for reproducibility

#### Methods

- **fit(X, y)**: Find optimal thresholds using Optuna optimization
- **predict(X)**: Convert continuous predictions to discrete classes using optimal thresholds
- **fit_predict(X, y)**: Train the optimizer and return predictions in one step
- **coefficients()**: Get the optimal thresholds found during training
- **evaluate(X, y)**: Evaluate the model on multiple metrics

## Use Cases

- **Regression to Classification**: Convert regression outputs to discrete classes
- **Ordinal Classification**: Optimize thresholds for ordinal targets
- **Ensemble Calibration**: Calibrate probability outputs from ensemble models
- **Competition Metrics**: Optimize directly for competition metrics like quadratic kappa

## License

This project is licensed under the MIT License - see the LICENSE file for details.