
import numpy as np

from sklearn.metrics import cohen_kappa_score

from oprounder import OptimizedRounder

# Generate synthetic data
np.random.seed(42)

# Assume 4 classes: 0, 1, 2, 3
n_classes = 4
n_samples = 1000
y_true = np.random.randint(0, n_classes, size=n_samples)

# Simulate continuous model outputs with noise
X_train = y_true + np.random.normal(0, 0.8, size=n_samples).clip(-0.2, n_classes - 0.8)
X_val = y_true + np.random.normal(0, 0.3, size=n_samples)

# Example 1: Basic usage
print('=== Basic Usage ===')
rounder = OptimizedRounder(
    n_classes=n_classes, n_trials=50, random_state=42)
rounder.fit(X_train, y_true)
print(f'Best thresholds: {rounder.thresholds}')

# Evaluate on validation set
y_pred_val = rounder.predict(X_val)
kappa_val = cohen_kappa_score(y_true, y_pred_val, weights='quadratic')
print(f'Validation quadratic kappa: {kappa_val:.4f}')

# Example 2: Using cross-validation
print('\n=== With Cross-Validation (5-fold) ===')
rounder_cv = OptimizedRounder(n_classes=n_classes, n_trials=50, cv=5,
                        stratified=True, random_state=42)
rounder_cv.fit(X_train, y_true)
print(f'Best thresholds with CV: {rounder_cv.thresholds}')

# Show CV results
if rounder_cv.cv_results_:
    print('\nCross-Validation Results:')
    for metric, results in rounder_cv.cv_results_.items():
        print(f'{metric}: {results['mean']:.4f} ± {results['std']:.4f}')

# Example 3: Using a different metric
print('\n=== Using F1-Weighted Metric ===')
rounder_f1 = OptimizedRounder(n_classes=n_classes, n_trials=50,
                            metric='f1_weighted', random_state=42)
rounder_f1.fit(X_train, y_true)
print(f'Best thresholds with F1-weighted: {rounder_f1.thresholds}')

# Comprehensive evaluation
print('\n=== Comprehensive Evaluation ===')
metrics = rounder.evaluate(X_val, y_true)
for metric_name, value in metrics.items():
    print(f'{metric_name}: {value:.4f}')
