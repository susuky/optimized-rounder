
import warnings

from typing import List, Optional, Union, Callable, Dict

import numpy as np
import optuna

from sklearn.metrics import cohen_kappa_score, mean_squared_error, accuracy_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold


__all__ = ['OptimizedRounder']


class OptimizedRounder:
    '''
    An optimizer for finding optimal thresholds in ordinal classification problems.
    Uses Optuna for efficient threshold search with support for cross-validation
    and multiple evaluation metrics.
    '''

    def __init__(self, n_classes: Optional[int] = None, 
                 n_trials: int = 200,
                 cv: Optional[Union[int, object]] = None, 
                 stratified: bool = True,
                 metric: str = 'quadratic_kappa', 
                 verbose: bool = False,
                 random_state: int = 42):
        '''
        Initialize the OptimizedRounder.

        Args:
            n_classes: Number of target classes (0, 1, 2, ..., n_classes-1)
            n_trials: Number of optimization trials for Optuna
            cv: Number of cross-validation folds or a CV splitter object
            stratified: Whether to use stratified CV (only when cv is an integer)
            metric: Metric to optimize ('quadratic_kappa', 'linear_kappa', 
                   'rmse', 'accuracy', 'f1_macro', 'f1_weighted', 'f1_micro')
            verbose: Whether to display Optuna's optimization progress
            random_state: Random seed for reproducibility
        '''
        self.n_classes = n_classes
        self.n_trials = n_trials
        self.cv = cv
        self.stratified = stratified
        self.random_state = random_state
        self.thresholds = None
        self.study = None
        self.cv_results_ = None

        # Set up the metric function
        self.metric_name = metric
        self.metric_func = self._get_metric_function(metric)

        # Configure Optuna logging
        optuna.logging.set_verbosity(
            optuna.logging.WARNING if not verbose else 
            optuna.logging.INFO
        )

        # set default thresholds
        if self.n_classes is not None:
            self.default_thresholds = np.linspace(0.5, 
                                                  self.n_classes - 1.5, 
                                                  self.n_classes - 1)
        else:
            self.default_thresholds = None

    def _get_metric_function(self, metric: str) -> Callable:
        '''
        Get the appropriate metric function based on the metric name.

        Args:
            metric: Name of the metric

        Returns:
            Metric function that takes y_true and y_pred as input
        '''
        metric_map = {
            'quadratic_kappa': lambda y, y_pred: cohen_kappa_score(y, y_pred, weights='quadratic'),
            'linear_kappa': lambda y, y_pred: cohen_kappa_score(y, y_pred, weights='linear'),
            # Negative for maximization
            'rmse': lambda y, y_pred: -np.sqrt(mean_squared_error(y, y_pred)),
            'accuracy': accuracy_score,
            'f1_macro': lambda y, y_pred: f1_score(y, y_pred, average='macro'),
            'f1_weighted': lambda y, y_pred: f1_score(y, y_pred, average='weighted'),
            'f1_micro': lambda y, y_pred: f1_score(y, y_pred, average='micro')
        }

        if metric not in metric_map:
            raise ValueError(
                f'Unsupported metric: {metric}. Supported metrics: {list(metric_map.keys())}')

        return metric_map[metric]

    def apply_thresholds(self, X: np.ndarray, thresholds: List[float]) -> np.ndarray:
        '''
        Apply thresholds to convert continuous predictions to discrete classes.

        Args:
            X: Continuous predictions array
            thresholds: List of threshold values

        Returns:
            Discrete class predictions
        '''
        return np.searchsorted(thresholds, X, side='right')
    
    def apply_default_thresholds(self, X: np.ndarray) -> np.ndarray:
        '''
        Apply default thresholds to convert continuous predictions to discrete classes.

        Args:
            X: Continuous predictions array

        Returns:
            Discrete class predictions
        '''
        if self.apply_default_thresholds is None:
            raise ValueError('Model not trained. Call fit() before predict().')
        return self.apply_thresholds(X, self.default_thresholds)

    def _evaluate_thresholds(self, thresholds: List[float], X: np.ndarray, y: np.ndarray) -> float:
        '''
        Evaluate thresholds using the selected metric.

        Args:
            thresholds: List of threshold values
            X: Continuous predictions array
            y: True class labels

        Returns:
            Metric score
        '''
        y_pred = self.apply_thresholds(X, thresholds)
        return self.metric_func(y, y_pred)

    def _cv_evaluate_thresholds(self, thresholds: List[float], X: np.ndarray, y: np.ndarray) -> float:
        '''
        Evaluate thresholds using cross-validation.

        Args:
            thresholds: List of threshold values
            X: Continuous predictions array
            y: True class labels

        Returns:
            Average metric score across CV folds
        '''
        cv_scores = []

        for train_idx, val_idx in self.cv_splitter.split(X, y):
            # We don't need to retrain on train_idx since we're just applying thresholds
            X_val, y_val = X[val_idx], y[val_idx]
            score = self._evaluate_thresholds(thresholds, X_val, y_val)
            cv_scores.append(score)

        return np.mean(cv_scores)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'OptimizedRounder':
        '''
        Find optimal thresholds using Optuna optimization.

        Args:
            X: Continuous predictions array
            y: True class labels (values should be 0, 1, 2, ..., n_classes-1)

        Returns:
            self: The fitted optimizer
        '''
        if self.n_classes is None:
            self.n_classes = len(np.unique(y))
            self.default_thresholds = np.linspace(0.5, 
                                                  self.n_classes - 1.5, 
                                                  self.n_classes - 1)

        # Set up cross-validation if requested
        if self.cv is not None:
            if isinstance(self.cv, int):
                if self.stratified:
                    self.cv_splitter = StratifiedKFold(n_splits=self.cv, 
                                                       shuffle=True,
                                                       random_state=self.random_state)
                else:
                    self.cv_splitter = KFold(n_splits=self.cv, 
                                             shuffle=True,
                                             random_state=self.random_state)
            else:
                self.cv_splitter = self.cv

        x_min, x_max = X.min(), X.max()

        def objective(trial):
            # Generate monotonically increasing thresholds
            thresholds = []
            lower_bound = x_min
            for i in range(self.n_classes - 1):
                t = trial.suggest_float(f'threshold_{i}', lower_bound, x_max)
                thresholds.append(t)
                lower_bound = t  # Ensure next threshold is greater than current

            # Use cross-validation if requested
            if self.cv is not None:
                return self._cv_evaluate_thresholds(thresholds, X, y)
            else:
                return self._evaluate_thresholds(thresholds, X, y)

        # Create and run optimization study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        self.study.optimize(objective, n_trials=self.n_trials)

        # Extract and sort best thresholds
        best_params = self.study.best_params
        self.thresholds = sorted([
            best_params[f'threshold_{i}'] for i in range(self.n_classes - 1)
        ])

        # Calculate CV results if CV was used
        if self.cv is not None:
            self.cv_results_ = self._calculate_cv_metrics(X, y)

        return self

    def _calculate_cv_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
        '''
        Calculate various metrics across CV folds using the best thresholds.

        Args:
            X: Continuous predictions array
            y: True class labels

        Returns:
            Dictionary with CV metrics
        '''
        metrics = {
            'quadratic_kappa': lambda y, y_pred: cohen_kappa_score(y, y_pred, weights='quadratic'),
            'linear_kappa': lambda y, y_pred: cohen_kappa_score(y, y_pred, weights='linear'),
            'rmse': lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)),
            'accuracy': accuracy_score,
            'f1_macro': lambda y, y_pred: f1_score(y, y_pred, average='macro'),
            'f1_weighted': lambda y, y_pred: f1_score(y, y_pred, average='weighted')
        }

        results = {name: {'fold_scores': [], 'mean': 0.0, 'std': 0.0}
                   for name in metrics}

        for train_idx, val_idx in self.cv_splitter.split(X, y):
            X_val, y_val = X[val_idx], y[val_idx]
            y_pred = self.apply_thresholds(X_val, self.thresholds)

            for name, metric_func in metrics.items():
                try:
                    score = metric_func(y_val, y_pred)
                    results[name]['fold_scores'].append(score)
                except Exception as e:
                    warnings.warn(f'Error calculating {name}: {str(e)}')

        # Calculate mean and std for each metric
        for name in metrics:
            if results[name]['fold_scores']:
                results[name]['mean'] = np.mean(results[name]['fold_scores'])
                results[name]['std'] = np.std(results[name]['fold_scores'])

        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Convert continuous predictions to discrete classes using optimal thresholds.

        Args:
            X: Continuous predictions array

        Returns:
            Discrete class predictions
        '''
        if self.thresholds is None:
            raise ValueError('Model not trained. Call fit() before predict().')
        return self.apply_thresholds(X, self.thresholds)

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        Train the optimizer and return predictions in one step.

        Args:
            X: Continuous predictions array
            y: True class labels

        Returns:
            Discrete class predictions
        '''
        return self.fit(X, y).predict(X)

    def coefficients(self) -> List[float]:
        '''
        Get the optimal thresholds found during training.

        Returns:
            List of optimal threshold values
        '''
        if self.thresholds is None:
            raise ValueError(
                'Model not trained. Call fit() before accessing thresholds.'
            )
        return self.thresholds

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        '''
        Evaluate the model on multiple metrics.

        Args:
            X: Continuous predictions array
            y: True class labels

        Returns:
            Dictionary with evaluation metrics
        '''
        if self.thresholds is None:
            raise ValueError(
                'Model not trained. Call fit() before evaluate().'
            )

        y_pred = self.predict(X)

        metrics = {
            'quadratic_kappa': cohen_kappa_score(y, y_pred, weights='quadratic'),
            'linear_kappa': cohen_kappa_score(y, y_pred, weights='linear'),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'accuracy': accuracy_score(y, y_pred),
            'f1_macro': f1_score(y, y_pred, average='macro'),
            'f1_weighted': f1_score(y, y_pred, average='weighted')
        }

        return metrics
