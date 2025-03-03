import pytest
import numpy as np

from sklearn.model_selection import KFold
from sklearn.datasets import make_classification

from oprounder import OptimizedRounder


class TestOptimizedRounder:
    @pytest.fixture
    def simple_data(self):
        '''Create a simple dataset for testing.'''
        X = np.array([0.1, 0.2, 0.7, 1.3, 1.8, 2.2, 2.7, 3.3, 3.6, 3.8])
        y = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        return X, y
    
    @pytest.fixture
    def classification_data(self):
        '''Create a synthetic dataset for classification testing.'''
        y = np.random.randint(0, 4, size=100)  # Reduced size from 1000 to 100
        X = y + np.random.normal(0, 0.9, size=100)
        return X, y
    
    def test_initialization_and_properties(self):
        '''Test initialization and basic properties.'''
        rounder = OptimizedRounder(n_classes=5, n_trials=50, cv=3)
        
        assert rounder.n_classes == 5
        assert rounder.n_trials == 50
        assert rounder.cv == 3
        assert rounder.thresholds is None
        assert rounder.metric_name == 'quadratic_kappa'
        assert callable(rounder.metric_func)
        assert np.allclose(rounder.default_thresholds, [0.5, 1.5, 2.5, 3.5])
    
    def test_threshold_application(self, simple_data):
        '''Test threshold application methods.'''
        X, expected = simple_data
        rounder = OptimizedRounder()
        
        # Test apply_thresholds
        thresholds = [0.5, 1.5, 2.5, 3.5]
        predictions = rounder.apply_thresholds(X, thresholds)
        assert np.array_equal(predictions, expected)
        
        # Test apply_default_thresholds
        rounder = OptimizedRounder(n_classes=5)
        predictions = rounder.apply_default_thresholds(X)
        assert np.array_equal(predictions, expected)
        
        # Test error case
        rounder = OptimizedRounder()
        with pytest.raises(ValueError, match='Model not trained'):
            rounder.apply_default_thresholds(X)
    
    def test_fit_and_predict(self, simple_data):
        '''Test fitting and prediction functionality.'''
        X, y = simple_data
        rounder = OptimizedRounder(n_classes=5, n_trials=20, random_state=42)
        
        # Test fit
        rounder.fit(X, y)
        assert rounder.thresholds is not None
        assert len(rounder.thresholds) == 4
        assert rounder.thresholds[0] < rounder.thresholds[1] < rounder.thresholds[2]
        
        # Test predict
        y_pred = rounder.predict(X)
        assert len(y_pred) == len(X)
        assert np.all((y_pred >= 0) & (y_pred < 5))
        
        # Test fit_predict
        y_pred2 = rounder.fit_predict(X, y)
        assert len(y_pred2) == len(X)
        
        # Test predict error
        rounder = OptimizedRounder()
        with pytest.raises(ValueError, match='Model not trained'):
            rounder.predict(X)
    
    def test_cross_validation(self, simple_data):
        '''Test cross-validation functionality.'''
        X, y = simple_data
        
        # Test stratified CV
        rounder = OptimizedRounder(n_classes=5, n_trials=10, cv=2, stratified=True, random_state=42)
        rounder.fit(X, y)
        assert rounder.cv_results_ is not None
        assert 'quadratic_kappa' in rounder.cv_results_
        
        # Test non-stratified CV
        rounder = OptimizedRounder(n_classes=5, n_trials=10, cv=2, stratified=False, random_state=42)
        rounder.fit(X, y)
        assert rounder.cv_results_ is not None
        
        # Test custom CV splitter
        custom_cv = KFold(n_splits=2, shuffle=True, random_state=42)
        rounder = OptimizedRounder(n_classes=5, n_trials=10, cv=custom_cv, random_state=42)
        rounder.fit(X, y)
        assert rounder.cv_results_ is not None
    
    @pytest.mark.parametrize('metric', ['quadratic_kappa', 'linear_kappa', 'accuracy', 'f1_macro'])
    def test_different_metrics(self, classification_data, metric):
        '''Test optimization with different metrics.'''
        X, y = classification_data
        rounder = OptimizedRounder(n_classes=4, n_trials=2, metric=metric, random_state=42)
        rounder.fit(X, y)
        assert rounder.thresholds is not None
    
    def test_invalid_metric(self):
        '''Test error with invalid metric.'''
        with pytest.raises(ValueError, match='Unsupported metric'):
            OptimizedRounder(metric='invalid_metric')
    
    def test_auto_detect_classes_and_coefficients(self, simple_data):
        '''Test automatic detection of classes and coefficients functionality.'''
        X, y = simple_data
        rounder = OptimizedRounder(n_trials=20, random_state=42)
        
        # Test auto-detection of classes
        rounder.fit(X, y)
        assert rounder.n_classes == 5
        assert len(rounder.thresholds) == 4
        
        # Test coefficients method
        thresholds = rounder.coefficients()
        assert len(thresholds) == 4
        assert thresholds[0] < thresholds[1] < thresholds[2] < thresholds[3]
        
        # Test coefficients error
        rounder = OptimizedRounder()
        with pytest.raises(ValueError, match='Model not trained'):
            rounder.coefficients()
    
    def test_evaluation(self, simple_data):
        '''Test evaluation metrics.'''
        X, y = simple_data
        rounder = OptimizedRounder(n_classes=5, n_trials=20, random_state=42)
        
        rounder.fit(X, y)
        metrics = rounder.evaluate(X, y)
        
        # Check that all expected metrics are present
        expected_metrics = ['quadratic_kappa', 'linear_kappa', 'rmse', 
                           'accuracy', 'f1_macro', 'f1_weighted']
        for metric in expected_metrics:
            assert metric in metrics
        
        # Test evaluate error
        rounder = OptimizedRounder()
        with pytest.raises(ValueError, match='Model not trained'):
            rounder.evaluate(X, y)
