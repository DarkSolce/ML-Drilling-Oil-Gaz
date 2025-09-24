"""
Unit tests for ML models
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.formation_pressure import (
    PCRFormationPressure, XGBoostFormationPressure, EnsembleFormationPressure
)
from src.models.kick_detection import (
    PCAKickDetection, EnsembleKickDetection
)

class TestFormationPressureModels:
    """Test formation pressure prediction models"""
    
    @pytest.fixture
    def sample_formation_data(self):
        """Create sample formation data for testing"""
        np.random.seed(42)
        n_samples = 200
        
        data = pd.DataFrame({
            'WellDepth': np.cumsum(np.random.normal(1, 0.1, n_samples)) + 2000,
            'WoBit': np.random.normal(25, 5, n_samples),
            'RoPen': np.random.normal(15, 3, n_samples),
            'BTBR': np.random.normal(120, 15, n_samples),
            'WBoPress': np.random.normal(2000, 200, n_samples),
            'HLoad': np.random.normal(150, 15, n_samples),
            'DPPress': np.random.normal(180, 20, n_samples)
        })
        
        # Create synthetic formation pressure
        formation_pressure = (
            0.01 * data['WellDepth'] + 
            0.05 * data['WBoPress'] +
            np.random.normal(0, 50, n_samples)
        )
        
        return data, pd.Series(formation_pressure, name='FPress')
    
    def test_pcr_model_training(self, sample_formation_data):
        """Test PCR model training"""
        X, y = sample_formation_data
        
        model = PCRFormationPressure(n_components=4)
        metrics = model.train(X, y, validation_split=0.2)
        
        # Check if model is fitted
        assert model.is_fitted
        
        # Check metrics
        assert 'val_r2' in metrics
        assert 'val_rmse' in metrics
        assert 'val_mae' in metrics
        
        # Check reasonable performance
        assert metrics['val_r2'] > 0.5  # Minimum acceptable R²
        assert metrics['val_rmse'] < 200  # Reasonable RMSE for pressure
    
    def test_pcr_model_prediction(self, sample_formation_data):
        """Test PCR model prediction"""
        X, y = sample_formation_data
        
        model = PCRFormationPressure(n_components=3)
        model.train(X, y)
        
        # Test prediction
        predictions = model.predict(X.head(10))
        
        assert len(predictions) == 10
        assert all(pred > 0 for pred in predictions)  # Positive pressures
        assert all(pred < 10000 for pred in predictions)  # Reasonable range
    
    def test_xgboost_model_training(self, sample_formation_data):
        """Test XGBoost model training"""
        X, y = sample_formation_data
        
        model = XGBoostFormationPressure()
        metrics = model.train(X, y, validation_split=0.2)
        
        # Check if model is fitted
        assert model.is_fitted
        
        # Check metrics
        assert 'val_r2' in metrics
        assert metrics['val_r2'] > 0.6  # XGBoost should perform well
    
    def test_ensemble_model_training(self, sample_formation_data):
        """Test ensemble model training"""
        X, y = sample_formation_data
        
        model = EnsembleFormationPressure(['pcr', 'xgboost'])
        metrics = model.train(X, y, validation_split=0.2)
        
        # Check if model is fitted
        assert model.is_fitted
        
        # Check that ensemble has model weights
        assert 'model_weights' in metrics
        assert len(metrics['model_weights']) == 2
        
        # Check ensemble performance
        assert metrics['val_r2'] > 0.6
    
    def test_model_save_load(self, sample_formation_data, tmp_path):
        """Test model saving and loading"""
        X, y = sample_formation_data
        
        # Train model
        model = PCRFormationPressure(n_components=3)
        model.train(X, y)
        original_prediction = model.predict(X.head(5))
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        saved_path = model.save_model(str(model_path))
        
        assert Path(saved_path).exists()
        
        # Load model
        new_model = PCRFormationPressure()
        new_model.load_model(saved_path)
        
        # Test loaded model
        assert new_model.is_fitted
        loaded_prediction = new_model.predict(X.head(5))
        
        # Predictions should be identical
        np.testing.assert_array_almost_equal(original_prediction, loaded_prediction)

class TestKickDetectionModels:
    """Test kick detection models"""
    
    @pytest.fixture
    def sample_kick_data(self):
        """Create sample kick detection data"""
        np.random.seed(42)
        n_samples = 500
        
        # Normal drilling data
        data = pd.DataFrame({
            'WellDepth': np.cumsum(np.random.normal(1, 0.1, n_samples)) + 2000,
            'WoBit': np.random.normal(25, 3, n_samples),
            'RoPen': np.random.normal(15, 2, n_samples),
            'BTBR': np.random.normal(120, 10, n_samples),
            'WBoPress': np.random.normal(2000, 100, n_samples),
            'HLoad': np.random.normal(150, 10, n_samples),
            'FIn': np.random.normal(300, 20, n_samples),
            'FOut': np.random.normal(302, 20, n_samples),
            'ActiveGL': np.random.normal(100, 5, n_samples),
            'MRFlow': np.random.normal(295, 15, n_samples),
            'SMSpeed': np.random.normal(50, 10, n_samples)
        })
        
        # Add some anomalies (synthetic kicks)
        n_anomalies = 25
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        
        # Simulate kick characteristics
        # Simulate kick characteristics
        data.loc[anomaly_indices, 'ActiveGL'] += np.random.normal(15, 5, n_anomalies)  # Pit gain
        data.loc[anomaly_indices, 'FOut'] += np.random.normal(10, 3, n_anomalies)      # Flow increase
        data.loc[anomaly_indices, 'WBoPress'] += np.random.normal(100, 20, n_anomalies) # Pressure increase
        
        # Create labels
        labels = np.zeros(n_samples)
        labels[anomaly_indices] = 1
        
        return data, pd.Series(labels, name='kick_label')
    
    def test_pca_kick_detection_training(self, sample_kick_data):
        """Test PCA kick detection model training"""
        X, y = sample_kick_data
        
        model = PCAKickDetection(variance_threshold=0.9)
        metrics = model.train(X, y, validation_split=0.2)
        
        # Check if model is fitted
        assert model.is_fitted
        
        # Check metrics
        assert 'val_anomaly_rate' in metrics
        assert 'spe_threshold' in metrics
        assert 'explained_variance_ratio' in metrics
        
        # Check reasonable performance
        assert 0.01 <= metrics['val_anomaly_rate'] <= 0.2  # Reasonable anomaly rate
        assert metrics['explained_variance_ratio'] > 0.8  # Good variance explanation
    
    def test_pca_kick_detection_prediction(self, sample_kick_data):
        """Test PCA kick detection prediction"""
        X, y = sample_kick_data
        
        model = PCAKickDetection()
        model.train(X, y)
        
        # Test prediction
        predictions = model.predict(X.head(50))
        
        assert len(predictions) == 50
        assert all(pred in [0, 1] for pred in predictions)  # Binary predictions
        
        # Test anomaly scores
        scores = model.get_anomaly_scores(X.head(50))
        assert len(scores) == 50
        assert all(score >= 0 for score in scores)  # Positive scores
    
    def test_ensemble_kick_detection(self, sample_kick_data):
        """Test ensemble kick detection"""
        X, y = sample_kick_data
        
        model = EnsembleKickDetection(['pca', 'isolation_forest'])
        metrics = model.train(X, y, validation_split=0.2)
        
        # Check if model is fitted
        assert model.is_fitted
        
        # Check ensemble metrics
        assert 'model_weights' in metrics
        assert len(metrics['model_weights']) == 2
        
        # Test predictions
        predictions = model.predict(X.head(20))
        assert len(predictions) == 20
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_kick_detection_performance(self, sample_kick_data):
        """Test kick detection performance metrics"""
        X, y = sample_kick_data
        
        model = PCAKickDetection()
        model.train(X, y)
        
        # Test on known data
        test_X = X.tail(100)
        test_y = y.tail(100)
        
        predictions = model.predict(test_X)
        
        # Calculate basic metrics
        true_positives = np.sum((predictions == 1) & (test_y == 1))
        false_positives = np.sum((predictions == 1) & (test_y == 0))
        false_negatives = np.sum((predictions == 0) & (test_y == 1))
        
        # For safety-critical application, we want high recall
        if np.sum(test_y == 1) > 0:  # If there are actual kicks in test set
            recall = true_positives / (true_positives + false_negatives)
            assert recall >= 0.8  # High recall is critical for safety

class TestModelUtilities:
    """Test model utility functions"""
    
    def test_model_comparison(self):
        """Test model comparison functionality"""
        # Create sample data
        np.random.seed(42)
        n_samples = 200
        
        X = pd.DataFrame({
            'WellDepth': np.random.normal(5000, 1000, n_samples),
            'WoBit': np.random.normal(25, 5, n_samples),
            'RoPen': np.random.normal(15, 3, n_samples),
            'BTBR': np.random.normal(120, 15, n_samples),
            'WBoPress': np.random.normal(2000, 200, n_samples),
            'HLoad': np.random.normal(150, 15, n_samples),
            'DPPress': np.random.normal(180, 20, n_samples)
        })
        
        y = pd.Series(
            0.01 * X['WellDepth'] + 0.05 * X['WBoPress'] + np.random.normal(0, 50, n_samples),
            name='FPress'
        )
        
        # Train multiple models
        pcr_model = PCRFormationPressure(n_components=3)
        pcr_model.train(X, y)
        
        xgb_model = XGBoostFormationPressure()
        xgb_model.train(X, y)
        
        # Compare models
        from src.models.formation_pressure import FormationPressureAnalyzer
        
        models = {'PCR': pcr_model, 'XGBoost': xgb_model}
        comparison = FormationPressureAnalyzer.compare_models(models, X.tail(50), y.tail(50))
        
        # Check comparison results
        assert len(comparison) == 2
        assert 'Model' in comparison.columns
        assert 'R²' in comparison.columns
        assert 'RMSE' in comparison.columns
        assert 'MAE' in comparison.columns
    
    def test_feature_importance_analysis(self):
        """Test feature importance analysis"""
        # Create sample data with clear feature importance
        np.random.seed(42)
        n_samples = 300
        
        X = pd.DataFrame({
            'ImportantFeature1': np.random.normal(0, 1, n_samples),
            'ImportantFeature2': np.random.normal(0, 1, n_samples),
            'NoiseFeature1': np.random.normal(0, 1, n_samples),
            'NoiseFeature2': np.random.normal(0, 1, n_samples),
        })
        
        # Create target with clear dependencies
        y = pd.Series(
            2 * X['ImportantFeature1'] + 3 * X['ImportantFeature2'] + 0.1 * np.random.normal(0, 1, n_samples),
            name='target'
        )
        
        # Train model
        model = XGBoostFormationPressure()
        model.train(X, y)
        
        # Get feature importance
        importance = model.get_feature_importance()
        
        # Check that important features are ranked higher
        assert importance['ImportantFeature1'] > importance['NoiseFeature1']
        assert importance['ImportantFeature2'] > importance['NoiseFeature2']

class TestModelValidation:
    """Test model validation and error handling"""
    
    def test_invalid_data_handling(self):
        """Test model behavior with invalid data"""
        # Create invalid data (NaN values, wrong shapes, etc.)
        
        # Test with NaN values
        X_invalid = pd.DataFrame({
            'WellDepth': [5000, np.nan, 5200],
            'WoBit': [25, 26, np.nan],
            'RoPen': [15, 16, 17]
        })
        
        y_invalid = pd.Series([2500, 2600, 2700])
        
        model = PCRFormationPressure()
        
        # Model should handle NaN values gracefully
        try:
            model.train(X_invalid, y_invalid)
            assert model.is_fitted
        except Exception as e:
            # If training fails, it should be for a good reason
            assert "NaN" in str(e) or "missing" in str(e).lower()
    
    def test_empty_data_handling(self):
        """Test model behavior with empty data"""
        empty_X = pd.DataFrame()
        empty_y = pd.Series([])
        
        model = PCRFormationPressure()
        
        with pytest.raises(Exception):
            model.train(empty_X, empty_y)
    
    def test_prediction_without_training(self):
        """Test that prediction fails on untrained model"""
        X = pd.DataFrame({
            'WellDepth': [5000],
            'WoBit': [25],
            'RoPen': [15]
        })
        
        model = PCRFormationPressure()
        
        with pytest.raises(ValueError, match="must be trained"):
            model.predict(X)
    
    def test_mismatched_features(self):
        """Test prediction with mismatched features"""
        # Train with one set of features
        np.random.seed(42)
        X_train = pd.DataFrame({
            'Feature1': np.random.normal(0, 1, 100),
            'Feature2': np.random.normal(0, 1, 100),
            'Feature3': np.random.normal(0, 1, 100)
        })
        y_train = pd.Series(np.random.normal(0, 1, 100))
        
        model = PCRFormationPressure()
        model.train(X_train, y_train)
        
        # Try to predict with different features
        X_predict = pd.DataFrame({
            'Feature1': [1.0],
            'Feature2': [1.0],
            'DifferentFeature': [1.0]  # Missing Feature3, extra DifferentFeature
        })
        
        with pytest.raises(ValueError, match="Missing features"):
            model.predict(X_predict)

class TestModelPerformance:
    """Test model performance requirements"""
    
    @pytest.mark.slow
    def test_training_time_requirements(self, sample_formation_data):
        """Test that model training completes in reasonable time"""
        import time
        
        X, y = sample_formation_data
        
        # PCR should train quickly
        start_time = time.time()
        pcr_model = PCRFormationPressure()
        pcr_model.train(X, y)
        pcr_time = time.time() - start_time
        
        assert pcr_time < 10  # Should complete in less than 10 seconds
        
        # XGBoost might take longer but should still be reasonable
        start_time = time.time()
        xgb_model = XGBoostFormationPressure()
        xgb_model.train(X, y)
        xgb_time = time.time() - start_time
        
        assert xgb_time < 30  # Should complete in less than 30 seconds
    
    def test_prediction_speed(self, sample_formation_data):
        """Test prediction speed requirements"""
        import time
        
        X, y = sample_formation_data
        
        model = PCRFormationPressure()
        model.train(X, y)
        
        # Test single prediction speed
        single_X = X.head(1)
        
        start_time = time.time()
        prediction = model.predict(single_X)
        prediction_time = time.time() - start_time
        
        # Single prediction should be very fast (< 0.1 seconds)
        assert prediction_time < 0.1
        
        # Test batch prediction speed
        batch_X = X.head(100)
        
        start_time = time.time()
        batch_predictions = model.predict(batch_X)
        batch_time = time.time() - start_time
        
        # Batch prediction should still be fast
        assert batch_time < 1.0
        assert len(batch_predictions) == 100
    
    def test_memory_usage(self, sample_formation_data):
        """Test reasonable memory usage"""
        import psutil
        import os
        
        X, y = sample_formation_data
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Train model
        model = EnsembleFormationPressure(['pcr', 'xgboost'])
        model.train(X, y)
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 500 MB for small dataset)
        assert memory_increase < 500

# Pytest configuration and fixtures
@pytest.fixture(scope="session")
def setup_test_environment():
    """Setup test environment"""
    # Create test directories
    test_dirs = ['test_data', 'test_models', 'test_outputs']
    for dir_name in test_dirs:
        Path(dir_name).mkdir(exist_ok=True)
    
    yield
    
    # Cleanup
    import shutil
    for dir_name in test_dirs:
        if Path(dir_name).exists():
            shutil.rmtree(dir_name)

if __name__ == "__main__":
    # Run tests when script is executed directly
    import sys
    extra_args = sys.argv[1:] if len(sys.argv) > 1 else []
    default_args = ["-v", "--tb=short"]
    # Run pytest
    import pytest
    sys.exit(pytest.main([__file__, *default_args, *extra_args]))
