"""
Base Model Class for ML Drilling Project
Provides common functionality for all drilling ML models
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import json
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import config

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract base class for all drilling ML models"""
    
    def __init__(self, model_name: str, model_type: str = 'regression'):
        """
        Initialize base model
        
        Args:
            model_name: Name of the model
            model_type: 'regression' or 'classification'
        """
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.is_fitted = False
        
        # Training history
        self.training_history = {
            'timestamp': None,
            'train_score': None,
            'val_score': None,
            'feature_importance': None,
            'hyperparameters': None,
            'data_shape': None
        }
        
        # Model configuration
        self.feature_columns = []
        self.target_column = None
        self.preprocessing_steps = []
        
        # Performance metrics
        self.metrics = {}
        
    @abstractmethod
    def _build_model(self, **kwargs) -> Any:
        """Build the specific model (must be implemented by subclasses)"""
        pass
    
    @abstractmethod
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default hyperparameters (must be implemented by subclasses)"""
        pass
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, 
                    validation_split: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training with time-based split
        
        Args:
            X: Feature matrix
            y: Target vector
            validation_split: Fraction for validation set
            
        Returns:
            X_train, X_val, y_train, y_val
        """
        # Store feature and target information
        self.feature_columns = X.columns.tolist()
        self.target_column = y.name if hasattr(y, 'name') else 'target'
        
        # Time-based split (important for drilling data)
        split_index = int(len(X) * (1 - validation_split))
        
        X_train = X.iloc[:split_index].copy()
        X_val = X.iloc[split_index:].copy()
        y_train = y.iloc[:split_index].copy()
        y_val = y.iloc[split_index:].copy()
        
        # Handle any remaining NaN values
        X_train = X_train.fillna(X_train.mean())
        X_val = X_val.fillna(X_train.mean())  # Use training mean for validation
        
        logger.info(f"Data prepared: Train={len(X_train)}, Val={len(X_val)}")
        
        return X_train, X_val, y_train, y_val
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
             validation_split: float = 0.2,
             **model_params) -> Dict[str, float]:
        """
        Train the model
        
        Args:
            X: Feature matrix
            y: Target vector
            validation_split: Validation set fraction
            **model_params: Model-specific parameters
            
        Returns:
            Training metrics dictionary
        """
        logger.info(f"Starting training for {self.model_name}")
        
        # Prepare data
        X_train, X_val, y_train, y_val = self.prepare_data(X, y, validation_split)
        
        # Build model with parameters
        params = self._get_default_params()
        params.update(model_params)
        
        self.model = self._build_model(**params)
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Evaluate on training and validation sets
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        # Calculate metrics
        if self.model_type == 'regression':
            train_score = r2_score(y_train, train_pred)
            val_score = r2_score(y_val, val_pred)
            
            self.metrics = {
                'train_r2': train_score,
                'val_r2': val_score,
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                'train_mae': mean_absolute_error(y_train, train_pred),
                'val_mae': mean_absolute_error(y_val, val_pred)
            }
            
        elif self.model_type == 'classification':
            train_score = self.model.score(X_train, y_train)
            val_score = self.model.score(X_val, y_val)
            
            # Get probability predictions if available
            if hasattr(self.model, 'predict_proba'):
                train_proba = self.model.predict_proba(X_train)[:, 1]
                val_proba = self.model.predict_proba(X_val)[:, 1]
                
                self.metrics = {
                    'train_accuracy': train_score,
                    'val_accuracy': val_score,
                    'train_auc': roc_auc_score(y_train, train_proba),
                    'val_auc': roc_auc_score(y_val, val_proba)
                }
            else:
                self.metrics = {
                    'train_accuracy': train_score,
                    'val_accuracy': val_score
                }
        
        # Store training history
        self.training_history.update({
            'timestamp': datetime.now().isoformat(),
            'train_score': train_score,
            'val_score': val_score,
            'hyperparameters': params,
            'data_shape': X.shape,
            'feature_columns': self.feature_columns
        })
        
        # Get feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.training_history['feature_importance'] = dict(
                zip(self.feature_columns, self.model.feature_importances_)
            )
        
        logger.info(f"Training completed. Val score: {val_score:.4f}")
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure feature columns match
        if not all(col in X.columns for col in self.feature_columns):
            missing_cols = [col for col in self.feature_columns if col not in X.columns]
            raise ValueError(f"Missing features: {missing_cols}")
        
        X_pred = X[self.feature_columns].copy()
        X_pred = X_pred.fillna(X_pred.mean())
        
        return self.model.predict(X_pred)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities (for classification)"""
        if self.model_type != 'classification':
            raise ValueError("predict_proba only available for classification models")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not support probability predictions")
        
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_pred = X[self.feature_columns].copy()
        X_pred = X_pred.fillna(X_pred.mean())
        
        return self.model.predict_proba(X_pred)
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      cv_folds: int = 5) -> Dict[str, float]:
        """Perform time series cross validation"""
        
        if not self.is_fitted:
            logger.warning("Model not fitted, using default parameters for CV")
            self.model = self._build_model(**self._get_default_params())
        
        # Use TimeSeriesSplit for time-aware cross validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        X_clean = X[self.feature_columns] if self.feature_columns else X
        X_clean = X_clean.fillna(X_clean.mean())
        
        if self.model_type == 'regression':
            scores = cross_val_score(self.model, X_clean, y, cv=tscv, scoring='r2')
            rmse_scores = -cross_val_score(self.model, X_clean, y, cv=tscv, 
                                         scoring='neg_root_mean_squared_error')
        else:
            scores = cross_val_score(self.model, X_clean, y, cv=tscv, scoring='accuracy')
            rmse_scores = None
        
        cv_results = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores.tolist()
        }
        
        if rmse_scores is not None:
            cv_results.update({
                'mean_rmse': rmse_scores.mean(),
                'std_rmse': rmse_scores.std()
            })
        
        logger.info(f"Cross validation completed: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
        
        return cv_results
    
    def get_feature_importance(self, top_k: int = 10) -> Dict[str, float]:
        """Get feature importance scores"""
        
        if not self.is_fitted:
            raise ValueError("Model must be trained to get feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_columns, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            # For linear models, use absolute coefficients
            importance_dict = dict(zip(self.feature_columns, np.abs(self.model.coef_)))
        else:
            logger.warning("Model does not support feature importance")
            return {}
        
        # Sort by importance
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
        )
        
        return sorted_importance
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        if self.model_type == 'regression':
            metrics = {
                'r2_score': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
            }
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        if self.model_type == 'regression':
            metrics = {
                'r2_score': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
            }
            
            # Additional drilling-specific metrics
            if 'FPress' in self.target_column or 'formation' in self.model_name.lower():
                # Formation pressure specific metrics
                metrics['accuracy_within_5pct'] = np.mean(np.abs((y_test - y_pred) / y_test) < 0.05) * 100
                metrics['accuracy_within_10pct'] = np.mean(np.abs((y_test - y_pred) / y_test) < 0.10) * 100
            
        elif self.model_type == 'classification':
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }
            
            # Add AUC if probabilities available
            if hasattr(self.model, 'predict_proba'):
                y_proba = self.predict_proba(X_test)
                if y_proba.shape[1] == 2:  # Binary classification
                    metrics['auc'] = roc_auc_score(y_test, y_proba[:, 1])
                else:  # Multi-class
                    metrics['auc'] = roc_auc_score(y_test, y_proba, multi_class='ovr')
        
        logger.info(f"Model evaluation completed: {metrics}")
        return metrics
    
    def save_model(self, filepath: str = None) -> str:
        """Save model to disk"""
        
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        if filepath is None:
            # Create default filepath
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = config.get_model_path(f"{self.model_name}_{timestamp}.pkl")
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'training_history': self.training_history,
            'metrics': self.metrics,
            'preprocessing_steps': self.preprocessing_steps
        }
        
        joblib.dump(model_data, filepath)
        
        logger.info(f"Model saved to: {filepath}")
        return filepath
    
    def load_model(self, filepath: str) -> None:
        """Load model from disk"""
        
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.model_name = model_data['model_name']
            self.model_type = model_data['model_type']
            self.feature_columns = model_data['feature_columns']
            self.target_column = model_data['target_column']
            self.training_history = model_data['training_history']
            self.metrics = model_data.get('metrics', {})
            self.preprocessing_steps = model_data.get('preprocessing_steps', [])
            
            self.is_fitted = True
            
            logger.info(f"Model loaded from: {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        
        info = {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'is_fitted': self.is_fitted,
            'n_features': len(self.feature_columns),
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'training_history': self.training_history,
            'metrics': self.metrics,
            'preprocessing_steps': self.preprocessing_steps
        }
        
        # Add model-specific info
        if self.is_fitted and hasattr(self.model, 'get_params'):
            info['model_parameters'] = self.model.get_params()
        
        return info
    
    def export_model_report(self, filepath: str = None) -> str:
        """Export detailed model report"""
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = config.get_output_path("reports") / f"{self.model_name}_report_{timestamp}.json"
        
        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Get comprehensive model info
        report = self.get_model_info()
        
        # Add feature importance if available
        if self.is_fitted:
            try:
                report['feature_importance'] = self.get_feature_importance(top_k=20)
            except:
                pass
        
        # Save report
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Model report exported to: {filepath}")
        return str(filepath)

class ModelValidator:
    """Model validation utilities"""
    
    @staticmethod
    def validate_drilling_model(model: BaseModel, X_test: pd.DataFrame, 
                              y_test: pd.Series, model_type: str = 'formation') -> Dict[str, Any]:
        """Validate drilling model with domain-specific checks"""
        
        validation_results = {
            'basic_metrics': model.evaluate_model(X_test, y_test),
            'domain_checks': {},
            'warnings': [],
            'passed': True
        }
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        if model_type == 'formation' and model.model_type == 'regression':
            # Formation pressure specific validation
            
            # Check prediction range
            pred_min, pred_max = y_pred.min(), y_pred.max()
            actual_min, actual_max = y_test.min(), y_test.max()
            
            if pred_min < actual_min * 0.5 or pred_max > actual_max * 1.5:
                validation_results['warnings'].append("Predictions outside reasonable range")
            
            # Check for negative predictions (impossible for pressure)
            if (y_pred < 0).any():
                validation_results['warnings'].append("Negative pressure predictions found")
                validation_results['passed'] = False
            
            # Formation pressure gradient check
            if 'WellDepth' in X_test.columns:
                pressure_gradient = y_pred / X_test['WellDepth']
                if (pressure_gradient > 1.5).any() or (pressure_gradient < 0.3).any():
                    validation_results['warnings'].append("Unusual pressure gradients detected")
            
            # Accuracy within drilling tolerances
            tolerance_5pct = np.mean(np.abs((y_test - y_pred) / y_test) < 0.05) * 100
            tolerance_10pct = np.mean(np.abs((y_test - y_pred) / y_test) < 0.10) * 100
            
            validation_results['domain_checks'] = {
                'accuracy_within_5pct': tolerance_5pct,
                'accuracy_within_10pct': tolerance_10pct,
                'pred_range': (pred_min, pred_max),
                'negative_predictions': (y_pred < 0).sum()
            }
            
            # Pass/fail criteria
            if tolerance_10pct < 70:  # Less than 70% within 10% tolerance
                validation_results['warnings'].append("Low accuracy for drilling operations")
                validation_results['passed'] = False
        
        elif model_type == 'kick' and model.model_type == 'classification':
            # Kick detection specific validation
            
            from sklearn.metrics import precision_score, recall_score
            
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            
            validation_results['domain_checks'] = {
                'precision': precision,
                'recall': recall,
                'false_positive_rate': 1 - precision,
                'false_negative_rate': 1 - recall
            }
            
            # Critical for safety - high recall needed
            if recall < 0.8:
                validation_results['warnings'].append("Low recall - may miss kicks (safety concern)")
                validation_results['passed'] = False
            
            # False positives should be acceptable if recall is high
            if precision < 0.3:
                validation_results['warnings'].append("High false positive rate")
        
        return validation_results

class ModelEnsemble:
    """Ensemble methods for drilling models"""
    
    def __init__(self, models: List[BaseModel], weights: List[float] = None):
        """
        Initialize ensemble
        
        Args:
            models: List of trained models
            weights: Optional weights for weighted average
        """
        self.models = models
        self.weights = weights if weights else [1.0] * len(models)
        self.is_fitted = all(model.is_fitted for model in models)
        
        # Validate models
        model_types = set(model.model_type for model in models)
        if len(model_types) > 1:
            raise ValueError("All models must be same type (regression/classification)")
        
        self.model_type = list(model_types)[0]
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions"""
        
        if not self.is_fitted:
            raise ValueError("All models must be fitted")
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        
        return ensemble_pred
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Ensemble probability predictions"""
        
        if self.model_type != 'classification':
            raise ValueError("predict_proba only for classification")
        
        # Get probabilities from all models
        probabilities = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                probabilities.append(proba)
        
        if not probabilities:
            raise ValueError("No models support probability predictions")
        
        # Weighted average of probabilities
        ensemble_proba = np.average(probabilities, axis=0, weights=self.weights[:len(probabilities)])
        
        return ensemble_proba
    
    def get_model_contributions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get individual model contributions to ensemble"""
        
        contributions = {}
        for i, model in enumerate(self.models):
            model_name = f"{model.model_name}_{i}"
            contributions[model_name] = {
                'prediction': model.predict(X),
                'weight': self.weights[i]
            }
        
        return contributions

if __name__ == "__main__":
    # Test base model functionality
    print("Base model module loaded successfully!")
    print("This module provides the foundation for all drilling ML models.")
    
    # Example of how to create a simple model class
    from sklearn.linear_model import LinearRegression
    
    class SimpleLinearModel(BaseModel):
        def __init__(self):
            super().__init__("Simple_Linear", "regression")
        
        def _build_model(self, **kwargs):
            return LinearRegression(**kwargs)
        
        def _get_default_params(self):
            return {}
    
    # Test with sample data
    try:
        # Create sample data
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100)
        })
        y = pd.Series(2 * X['feature1'] + 3 * X['feature2'] + np.random.normal(0, 0.1, 100))
        
        # Test model
        model = SimpleLinearModel()
        metrics = model.train(X, y)
        
        print(f"Sample model training completed!")
        print(f"Validation R²: {metrics['val_r2']:.4f}")
        
        # Test predictions
        predictions = model.predict(X.head())
        print(f"Sample predictions: {predictions[:3]}")
        
    except Exception as e:
        print(f"Error in base model test: {str(e)}")