"""
Formation Pressure Prediction Models
Advanced ML models for predicting formation pressure in drilling operations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from scipy.signal import savgol_filter
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models.base_model import BaseModel
from utils.config import config
import logging

logger = logging.getLogger(__name__)

class PCRFormationPressure(BaseModel):
    """Principal Component Regression for Formation Pressure Prediction"""
    
    def __init__(self, n_components: int = None):
        super().__init__("PCR_Formation_Pressure", "regression")
        self.n_components = n_components or config.model.formation_n_components
        self.pca = None
        self.scaler = None
        self.regressor = None
        
    def _build_model(self, **kwargs) -> Dict[str, Any]:
        """Build PCR model components"""
        
        n_comp = kwargs.get('n_components', self.n_components)
        
        # Initialize components
        self.pca = PCA(n_components=n_comp)
        self.scaler = StandardScaler()
        self.regressor = LinearRegression()
        
        return {
            'pca': self.pca,
            'scaler': self.scaler,
            'regressor': self.regressor,
            'n_components': n_comp
        }
    
    def _get_default_params(self) -> Dict[str, Any]:
        return {
            'n_components': self.n_components
        }
    
    def _preprocess_data(self, X: pd.DataFrame, apply_smoothing: bool = True) -> pd.DataFrame:
        """Preprocess data with smoothing and standardization"""
        
        X_processed = X.copy()
        
        if apply_smoothing:
            # Apply Savitzky-Golay smoothing
            for col in X_processed.select_dtypes(include=[np.number]).columns:
                if len(X_processed[col].dropna()) > 7:
                    try:
                        X_processed[col] = savgol_filter(
                            X_processed[col].fillna(method='ffill'), 
                            window_length=7, 
                            polyorder=2
                        )
                    except:
                        logger.warning(f"Could not smooth column: {col}")
        
        return X_processed
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
             validation_split: float = 0.2,
             **model_params) -> Dict[str, float]:
        """Train PCR model"""
        
        logger.info("Starting PCR Formation Pressure training...")
        
        # Preprocess data
        X_processed = self._preprocess_data(X)
        
        # Prepare data splits
        X_train, X_val, y_train, y_val = self.prepare_data(X_processed, y, validation_split)
        
        # Build model
        self._build_model(**model_params)
        
        # Step 1: Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Step 2: Apply PCA
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_val_pca = self.pca.transform(X_val_scaled)
        
        # Step 3: Fit regression on principal components
        self.regressor.fit(X_train_pca, y_train)
        
        # Mark as fitted
        self.is_fitted = True
        
        # Make predictions
        y_train_pred = self.regressor.predict(X_train_pca)
        y_val_pred = self.regressor.predict(X_val_pca)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        self.metrics = {
            'train_r2': train_r2,
            'val_r2': val_r2,
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'n_components': self.n_components,
            'explained_variance_ratio': sum(self.pca.explained_variance_ratio_)
        }
        
        logger.info(f"PCR training completed. Val R²: {self.metrics['val_r2']:.4f}")
        logger.info(f"Explained variance: {self.metrics['explained_variance_ratio']:.4f}")
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make PCR predictions"""
        
        if not self.is_fitted:
            raise ValueError("PCR model must be trained before making predictions")
        
        # Preprocess data
        X_processed = self._preprocess_data(X)
        
        # Transform data
        X_scaled = self.scaler.transform(X_processed)
        X_pca = self.pca.transform(X_scaled)
        
        # Make predictions
        predictions = self.regressor.predict(X_pca)
        
        return predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from PCR model"""
        
        if not self.is_fitted:
            raise ValueError("Model must be trained before getting feature importance")
        
        # For PCR, we can reconstruct the coefficients in original feature space
        # PCA components transformed back to original features
        pca_components = self.pca.components_  # shape: [n_components, n_features]
        regression_coef = self.regressor.coef_  # shape: [n_components] or [1, n_components]
        
        # Transform coefficients back to original feature space
        if len(regression_coef.shape) == 1:
            # Single output
            original_coef = np.dot(regression_coef, pca_components)
        else:
            # Multiple outputs (if ever needed)
            original_coef = np.dot(regression_coef, pca_components)[0]
        
        # Create feature importance dictionary
        feature_importance = {}
        for i, feature in enumerate(self.feature_names if hasattr(self, 'feature_names') 
                                   else range(len(original_coef))):
            feature_importance[str(feature)] = abs(original_coef[i])
        
        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        return feature_importance
    
    def get_pca_components(self) -> pd.DataFrame:
        """Get PCA component loadings"""
        
        if not self.is_fitted:
            raise ValueError("Model must be trained before getting PCA components")
        
        components_df = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
            index=self.feature_names if hasattr(self, 'feature_names') 
                   else [f'Feature_{i}' for i in range(self.pca.n_features_in_)]
        )
        
        return components_df
    
    def get_explained_variance(self) -> Dict[str, float]:
        """Get explained variance for each principal component"""
        
        if not self.is_fitted:
            raise ValueError("Model must be trained before getting explained variance")
        
        explained_variance = {
            f'PC{i+1}': ratio 
            for i, ratio in enumerate(self.pca.explained_variance_ratio_)
        }
        
        return explained_variance

class FormationPressureAnalyzer:
    """Analysis tools for formation pressure predictions"""
    
    @staticmethod
    def analyze_pressure_trends(predictions: np.ndarray, 
                              depths: np.ndarray) -> Dict[str, float]:
        """Analyze formation pressure trends with depth"""
        
        # Calculate pressure gradient
        pressure_gradient = np.gradient(predictions, depths)
        
        # Normal hydrostatic gradient is ~0.433 psi/ft
        normal_gradient = 0.433
        
        analysis = {
            'mean_gradient': np.mean(pressure_gradient),
            'std_gradient': np.std(pressure_gradient),
            'normal_gradient': normal_gradient,
            'overpressure_zones': np.sum(pressure_gradient > normal_gradient * 1.2),
            'underpressure_zones': np.sum(pressure_gradient < normal_gradient * 0.8),
            'gradient_range': (np.min(pressure_gradient), np.max(pressure_gradient))
        }
        
        return analysis
    
    @staticmethod
    def detect_pressure_anomalies(predictions: np.ndarray, 
                                 threshold: float = 2.0) -> Dict[str, Any]:
        """Detect pressure anomalies using statistical methods"""
        
        # Z-score based anomaly detection
        z_scores = np.abs((predictions - np.mean(predictions)) / np.std(predictions))
        anomaly_indices = np.where(z_scores > threshold)[0]
        
        # Rate of change anomalies
        pressure_changes = np.diff(predictions)
        change_z_scores = np.abs((pressure_changes - np.mean(pressure_changes)) / np.std(pressure_changes))
        rapid_change_indices = np.where(change_z_scores > threshold)[0]
        
        return {
            'anomaly_count': len(anomaly_indices),
            'anomaly_indices': anomaly_indices.tolist(),
            'rapid_change_count': len(rapid_change_indices),
            'rapid_change_indices': rapid_change_indices.tolist(),
            'max_z_score': np.max(z_scores),
            'anomaly_values': predictions[anomaly_indices].tolist()
        }
    
    @staticmethod
    def compare_models(models: Dict[str, BaseModel], 
                      X_test: pd.DataFrame, 
                      y_test: pd.Series) -> pd.DataFrame:
        """Compare multiple formation pressure models"""
        
        comparison_data = []
        
        for name, model in models.items():
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
                
                # Drilling-specific metrics
                accuracy_5pct = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8)) < 0.05) * 100
                accuracy_10pct = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8)) < 0.10) * 100
                
                comparison_data.append({
                    'Model': name,
                    'R²': r2,
                    'RMSE': rmse,
                    'MAE': mae,
                    'MAPE (%)': mape,
                    'Accuracy_5%': accuracy_5pct,
                    'Accuracy_10%': accuracy_10pct,
                    'Training_Time': getattr(model, 'training_time', 'N/A')
                })
                
            except Exception as e:
                logger.error(f"Error evaluating {name}: {str(e)}")
        
        return pd.DataFrame(comparison_data).sort_values('R²', ascending=False)
    
    @staticmethod
    def calculate_drilling_efficiency_metrics(predictions: np.ndarray, 
                                            actual: np.ndarray,
                                            drilling_params: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate drilling efficiency metrics based on formation pressure predictions"""
        
        metrics = {}
        
        # Prediction accuracy metrics
        prediction_error = np.abs(predictions - actual)
        metrics['mean_absolute_error'] = np.mean(prediction_error)
        metrics['max_error'] = np.max(prediction_error)
        metrics['rmse'] = np.sqrt(np.mean((predictions - actual) ** 2))
        
        # Operational efficiency metrics
        if 'mud_weight' in drilling_params:
            mud_weight = drilling_params['mud_weight']
            
            # Calculate optimal mud weight (should be slightly above formation pressure)
            optimal_mud_weight = predictions * 0.052 + 0.5  # Convert to ppg with safety margin
            
            # Efficiency metrics
            overbalance = mud_weight - (predictions * 0.052)
            metrics['mean_overbalance'] = np.mean(overbalance)
            metrics['overbalance_efficiency'] = np.mean((overbalance > 0.5) & (overbalance < 2.0))
            
            # Safety metrics
            metrics['underbalanced_risk'] = np.mean(overbalance < 0.5)
            metrics['excessive_overbalance'] = np.mean(overbalance > 3.0)
        
        # Formation pressure gradient analysis
        if len(predictions) > 1:
            gradients = np.diff(predictions)
            metrics['pressure_stability'] = 1.0 / (1.0 + np.std(gradients))
            metrics['gradient_consistency'] = np.mean(np.abs(gradients - np.mean(gradients)) < np.std(gradients))
        
        return metrics
    
    @staticmethod
    def generate_drilling_recommendations(predictions: np.ndarray,
                                        depths: np.ndarray,
                                        current_params: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate drilling parameter recommendations based on formation pressure predictions"""
        
        recommendations = []
        
        # Analyze pressure trends
        pressure_gradient = np.gradient(predictions, depths)
        mean_gradient = np.mean(pressure_gradient)
        
        # Mud weight recommendations
        current_mud_weight = current_params.get('mud_weight', 10.0)
        predicted_pressure_ppg = predictions[-1] * 0.052  # Convert to ppg equivalent
        
        if predicted_pressure_ppg > current_mud_weight - 0.5:
            recommendations.append({
                'type': 'mud_weight',
                'priority': 'HIGH',
                'current_value': current_mud_weight,
                'recommended_value': predicted_pressure_ppg + 1.0,
                'reason': 'Formation pressure approaching mud weight - increase for safety',
                'action': 'Increase mud weight to maintain adequate overbalance'
            })
        
        # ROP recommendations based on formation strength
        if mean_gradient > 0.5:  # High pressure gradient indicates harder formation
            recommendations.append({
                'type': 'drilling_parameters',
                'priority': 'MEDIUM',
                'parameter': 'WOB',
                'recommendation': 'increase',
                'reason': 'High pressure gradient indicates harder formation',
                'action': 'Consider increasing WOB for better penetration'
            })
        
        # Flow rate recommendations
        if np.max(predictions) > 5000:  # High pressure formation
            recommendations.append({
                'type': 'circulation',
                'priority': 'MEDIUM',
                'parameter': 'flow_rate',
                'recommendation': 'optimize',
                'reason': 'High formation pressure detected',
                'action': 'Optimize flow rate for better hole cleaning and pressure control'
            })
        
        # Early warning for rapid pressure changes
        if len(pressure_gradient) > 10:
            recent_gradient = pressure_gradient[-10:]
            if np.std(recent_gradient) > np.std(pressure_gradient) * 1.5:
                recommendations.append({
                    'type': 'alert',
                    'priority': 'HIGH',
                    'reason': 'Rapid formation pressure changes detected',
                    'action': 'Monitor closely for potential formation change or drilling issues'
                })
        
        return recommendations

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Configure logging
logger = logging.getLogger(__name__)

class FormationPressureOptimizer:
    """Optimization tools for formation pressure models"""
    
    @staticmethod
    def optimize_hyperparameters(X: pd.DataFrame, y: pd.Series, 
                               model_type: str = 'xgboost',
                               n_trials: int = 100) -> Tuple[Any, Dict[str, Any]]:
        """Optimize hyperparameters using Optuna"""
        
        try:
            import optuna
        except ImportError:
            logger.error("Optuna not installed. Install with: pip install optuna")
            raise ImportError("Optuna not installed")
        
        # Define the model classes (you'll need to define these or import them)
        # For now, let's create simplified versions
        class XGBoostFormationPressure:
            def __init__(self, **kwargs):
                import xgboost as xgb
                self.model = xgb.XGBRegressor(**kwargs, random_state=42)
            
            def train(self, X, y, **params):
                # Update model parameters if provided
                if params:
                    self.model.set_params(**params)
                
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
                self.model.fit(X_train, y_train)
                
                y_pred = self.model.predict(X_val)
                val_r2 = r2_score(y_val, y_pred)
                return {'val_r2': val_r2}
            
            def predict(self, X):
                return self.model.predict(X)
        
        class RandomForestFormationPressure:
            def __init__(self, **kwargs):
                from sklearn.ensemble import RandomForestRegressor
                self.model = RandomForestRegressor(**kwargs, random_state=42)
            
            def train(self, X, y, **params):
                if params:
                    self.model.set_params(**params)
                
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
                self.model.fit(X_train, y_train)
                
                y_pred = self.model.predict(X_val)
                val_r2 = r2_score(y_val, y_pred)
                return {'val_r2': val_r2}
            
            def predict(self, X):
                return self.model.predict(X)
        
        class PCRFormationPressure:
            def __init__(self, **kwargs):
                from sklearn.decomposition import PCA
                from sklearn.linear_model import LinearRegression
                self.n_components = kwargs.get('n_components', 4)
                self.pca = PCA(n_components=self.n_components)
                self.regressor = LinearRegression()
            
            def train(self, X, y, **params):
                if 'n_components' in params:
                    self.n_components = params['n_components']
                    self.pca = PCA(n_components=self.n_components)
                
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Apply PCA
                X_train_pca = self.pca.fit_transform(X_train)
                X_val_pca = self.pca.transform(X_val)
                
                # Train regression
                self.regressor.fit(X_train_pca, y_train)
                
                # Calculate metrics
                y_pred = self.regressor.predict(X_val_pca)
                val_r2 = r2_score(y_val, y_pred)
                
                return {'val_r2': val_r2}
            
            def predict(self, X):
                X_pca = self.pca.transform(X)
                return self.regressor.predict(X_pca)
        
        def objective(trial):
            if model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                }
                model = XGBoostFormationPressure(**params)
                
            elif model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
                }
                model = RandomForestFormationPressure(**params)
                
            elif model_type == 'pcr':
                params = {
                    'n_components': trial.suggest_int('n_components', 2, min(len(X.columns), 20))
                }
                model = PCRFormationPressure(**params)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Train model with cross-validation
            try:
                metrics = model.train(X, y, **params)
                return metrics['val_r2']
            except Exception as e:
                logger.warning(f"Trial failed: {str(e)}")
                return -10.0  # Return a very low score for failed trials
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Train best model on full data
        best_params = study.best_params
        
        if model_type == 'xgboost':
            best_model = XGBoostFormationPressure(**best_params)
        elif model_type == 'random_forest':
            best_model = RandomForestFormationPressure(**best_params)
        elif model_type == 'pcr':
            best_model = PCRFormationPressure(**best_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train on full dataset
        best_model.train(X, y, **best_params)
        
        optimization_results = {
            'best_params': best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials),
            'optimization_history': [(trial.value, trial.params) for trial in study.trials if trial.value is not None]
        }
        
        logger.info(f"Optimization completed. Best R²: {study.best_value:.4f}")
        logger.info(f"Best params: {best_params}")
        
        return best_model, optimization_results

    @staticmethod
    def optimize_multiple_models(X: pd.DataFrame, y: pd.Series, 
                                model_types: list = None,
                                n_trials: int = 50) -> Dict[str, Any]:
        """Optimize multiple models and compare results"""
        
        if model_types is None:
            model_types = ['xgboost', 'random_forest', 'pcr']
        
        results = {}
        
        for model_type in model_types:
            print(f"\n--- Optimizing {model_type} ---")
            try:
                model, optimization_info = FormationPressureOptimizer.optimize_hyperparameters(
                    X, y, model_type=model_type, n_trials=n_trials
                )
                results[model_type] = {
                    'model': model,
                    'optimization_info': optimization_info,
                    'best_score': optimization_info['best_score']
                }
            except Exception as e:
                logger.error(f"Failed to optimize {model_type}: {str(e)}")
                results[model_type] = {'error': str(e)}
        
        # Sort by best score
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        if successful_results:
            best_model_type = max(successful_results.keys(), 
                                key=lambda x: successful_results[x]['best_score'])
            print(f"\n🎯 Best model: {best_model_type} (R²: {successful_results[best_model_type]['best_score']:.4f})")
        
        return results
    
    @staticmethod
    def feature_importance_analysis(model: BaseModel, X: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive feature importance analysis"""
        
        if not model.is_fitted:
            raise ValueError("Model must be trained first")
        
        analysis = {
            'model_feature_importance': {},
            'permutation_importance': {},
            'correlation_analysis': {},
            'drilling_domain_importance': {}
        }
        
        # Model-specific feature importance
        if hasattr(model, 'get_feature_importance'):
            analysis['model_feature_importance'] = model.get_feature_importance()
        
        # Permutation importance
        try:
            from sklearn.inspection import permutation_importance
            
            # Get sample for permutation importance (expensive computation)
            X_sample = X.sample(min(1000, len(X)))
            y_sample = model.predict(X_sample)
            
            perm_importance = permutation_importance(
                model.model, X_sample, y_sample, 
                n_repeats=5, random_state=42
            )
            
            analysis['permutation_importance'] = dict(
                zip(model.feature_columns, perm_importance.importances_mean)
            )
            
        except Exception as e:
            logger.warning(f"Could not calculate permutation importance: {str(e)}")
        
        # Domain knowledge importance
        drilling_importance = {
            'WoBit': 'Critical - Direct impact on drilling performance',
            'RoPen': 'Critical - Key performance indicator', 
            'BTBR': 'High - Torque indicates formation resistance',
            'WBoPress': 'High - Wellbore pressure management',
            'FPress': 'Target - Formation pressure is what we predict',
            'WellDepth': 'High - Depth significantly affects pressure',
            'DPPress': 'Medium - Differential pressure indicator',
            'HLoad': 'Medium - Mechanical load indicator'
        }
        
        analysis['drilling_domain_importance'] = drilling_importance
        
        return analysis

from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Set up logging
logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Base class for all formation pressure models"""
    
    def __init__(self):
        self.is_fitted = False
        self.metrics = {}
    
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    def train(self, X, y, val_size=0.2):
        """Convenience method that includes validation split"""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, random_state=42
        )
        return self.fit(X_train, y_train, X_val, y_val)

class PCRFormationPressure(BaseModel):
    """Principal Component Regression for formation pressure"""
    
    def __init__(self, n_components: int = 4):
        super().__init__()
        self.n_components = n_components
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components)),
            ('regressor', LinearRegression())
        ])
    
    def fit(self, X, y, X_val=None, y_val=None):
        self.pipeline.fit(X, y)
        self.is_fitted = True
        
        # Calculate metrics if validation data provided
        if X_val is not None and y_val is not None:
            y_pred = self.predict(X_val)
            self.metrics = {
                'val_r2': r2_score(y_val, y_pred),
                'val_mae': mean_absolute_error(y_val, y_pred),
                'explained_variance_ratio': sum(self.pipeline.named_steps['pca'].explained_variance_ratio_)
            }
        
        return self.metrics
    
    def predict(self, X):
        return self.pipeline.predict(X)

class XGBoostFormationPressure(BaseModel):
    """XGBoost model for formation pressure prediction"""
    
    def __init__(self, **kwargs):
        super().__init__()
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
        default_params.update(kwargs)
        self.model = xgb.XGBRegressor(**default_params)
    
    def fit(self, X, y, X_val=None, y_val=None):
        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
        self.model.fit(X, y, eval_set=eval_set, verbose=False)
        self.is_fitted = True
        
        if X_val is not None and y_val is not None:
            y_pred = self.predict(X_val)
            self.metrics = {
                'val_r2': r2_score(y_val, y_pred),
                'val_mae': mean_absolute_error(y_val, y_pred)
            }
        
        return self.metrics
    
    def predict(self, X):
        return self.model.predict(X)

class RandomForestFormationPressure(BaseModel):
    """Random Forest for formation pressure prediction"""
    
    def __init__(self, **kwargs):
        super().__init__()
        default_params = {
            'n_estimators': 100,
            'random_state': 42
        }
        default_params.update(kwargs)
        self.model = RandomForestRegressor(**default_params)
    
    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)
        self.is_fitted = True
        
        if X_val is not None and y_val is not None:
            y_pred = self.predict(X_val)
            self.metrics = {
                'val_r2': r2_score(y_val, y_pred),
                'val_mae': mean_absolute_error(y_val, y_pred)
            }
        
        return self.metrics
    
    def predict(self, X):
        return self.model.predict(X)

class PLSFormationPressure(BaseModel):
    """Partial Least Squares for formation pressure prediction"""
    
    def __init__(self, n_components: int = 4):
        super().__init__()
        self.n_components = n_components
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pls', PLSRegression(n_components=n_components))
        ])
    
    def fit(self, X, y, X_val=None, y_val=None):
        self.pipeline.fit(X, y)
        self.is_fitted = True
        
        if X_val is not None and y_val is not None:
            y_pred = self.predict(X_val)
            self.metrics = {
                'val_r2': r2_score(y_val, y_pred),
                'val_mae': mean_absolute_error(y_val, y_pred)
            }
        
        return self.metrics
    
    def predict(self, X):
        return self.pipeline.predict(X)

class EnsembleFormationPressure(BaseModel):
    """Advanced ensemble model with proper training metrics and validation"""
    
    def __init__(self, model_types: List[str]):
        super().__init__()
        self.models = {}
        self.model_weights = {}
        
        for model_type in model_types:
            if model_type == 'pcr':
                self.models[model_type] = PCRFormationPressure()
            elif model_type == 'xgboost':
                self.models[model_type] = XGBoostFormationPressure()
            elif model_type == 'random_forest':
                self.models[model_type] = RandomForestFormationPressure()
            elif model_type == 'pls':
                self.models[model_type] = PLSFormationPressure()
            else:
                raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X, y, X_val=None, y_val=None):
        """Train ensemble with validation and metrics tracking"""
        
        # Split data if validation set not provided
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            X_train, y_train = X, y
        
        # Train individual models and collect performances
        model_performances = {}
        all_val_predictions = {}
        
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name} model...")
                model.fit(X_train, y_train, X_val, y_val)
                
                # Get predictions for metrics
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                
                # Calculate metrics
                train_mae = mean_absolute_error(y_train, y_train_pred)
                val_mae = mean_absolute_error(y_val, y_val_pred)
                train_r2 = r2_score(y_train, y_train_pred)
                val_r2 = r2_score(y_val, y_val_pred)
                
                model_performances[name] = {
                    'train_mae': train_mae,
                    'val_mae': val_mae,
                    'train_r2': train_r2,
                    'val_r2': val_r2,
                    'y_val_pred': y_val_pred
                }
                
                all_val_predictions[name] = y_val_pred
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                model_performances[name] = {'error': str(e)}
        
        # Calculate model weights based on validation performance
        self.model_weights = self._calculate_weights(model_performances)
        
        # Calculate ensemble predictions for final metrics
        y_train_pred = self._get_ensemble_predictions(X_train, model_performances)
        y_val_pred = self._get_ensemble_predictions(X_val, model_performances)
        
        # Ensemble training metrics
        self.metrics = {
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'val_r2': r2_score(y_val, y_val_pred),
            'model_weights': self.model_weights,
            'individual_performances': model_performances
        }
        
        self.is_fitted = True
        
        logger.info(f"Ensemble training completed. Val R²: {self.metrics.get('val_r2', float('nan')):.4f}")
        logger.info(f"Model weights: {self.model_weights}")
        
        return self.metrics
    
    def _calculate_weights(self, model_performances: Dict) -> Dict[str, float]:
        """Calculate model weights based on validation performance"""
        weights = {}
        total_weight = 0
        
        for name, perf in model_performances.items():
            if 'val_mae' in perf and not np.isnan(perf['val_mae']):
                # Lower MAE = higher weight (inverse weighting)
                weight = 1.0 / (perf['val_mae'] + 1e-8)
                weights[name] = weight
                total_weight += weight
            else:
                weights[name] = 0.0
        
        # Normalize weights to sum to 1
        if total_weight > 0:
            for name in weights:
                weights[name] /= total_weight
        
        return weights
    
    def _get_ensemble_predictions(self, X: pd.DataFrame, model_performances: Dict) -> np.ndarray:
        """Get weighted ensemble predictions"""
        predictions = []
        weights = []
        
        for name, perf in model_performances.items():
            if 'val_mae' in perf and self.model_weights.get(name, 0) > 0:
                try:
                    pred = self.models[name].predict(X)
                    predictions.append(pred)
                    weights.append(self.model_weights[name])
                except Exception as e:
                    logger.warning(f"Model {name} prediction failed: {str(e)}")
        
        if not predictions:
            raise ValueError("No models available for prediction")
        
        return np.average(predictions, axis=0, weights=weights)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be trained before making predictions")
        
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            if model.is_fitted and self.model_weights.get(name, 0) > 0:
                try:
                    pred = model.predict(X)
                    predictions.append(pred)
                    weights.append(self.model_weights[name])
                except Exception as e:
                    logger.warning(f"Model {name} prediction failed: {str(e)}")
        
        if not predictions:
            raise ValueError("No models available for prediction")
        
        return np.average(predictions, axis=0, weights=weights)
    
    def get_model_contributions(self, X: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Get individual model contributions to ensemble"""
        contributions = {}
        
        for name, model in self.models.items():
            if model.is_fitted:
                try:
                    pred = model.predict(X)
                    contributions[name] = {
                        'predictions': pred,
                        'weight': self.model_weights.get(name, 0),
                        'metrics': model.metrics
                    }
                except Exception as e:
                    logger.warning(f"Could not get contributions for {name}: {str(e)}")
        
        return contributions

def create_formation_pressure_pipeline(data_type: str = 'advanced') -> BaseModel:
    """Create formation pressure prediction pipeline"""
    
    if data_type == 'simple':
        return PCRFormationPressure(n_components=4)
    
    elif data_type == 'advanced':
        return EnsembleFormationPressure(['pcr', 'xgboost', 'random_forest'])
    
    elif data_type == 'production':
        return XGBoostFormationPressure()
    
    elif data_type == 'research':
        return EnsembleFormationPressure(['pcr', 'xgboost', 'random_forest', 'pls'])
    
    else:
        raise ValueError(f"Unknown pipeline type: {data_type}")

def batch_predict_formation_pressure(model: BaseModel, 
                                   data_files: List[str],
                                   output_path: str) -> Dict[str, Any]:
    """Batch prediction for multiple data files"""
    
    results = {
        'predictions': {},
        'summary': {},
        'errors': []
    }
    
    for file_path in data_files:
        try:
            # Load data
            data = pd.read_csv(file_path)
            
            # Make predictions
            predictions = model.predict(data)
            
            # Store results
            file_name = Path(file_path).stem
            results['predictions'][file_name] = predictions.tolist()
            
            # Calculate summary statistics
            results['summary'][file_name] = {
                'count': len(predictions),
                'mean': np.mean(predictions),
                'std': np.std(predictions),
                'min': np.min(predictions),
                'max': np.max(predictions)
            }
            
        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            results['errors'].append(error_msg)
            logger.error(error_msg)
    
    # Save results
    import json
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Batch prediction results saved to: {output_path}")
    
    return results

class FormationPressureAnalyzer:
    """Analysis tools for formation pressure predictions"""
    
    @staticmethod
    def analyze_pressure_trends(predictions: np.ndarray, 
                              depths: np.ndarray) -> Dict[str, float]:
        """Analyze formation pressure trends with depth"""
        
        if len(predictions) != len(depths):
            raise ValueError("Predictions and depths must have same length")
        
        # Calculate pressure gradient
        pressure_gradient = np.gradient(predictions, depths)
        
        # Normal hydrostatic gradient is ~0.433 psi/ft
        normal_gradient = 0.433
        
        analysis = {
            'mean_gradient': np.mean(pressure_gradient),
            'std_gradient': np.std(pressure_gradient),
            'normal_gradient': normal_gradient,
            'overpressure_zones': np.sum(pressure_gradient > normal_gradient * 1.2),
            'underpressure_zones': np.sum(pressure_gradient < normal_gradient * 0.8),
            'gradient_range': (np.min(pressure_gradient), np.max(pressure_gradient))
        }
        
        return analysis
    
    @staticmethod
    def detect_pressure_anomalies(predictions: np.ndarray, 
                                 threshold: float = 2.0) -> Dict[str, Any]:
        """Detect pressure anomalies using statistical methods"""
        
        # Z-score based anomaly detection
        z_scores = np.abs((predictions - np.mean(predictions)) / np.std(predictions))
        anomaly_indices = np.where(z_scores > threshold)[0]
        
        # Rate of change anomalies
        pressure_changes = np.diff(predictions)
        if len(pressure_changes) > 0:
            change_z_scores = np.abs((pressure_changes - np.mean(pressure_changes)) / np.std(pressure_changes))
            rapid_change_indices = np.where(change_z_scores > threshold)[0]
        else:
            rapid_change_indices = np.array([])
        
        return {
            'anomaly_count': len(anomaly_indices),
            'anomaly_indices': anomaly_indices.tolist(),
            'rapid_change_count': len(rapid_change_indices),
            'rapid_change_indices': rapid_change_indices.tolist(),
            'max_z_score': np.max(z_scores) if len(z_scores) > 0 else 0,
            'anomaly_values': predictions[anomaly_indices].tolist() if len(anomaly_indices) > 0 else []
        }
    
    @staticmethod
    def compare_models(models: Dict[str, BaseModel], 
                      X_test: pd.DataFrame, 
                      y_test: pd.Series) -> pd.DataFrame:
        """Compare multiple formation pressure models"""
        
        comparison_data = []
        
        for name, model in models.items():
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                comparison_data.append({
                    'Model': name,
                    'R²': r2,
                    'RMSE': rmse,
                    'MAE': mae,
                    'MAPE (%)': mape,
                    'Training_Time': getattr(model, 'training_time', 'N/A')
                })
                
            except Exception as e:
                logger.error(f"Error evaluating {name}: {str(e)}")
        
        return pd.DataFrame(comparison_data).sort_values('R²', ascending=False)

# Example usage
if __name__ == "__main__":
    # Test formation pressure models
    try:
        # Test with sample data
        print("Creating sample data for testing...")
        np.random.seed(42)
        n_samples = 500
        
        X_sample = pd.DataFrame({
            'WellDepth': np.cumsum(np.random.normal(1, 0.1, n_samples)) + 1000,
            'BTBR': np.random.normal(120, 10, n_samples),
            'WBoPress': np.random.normal(200, 20, n_samples),
            'HLoad': np.random.normal(150, 15, n_samples),
            'WoBit': np.random.normal(25, 5, n_samples),
            'RoPen': np.random.normal(15, 3, n_samples),
            'DPPress': np.random.normal(180, 18, n_samples)
        })
        
        # Create synthetic formation pressure
        y_sample = pd.Series(
            0.01 * X_sample['WellDepth'] + 
            0.1 * X_sample['WBoPress'] +
            0.05 * X_sample['BTBR'] +
            np.random.normal(0, 5, n_samples),
            name='FPress'
        )
        
        print(f"Sample data created: {X_sample.shape}")
        
        # Test PCR model
        print("\n--- Testing PCR Model ---")
        pcr_model = PCRFormationPressure(n_components=4)
        pcr_metrics = pcr_model.train(X_sample, y_sample)
        print(f"PCR Val R²: {pcr_metrics['val_r2']:.4f}")
        print(f"PCR Explained Variance: {pcr_metrics['explained_variance_ratio']:.4f}")
        
        # Test XGBoost model
        print("\n--- Testing XGBoost Model ---")
        xgb_model = XGBoostFormationPressure()
        xgb_metrics = xgb_model.train(X_sample, y_sample)
        print(f"XGBoost Val R²: {xgb_metrics['val_r2']:.4f}")
        
        # Test ensemble
        print("\n--- Testing Ensemble Model ---")
        ensemble_model = EnsembleFormationPressure(['pcr', 'xgboost'])
        ensemble_metrics = ensemble_model.train(X_sample, y_sample)
        print(f"Ensemble Val R²: {ensemble_metrics['val_r2']:.4f}")
        print(f"Model weights: {ensemble_metrics['model_weights']}")
        
        # Test predictions
        test_data = X_sample.head(10)
        pcr_pred = pcr_model.predict(test_data)
        xgb_pred = xgb_model.predict(test_data)
        ensemble_pred = ensemble_model.predict(test_data)
        
        print(f"\n--- Sample Predictions ---")
        print(f"PCR predictions: {pcr_pred[:3]}")
        print(f"XGBoost predictions: {xgb_pred[:3]}")
        print(f"Ensemble predictions: {ensemble_pred[:3]}")
        print(f"Actual values: {y_sample.head(3).values}")
        
        try:
            # Test model comparison
            models = {
                'PCR': pcr_model,
                'XGBoost': xgb_model,
                'Ensemble': ensemble_model
            }
            
            X_test = X_sample.tail(100)
            y_test = y_sample.tail(100)
            
            comparison = FormationPressureAnalyzer.compare_models(models, X_test, y_test)
            print("\n--- Model Comparison ---")
            print(comparison.to_string(index=False))
            
            print("\n✅ All formation pressure model tests passed!")
            
        except Exception as e:
            print(f"❌ Error in formation pressure model tests: {str(e)}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Error in creating sample data or training models: {str(e)}")
        import traceback
        traceback.print_exc()

def optimize_formation_pressure_model(X: pd.DataFrame, y: pd.Series, 
                                    model_type: str = 'xgboost',
                                    n_trials: int = 100) -> Tuple[BaseModel, Dict[str, Any]]:
    """Optimize formation pressure model using Optuna"""
    
    import optuna
    
    def objective(trial):
        if model_type == 'xgboost':
            model = XGBoostFormationPressure()
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
        elif model_type == 'random_forest':
            model = RandomForestFormationPressure()
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
            }
        elif model_type == 'pcr':
            model = PCRFormationPressure()
            params = {
                'n_components': trial.suggest_int('n_components', 2, min(len(X.columns), 20))
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        try:
            metrics = model.train(X, y, **params)
            return metrics['val_r2']
        except Exception as e:
            logger.warning(f"Trial failed: {str(e)}")
            return 0.0
    
    # Run optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # Train best model
    if model_type == 'xgboost':
        best_model = XGBoostFormationPressure()
    elif model_type == 'random_forest':
        best_model = RandomForestFormationPressure()
    elif model_type == 'pcr':
        best_model = PCRFormationPressure()
    
    best_params = study.best_params
    best_model.train(X, y, **best_params)
    
    optimization_results = {
        'best_params': best_params,
        'best_score': study.best_value,
        'n_trials': len(study.trials),
        'optimization_history': [(trial.value, trial.params) for trial in study.trials]
    }
    
    logger.info(f"Optimization completed. Best R²: {study.best_value:.4f}")
    logger.info(f"Best params: {best_params}")
    
    return best_model, optimization_results