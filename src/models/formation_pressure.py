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
            'model_weights': self.model_weights,
            'individual_performances': model_performances
        }
        
        self.is_fitted = True
        
        logger.info(f"Ensemble training completed. Val R²: {self.metrics['val_r2']:.4f}")
        logger.info(f"Model weights: {self.model_weights}")
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions"""
        
        if not self.is_fitted:
            raise ValueError("Ensemble must be trained before making predictions")
        
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            if model.is_fitted and self.model_weights[name] > 0:
                try:
                    pred = model.predict(X)
                    predictions.append(pred)
                    weights.append(self.model_weights[name])
                except Exception as e:
                    logger.warning(f"Model {name} prediction failed: {str(e)}")
        
        if not predictions:
            raise ValueError("No models available for prediction")
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        return ensemble_pred
    
    def get_model_contributions(self, X: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Get individual model contributions to ensemble"""
        
        contributions = {}
        
        for name, model in self.models.items():
            if model.is_fitted:
                try:
                    pred = model.predict(X)
                    contributions[name] = {
                        'predictions': pred,
                        'weight': self.model_weights[name],
                        'metrics': model.metrics
                    }
                except Exception as e:
                    logger.warning(f"Could not get contributions for {name}: {str(e)}")
        
        return contributions
    
    def get_ensemble_feature_importance(self) -> Dict[str, float]:
        """Get weighted feature importance from ensemble"""
        
        ensemble_importance = {}
        
        for name, model in self.models.items():
            if model.is_fitted and hasattr(model, 'get_feature_importance'):
                try:
                    model_importance = model.get_feature_importance()
                    weight = self.model_weights[name]
                    
                    for feature, importance in model_importance.items():
                        if feature not in ensemble_importance:
                            ensemble_importance[feature] = 0
                        ensemble_importance[feature] += importance * weight
                        
                except Exception as e:
                    logger.warning(f"Could not get feature importance for {name}: {str(e)}")
        
        # Sort by importance
        ensemble_importance = dict(
            sorted(ensemble_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        return ensemble_importance

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

class FormationPressureOptimizer:
    """Optimization tools for formation pressure models"""
    
    @staticmethod
    def optimize_hyperparameters(X: pd.DataFrame, y: pd.Series, 
                               model_type: str = 'xgboost',
                               n_trials: int = 100) -> Tuple[BaseModel, Dict[str, Any]]:
        """Optimize hyperparameters using Optuna"""
        
        try:
            import optuna
        except ImportError:
            logger.error("Optuna not installed. Install with: pip install optuna")
            raise
        
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
            
            # Train model with cross-validation
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

def create_formation_pressure_pipeline(data_type: str = 'advanced') -> BaseModel:
    """Create formation pressure prediction pipeline"""
    
    if data_type == 'simple':
        # Simple PCR model for quick predictions
        return PCRFormationPressure(n_components=4)
    
    elif data_type == 'advanced':
        # Advanced ensemble for best accuracy
        return EnsembleFormationPressure(['pcr', 'xgboost', 'random_forest'])
    
    elif data_type == 'production':
        # Optimized single model for production
        return XGBoostFormationPressure()
    
    elif data_type == 'research':
        # All models for comparison
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


# ==============================
# Ensemble training metrics
# ==============================

self.metrics = {
    'train_mae': mean_absolute_error(y_train, y_train_pred),
    'val_mae': mean_absolute_error(y_val, y_val_pred),
    'model_weights': self.model_weights,
    'individual_performances': model_performances
}

self.is_fitted = True

logger.info(f"Ensemble training completed. Val R²: {self.metrics.get('val_r2', float('nan')):.4f}")
logger.info(f"Model weights: {self.model_weights}")

return self.metrics

    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions"""
        
        if not self.is_fitted:
            raise ValueError("Ensemble must be trained before making predictions")
        
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            if model.is_fitted and self.model_weights[name] > 0:
                try:
                    pred = model.predict(X)
                    predictions.append(pred)
                    weights.append(self.model_weights[name])
                except Exception as e:
                    logger.warning(f"Model {name} prediction failed: {str(e)}")
        
        if not predictions:
            raise ValueError("No models available for prediction")
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        return ensemble_pred
    
    def get_model_contributions(self, X: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Get individual model contributions to ensemble"""
        
        contributions = {}
        
        for name, model in self.models.items():
            if model.is_fitted:
                try:
                    pred = model.predict(X)
                    contributions[name] = {
                        'predictions': pred,
                        'weight': self.model_weights[name],
                        'metrics': model.metrics
                    }
                except Exception as e:
                    logger.warning(f"Could not get contributions for {name}: {str(e)}")
        
        return contributions

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

def create_formation_pressure_pipeline(data_type: str = 'advanced') -> BaseModel:
    """Create formation pressure prediction pipeline"""
    
    if data_type == 'simple':
        # Simple PCR model for quick predictions
        return PCRFormationPressure(n_components=4)
    
    elif data_type == 'advanced':
        # Advanced ensemble for best accuracy
        return EnsembleFormationPressure(['pcr', 'xgboost', 'random_forest'])
    
    elif data_type == 'production':
        # Optimized single model for production
        return XGBoostFormationPressure()
    
    else:
        raise ValueError(f"Unknown pipeline type: {data_type}")

if __name__ == "__main__":
    # Test formation pressure models
    from data.data_loader import DataLoader
    from data.data_preprocessor import DataPreprocessor
    
    try:
        # Load and prepare data
        loader = DataLoader()
        formation_data = loader.load_formation_data()
        
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.prepare_formation_pressure_data(formation_data)
        
        # Split features and target
        X, y = loader.split_features_target('formation')
        
        print(f"Testing formation pressure models...")
        print(f"Data shape: {X.shape}")
        
        # Test PCR model
        print("\n--- Testing PCR Model ---")
        pcr_model = PCRFormationPressure(n_components=4)
        pcr_metrics = pcr_model.train(X, y)
        print(f"PCR Val R²: {pcr_metrics['val_r2']:.4f}")
        print(f"PCR Explained Variance: {pcr_metrics['explained_variance_ratio']:.4f}")
        
        # Test XGBoost model
        print("\n--- Testing XGBoost Model ---")
        xgb_model = XGBoostFormationPressure()
        xgb_metrics = xgb_model.train(X, y)
        print(f"XGBoost Val R²: {xgb_metrics['val_r2']:.4f}")
        
        # Test ensemble
        print("\n--- Testing Ensemble Model ---")
        ensemble_model = EnsembleFormationPressure(['pcr', 'xgboost'])
        ensemble_metrics = ensemble_model.train(X, y)
        print(f"Ensemble Val R²: {ensemble_metrics['val_r2']:.4f}")
        print(f"Model weights: {ensemble_metrics['model_weights']}")
        
        # Compare models
        models = {
            'PCR': pcr_model,
            'XGBoost': xgb_model,
            'Ensemble': ensemble_model
        }
        
        X_test = X.tail(100)
        y_test = y.tail(100)
        
        comparison = FormationPressureAnalyzer.compare_models(models, X_test, y_test)
        print("\n--- Model Comparison ---")
        print(comparison)
        
    except Exception as e:
        print(f"Error testing formation pressure models: {str(e)}")
        
        # Create sample data for testing
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
        
        print("Testing with sample data...")
        
        # Test PCR with sample data
        pcr_model = PCRFormationPressure(n_components=3)
        metrics = pcr_model.train(X_sample, y_sample)
        
        print(f"Sample PCR model trained successfully!")
        print(f"R² Score: {metrics['val_r2']:.4f}")
        print(f"RMSE: {metrics['val_rmse']:.4f}")
        
        # Test predictions
        sample_predictions = pcr_model.predict(X_sample.head(10))
        print(f"Sample predictions: {sample_predictions[:3]}")

        # Évaluation du modèle
        metrics = {
                'train_mae': mean_absolute_error(y_train, y_train_pred),
                'val_mae': mean_absolute_error(y_val, y_val_pred),
                'explained_variance_ratio': self.pca.explained_variance_ratio_.sum()
            }

        print("Metrics:", metrics)

        
        # Store training history
        self.training_history.update({
            'n_components': self.n_components,
            'explained_variance': self.pca.explained_variance_ratio_.tolist(),
            'feature_loadings': self.pca.components_.tolist()
        })
        
        logger.info(f"PCR training completed. Val R²: {val_r2:.4f}, Explained variance: {self.metrics['explained_variance_ratio']:.4f}")
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using PCR model"""
        
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess
        X_processed = self._preprocess_data(X, apply_smoothing=False)  # Don't smooth new data
        X_processed = X_processed[self.feature_columns]
        
        # Apply transformations
        X_scaled = self.scaler.transform(X_processed.fillna(X_processed.mean()))
        X_pca = self.pca.transform(X_scaled)
        
        # Predict
        return self.regressor.predict(X_pca)
    
    def get_component_loadings(self) -> pd.DataFrame:
        """Get PCA component loadings"""
        
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        
        loadings_df = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
            index=self.feature_columns
        )
        
        return loadings_df
    
    def analyze_components(self, top_features: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        """Analyze which features contribute most to each component"""
        
        loadings = self.get_component_loadings()
        component_analysis = {}
        
        for col in loadings.columns:
            # Get top contributing features for this component
            feature_contributions = [(feat, abs(val)) for feat, val in zip(loadings.index, loadings[col])]
            feature_contributions.sort(key=lambda x: x[1], reverse=True)
            
            component_analysis[col] = feature_contributions[:top_features]
        
        return component_analysis

class XGBoostFormationPressure(BaseModel):
    """XGBoost model for Formation Pressure Prediction"""
    
    def __init__(self):
        super().__init__("XGBoost_Formation_Pressure", "regression")
    
    def _build_model(self, **kwargs) -> xgb.XGBRegressor:
        """Build XGBoost model"""
        
        params = self._get_default_params()
        params.update(kwargs)
        
        return xgb.XGBRegressor(**params)
    
    def _get_default_params(self) -> Dict[str, Any]:
        return config.model.xgb_params.copy()
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
             validation_split: float = 0.2,
             **model_params) -> Dict[str, float]:
        """Train XGBoost model with early stopping"""
        
        logger.info("Starting XGBoost Formation Pressure training...")
        
        # Prepare data
        X_train, X_val, y_train, y_val = self.prepare_data(X, y, validation_split)
        
        # Build model
        self.model = self._build_model(**model_params)
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=False
        )
        
        self.is_fitted = True
        
        # Calculate metrics
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        
        self.metrics = {
            'train_r2': r2_score(y_train, y_train_pred),
            'val_r2': r2_score(y_val, y_val_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'val_mae': mean_absolute_error(y_val, y_val_pred)
        }
        
        logger.info(f"XGBoost training completed. Val R²: {self.metrics['val_r2']:.4f}")
        
        return self.metrics

class RandomForestFormationPressure(BaseModel):
    """Random Forest model for Formation Pressure Prediction"""
    
    def __init__(self):
        super().__init__("RandomForest_Formation_Pressure", "regression")
    
    def _build_model(self, **kwargs) -> RandomForestRegressor:
        """Build Random Forest model"""
        
        params = self._get_default_params()
        params.update(kwargs)
        
        return RandomForestRegressor(**params)
    
    def _get_default_params(self) -> Dict[str, Any]:
        return config.model.rf_params.copy()

class PLSFormationPressure(BaseModel):
    """Partial Least Squares Regression for Formation Pressure"""
    
    def __init__(self, n_components: int = None):
        super().__init__("PLS_Formation_Pressure", "regression")
        self.n_components = n_components or config.model.formation_n_components
        self.scaler = None
    
    def _build_model(self, **kwargs) -> PLSRegression:
        """Build PLS model"""
        
        n_comp = kwargs.get('n_components', self.n_components)
        self.scaler = StandardScaler()
        
        return PLSRegression(n_components=n_comp)
    
    def _get_default_params(self) -> Dict[str, Any]:
        return {'n_components': self.n_components}
    
    def train(self, X: pd.DataFrame, y: pd.Series,
             validation_split: float = 0.2,
             **model_params) -> Dict[str, float]:
        """Train PLS model"""
        
        logger.info("Starting PLS Formation Pressure training...")
        
        # Prepare data
        X_train, X_val, y_train, y_val = self.prepare_data(X, y, validation_split)
        
        # Build model
        self.model = self._build_model(**model_params)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled).flatten()
        y_val_pred = self.model.predict(X_val_scaled).flatten()
        
        # Calculate metrics
        self.metrics = {
            'train_r2': r2_score(y_train, y_train_pred),
            'val_r2': r2_score(y_val, y_val_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'val_mae': mean_absolute_error(y_val, y_val_pred)
        }
        
        logger.info(f"PLS training completed. Val R²: {self.metrics['val_r2']:.4f}")
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with PLS model"""
        
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_pred = X[self.feature_columns].fillna(X[self.feature_columns].mean())
        X_scaled = self.scaler.transform(X_pred)
        
        return self.model.predict(X_scaled).flatten()

class EnsembleFormationPressure(BaseModel):
    """Ensemble of multiple models for Formation Pressure Prediction"""
    
    def __init__(self, models: List[str] = None):
        super().__init__("Ensemble_Formation_Pressure", "regression")
        
        # Define available models
        self.available_models = {
            'pcr': PCRFormationPressure,
            'xgboost': XGBoostFormationPressure,
            'random_forest': RandomForestFormationPressure,
            'pls': PLSFormationPressure
        }
        
        # Initialize selected models
        model_names = models or ['pcr', 'xgboost', 'random_forest']
        self.models = {name: cls() for name, cls in self.available_models.items() if name in model_names}
        self.model_weights = {}
        
    def _build_model(self, **kwargs) -> Dict[str, BaseModel]:
        """Build ensemble models"""
        return self.models
    
    def _get_default_params(self) -> Dict[str, Any]:
        return {}
    
    def train(self, X: pd.DataFrame, y: pd.Series,
             validation_split: float = 0.2,
             **model_params) -> Dict[str, float]:
        """Train all models in ensemble"""
        
        logger.info(f"Training ensemble with {len(self.models)} models...")
        
        model_performances = {}
        
        # Train each model
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Extract model-specific params
                model_specific_params = model_params.get(name, {})
                
                # Train model
                metrics = model.train(X, y, validation_split, **model_specific_params)
                model_performances[name] = metrics['val_r2']
                
                logger.info(f"{name} completed. Val R²: {metrics['val_r2']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                model_performances[name] = 0.0
        
        # Calculate weights based on performance
        total_performance = sum(model_performances.values())
        if total_performance > 0:
            self.model_weights = {
                name: perf / total_performance 
                for name, perf in model_performances.items()
            }
        else:
            # Equal weights if all models failed
            self.model_weights = {name: 1.0/len(self.models) for name in self.models.keys()}