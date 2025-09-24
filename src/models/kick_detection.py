"""
Kick Detection Models for Drilling Operations
Advanced ML models for detecting kicks and drilling anomalies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models.base_model import BaseModel
from utils.config import config
import logging

logger = logging.getLogger(__name__)

class PCAKickDetection(BaseModel):
    """PCA-based anomaly detection for kick identification"""
    
    def __init__(self, variance_threshold: float = None):
        super().__init__("PCA_Kick_Detection", "classification")
        self.variance_threshold = variance_threshold or config.model.kick_pca_variance
        self.detection_threshold = config.model.kick_detection_threshold
        self.pca = None
        self.scaler = None
        self.spe_threshold = None
        
    def _build_model(self, **kwargs) -> Dict[str, Any]:
        """Build PCA model components"""
        
        variance_thresh = kwargs.get('variance_threshold', self.variance_threshold)
        
        self.pca = PCA(n_components=variance_thresh)
        self.scaler = StandardScaler()
        
        return {
            'pca': self.pca,
            'scaler': self.scaler,
            'variance_threshold': variance_thresh
        }
    
    def _get_default_params(self) -> Dict[str, Any]:
        return {
            'variance_threshold': self.variance_threshold,
            'detection_threshold': self.detection_threshold
        }
    
    def _calculate_spe(self, X: np.ndarray, X_reconstructed: np.ndarray) -> np.ndarray:
        """Calculate Squared Prediction Error (SPE)"""
        return np.sum((X - X_reconstructed) ** 2, axis=1)
    
    def train(self, X: pd.DataFrame, y: pd.Series = None,
             validation_split: float = 0.2,
             **model_params) -> Dict[str, float]:
        """Train PCA anomaly detection model"""
        
        logger.info("Starting PCA Kick Detection training...")
        
        # Build model
        self._build_model(**model_params)
        
        # Use only normal operations for training (assuming majority are normal)
        if y is not None:
            # If labels available, train only on normal data
            normal_mask = y == 0  # Assuming 0 = normal, 1 = kick
            X_normal = X[normal_mask]
            logger.info(f"Training on {len(X_normal)} normal samples out of {len(X)} total")
        else:
            # If no labels, assume all training data is normal
            X_normal = X
            logger.info("No labels provided, assuming all training data is normal")
        
        # Store feature columns
        self.feature_columns = X_normal.columns.tolist()
        
        # Prepare data splits
        split_idx = int(len(X_normal) * (1 - validation_split))
        X_train = X_normal.iloc[:split_idx]
        X_val = X_normal.iloc[split_idx:]
        
        # Scale data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Fit PCA
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_val_pca = self.pca.transform(X_val_scaled)
        
        # Reconstruct data
        X_train_reconstructed = self.pca.inverse_transform(X_train_pca)
        X_val_reconstructed = self.pca.inverse_transform(X_val_pca)
        
        # Calculate SPE scores
        spe_train = self._calculate_spe(X_train_scaled, X_train_reconstructed)
        spe_val = self._calculate_spe(X_val_scaled, X_val_reconstructed)
        
        # Set detection threshold based on training SPE
        detection_percentile = model_params.get('detection_threshold', self.detection_threshold)
        self.spe_threshold = np.percentile(spe_train, detection_percentile)
        
        # Calculate metrics
        # For training, we expect low anomaly rate
        train_anomalies = np.sum(spe_train > self.spe_threshold)
        val_anomalies = np.sum(spe_val > self.spe_threshold)
        
        self.metrics = {
            'train_anomaly_rate': train_anomalies / len(spe_train),
            'val_anomaly_rate': val_anomalies / len(spe_val),
            'spe_threshold': self.spe_threshold,
            'explained_variance_ratio': self.pca.explained_variance_ratio_.sum(),
            'n_components': self.pca.n_components_,
            'train_spe_mean': np.mean(spe_train),
            'train_spe_std': np.std(spe_train),
            'val_spe_mean': np.mean(spe_val),
            'val_spe_std': np.std(spe_val)
        }
        
        self.is_fitted = True
        
        # Store training history
        self.training_history.update({
            'spe_threshold': self.spe_threshold,
            'explained_variance': self.pca.explained_variance_ratio_.tolist(),
            'principal_components': self.pca.components_.tolist()
        })
        
        logger.info(f"PCA Kick Detection training completed.")
        logger.info(f"SPE threshold: {self.spe_threshold:.4f}")
        logger.info(f"Explained variance: {self.metrics['explained_variance_ratio']:.4f}")
        logger.info(f"Training anomaly rate: {self.metrics['train_anomaly_rate']:.4f}")
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict kicks (1) or normal operations (0)"""
        
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_pred = X[self.feature_columns].fillna(X[self.feature_columns].mean())
        
        # Scale data
        X_scaled = self.scaler.transform(X_pred)
        
        # Apply PCA and reconstruct
        X_pca = self.pca.transform(X_scaled)
        X_reconstructed = self.pca.inverse_transform(X_pca)
        
        # Calculate SPE
        spe_scores = self._calculate_spe(X_scaled, X_reconstructed)
        
        # Classify as kick if SPE > threshold
        predictions = (spe_scores > self.spe_threshold).astype(int)
        
        return predictions
    
    def get_anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Get anomaly scores (SPE values)"""
        
        if not self.is_fitted:
            raise ValueError("Model must be trained before getting scores")
        
        X_pred = X[self.feature_columns].fillna(X[self.feature_columns].mean())
        
        # Scale data
        X_scaled = self.scaler.transform(X_pred)
        
        # Apply PCA and reconstruct
        X_pca = self.pca.transform(X_scaled)
        X_reconstructed = self.pca.inverse_transform(X_pca)
        
        # Calculate and return SPE scores
        return self._calculate_spe(X_scaled, X_reconstructed)
    
    def analyze_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Analyze which features contribute most to anomalies"""
        
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        
        X_pred = X[self.feature_columns].fillna(X[self.feature_columns].mean())
        X_scaled = self.scaler.transform(X_pred)
        
        # Get PCA reconstruction
        X_pca = self.pca.transform(X_scaled)
        X_reconstructed = self.pca.inverse_transform(X_pca)
        
        # Calculate feature-wise contributions to SPE
        feature_contributions = (X_scaled - X_reconstructed) ** 2
        
        # Create DataFrame
        contributions_df = pd.DataFrame(
            feature_contributions,
            columns=self.feature_columns,
            index=X.index
        )
        
        return contributions_df

class IsolationForestKickDetection(BaseModel):
    """Isolation Forest for kick detection"""
    
    def __init__(self, contamination: float = 0.1):
        super().__init__("IsolationForest_Kick_Detection", "classification")
        self.contamination = contamination
    
    def _build_model(self, **kwargs) -> IsolationForest:
        """Build Isolation Forest model"""
        
        contamination = kwargs.get('contamination', self.contamination)
        
        return IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
    
    def _get_default_params(self) -> Dict[str, Any]:
        return {'contamination': self.contamination}
    
    def train(self, X: pd.DataFrame, y: pd.Series = None,
             validation_split: float = 0.2,
             **model_params) -> Dict[str, float]:
        """Train Isolation Forest model"""
        
        logger.info("Starting Isolation Forest Kick Detection training...")
        
        # Prepare data
        X_train, X_val, _, _ = self.prepare_data(X, y if y is not None else pd.Series([0]*len(X)), validation_split)
        
        # Build and train model
        self.model = self._build_model(**model_params)
        self.model.fit(X_train)
        
        self.is_fitted = True
        
        # Get anomaly scores
        train_scores = self.model.decision_function(X_train)
        val_scores = self.model.decision_function(X_val)
        
        # Make predictions
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        # Convert to binary (1 = anomaly/kick, 0 = normal)
        train_pred_binary = (train_pred == -1).astype(int)
        val_pred_binary = (val_pred == -1).astype(int)
        
        self.metrics = {
            'train_anomaly_rate': np.mean(train_pred_binary),
            'val_anomaly_rate': np.mean(val_pred_binary),
            'train_score_mean': np.mean(train_scores),
            'val_score_mean': np.mean(val_scores),
            'contamination': model_params.get('contamination', self.contamination)
        }
        
        logger.info(f"Isolation Forest training completed.")
        logger.info(f"Train anomaly rate: {self.metrics['train_anomaly_rate']:.4f}")
        logger.info(f"Val anomaly rate: {self.metrics['val_anomaly_rate']:.4f}")
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict kicks using Isolation Forest"""
        
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_pred = X[self.feature_columns].fillna(X[self.feature_columns].mean())
        predictions = self.model.predict(X_pred)
        
        # Convert to binary (1 = kick, 0 = normal)
        return (predictions == -1).astype(int)
    
    def get_anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Get anomaly scores from Isolation Forest"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)   
        if not self.is_fitted:
            raise ValueError("Model must be trained before getting scores")
        
        X_pred = X[self.feature_columns].fillna(X[self.feature_columns].mean())
        return -self.model.decision_function(X_scaled)

class EnsembleKickDetection(BaseModel):
    """Ensemble of multiple anomaly detection models for kick detection"""
    
    def __init__(self, models: List[str] = None):
        super().__init__("Ensemble_Kick_Detection", "classification")
        
        # Available models
        self.available_models = {
            'pca': PCAKickDetection,
            'isolation_forest': IsolationForestKickDetection,
            'one_class_svm': OneClassSVMKickDetection
        }
        
        # Initialize selected models
        model_names = models or ['pca', 'isolation_forest']
        self.models = {name: cls() for name, cls in self.available_models.items() if name in model_names}
        self.model_weights = {}
        
    def _build_model(self, **kwargs) -> Dict[str, BaseModel]:
        """Build ensemble models"""
        return self.models
    
    def _get_default_params(self) -> Dict[str, Any]:
        return {}
    
    def train(self, X: pd.DataFrame, y: pd.Series = None,
             validation_split: float = 0.2,
             **model_params) -> Dict[str, float]:
        """Train all models in ensemble"""
        
        logger.info(f"Training kick detection ensemble with {len(self.models)} models...")
        
        model_performances = {}
        
        # Train each model
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Extract model-specific params
                model_specific_params = model_params.get(name, {})
                
                # Train model
                metrics = model.train(X, y, validation_split, **model_specific_params)
                
                # Use inverse of anomaly rate as performance metric (lower is better for normal operations)
                model_performances[name] = 1.0 - metrics.get('val_anomaly_rate', 0.1)
                
                logger.info(f"{name} completed. Anomaly rate: {metrics.get('val_anomaly_rate', 0):.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                model_performances[name] = 0.1
        
        # Calculate weights (equal weights for now)
        self.model_weights = {name: 1.0/len(self.models) for name in self.models.keys()}
        
        # Calculate ensemble metrics
        if y is not None:
            X_train, X_val, y_train, y_val = self.prepare_data(X, y, validation_split)
            
            y_val_pred = self.predict(X_val)
            
            # Calculate classification metrics if ground truth available
            self.metrics = {
                'val_accuracy': accuracy_score(y_val, y_val_pred),
                'val_precision': precision_score(y_val, y_val_pred, average='binary'),
                'val_recall': recall_score(y_val, y_val_pred, average='binary'),
                'val_f1': f1_score(y_val, y_val_pred, average='binary'),
                'model_weights': self.model_weights,
                'individual_performances': model_performances
            }
        else:
            # No ground truth, use anomaly rates
            X_train, X_val, _, _ = self.prepare_data(X, pd.Series([0]*len(X)), validation_split)
            val_pred = self.predict(X_val)
            
            self.metrics = {
                'val_anomaly_rate': np.mean(val_pred),
                'model_weights': self.model_weights,
                'individual_performances': model_performances
            }
        
        self.is_fitted = True
        
        logger.info(f"Ensemble training completed.")
        logger.info(f"Model weights: {self.model_weights}")
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions"""
        
        if not self.is_fitted:
            raise ValueError("Ensemble must be trained before making predictions")
        
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            if model.is_fitted:
                try:
                    pred = model.predict(X)
                    predictions.append(pred)
                    weights.append(self.model_weights[name])
                except Exception as e:
                    logger.warning(f"Model {name} prediction failed: {str(e)}")
        
        if not predictions:
            raise ValueError("No models available for prediction")
        
        # Weighted voting
        ensemble_scores = np.average(predictions, axis=0, weights=weights)
        
        # Convert to binary predictions (threshold at 0.5)
        return (ensemble_scores > 0.5).astype(int)
    
    def get_ensemble_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Get ensemble anomaly scores"""
        
        scores = []
        weights = []
        
        for name, model in self.models.items():
            if model.is_fitted and hasattr(model, 'get_anomaly_scores'):
                try:
                    score = model.get_anomaly_scores(X)
                    # Normalize scores to 0-1 range
                    score_norm = (score - score.min()) / (score.max() - score.min() + 1e-8)
                    scores.append(score_norm)
                    weights.append(self.model_weights[name])
                except Exception as e:
                    logger.warning(f"Could not get scores from {name}: {str(e)}")
        
        if scores:
            return np.average(scores, axis=0, weights=weights)
        else:
            return np.zeros(len(X))

class KickDetectionAnalyzer:
    """Analysis tools for kick detection results"""
    
    @staticmethod
    def analyze_detection_performance(y_true: np.ndarray, y_pred: np.ndarray, 
                                    y_scores: np.ndarray = None) -> Dict[str, Any]:
        """Comprehensive analysis of kick detection performance"""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Safety-critical metrics
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        analysis = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'false_negative_rate': false_negative_rate,
            'false_positive_rate': false_positive_rate,
            'confusion_matrix': {
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp)
            },
            'safety_assessment': {
                'missed_kicks': int(fn),
                'false_alarms': int(fp),
                'safety_score': recall  # High recall is critical for safety
            }
        }
        
        # Add AUC if scores provided
        if y_scores is not None:
            try:
                auc = roc_auc_score(y_true, y_scores)
                analysis['auc'] = auc
            except:
                pass
        
        return analysis
    
    @staticmethod
    def detect_kick_patterns(anomaly_scores: np.ndarray, 
                           timestamps: pd.DatetimeIndex = None,
                           threshold: float = None) -> Dict[str, Any]:
        """Detect patterns in kick occurrences"""
        
        if threshold is None:
            threshold = np.percentile(anomaly_scores, 95)
        
        # Find kick events
        kick_events = anomaly_scores > threshold
        kick_indices = np.where(kick_events)[0]
        
        # Analyze patterns
        patterns = {
            'total_events': len(kick_indices),
            'event_rate': len(kick_indices) / len(anomaly_scores),
            'max_score': np.max(anomaly_scores),
            'threshold_used': threshold
        }
        
        if len(kick_indices) > 1:
            # Time between events
            intervals = np.diff(kick_indices)
            patterns.update({
                'mean_interval': np.mean(intervals),
                'std_interval': np.std(intervals),
                'min_interval': np.min(intervals),
                'max_interval': np.max(intervals)
            })
        
        # Duration analysis (consecutive anomalies)
        if len(kick_indices) > 0:
            durations = []
            current_duration = 1
            
            for i in range(1, len(kick_indices)):
                if kick_indices[i] == kick_indices[i-1] + 1:
                    current_duration += 1
                else:
                    durations.append(current_duration)
                    current_duration = 1
            durations.append(current_duration)
            
            patterns.update({
                'mean_duration': np.mean(durations),
                'max_duration': np.max(durations),
                'total_durations': len(durations)
            })
        
        return patterns
    
    @staticmethod
    def compare_detection_models(models: Dict[str, BaseModel], 
                               X_test: pd.DataFrame, 
                               y_test: pd.Series = None) -> pd.DataFrame:
        """Compare multiple kick detection models"""
        
        comparison_data = []
        
        for name, model in models.items():
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                anomaly_rate = np.mean(y_pred)
                
                model_data = {
                    'Model': name,
                    'Anomaly_Rate': anomaly_rate,
                    'Predictions_Available': True
                }
                
                # Add performance metrics if ground truth available
                if y_test is not None:
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
                    
                    model_data.update({
                        'Accuracy': accuracy,
                        'Precision': precision,
                        'Recall': recall,
                        'F1_Score': f1,
                        'Safety_Score': recall  # Critical for kick detection
                    })
                
                # Get anomaly scores if available
                if hasattr(model, 'get_anomaly_scores'):
                    try:
                        scores = model.get_anomaly_scores(X_test)
                        model_data['Mean_Anomaly_Score'] = np.mean(scores)
                        model_data['Max_Anomaly_Score'] = np.max(scores)
                    except:
                        pass
                
                comparison_data.append(model_data)
                
            except Exception as e:
                logger.error(f"Error evaluating {name}: {str(e)}")
                comparison_data.append({
                    'Model': name,
                    'Predictions_Available': False,
                    'Error': str(e)
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by safety score (recall) if available
        if 'Safety_Score' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('Safety_Score', ascending=False)
        
        return comparison_df

def optimize_kick_detection_threshold(model: BaseModel, X_val: pd.DataFrame, 
                                    y_val: pd.Series, metric: str = 'f1') -> Tuple[float, Dict[str, float]]:
    """Optimize detection threshold for kick detection model"""
    
    if not hasattr(model, 'get_anomaly_scores'):
        raise ValueError("Model must support anomaly scores for threshold optimization")
    
    # Get anomaly scores
    scores = model.get_anomaly_scores(X_val)
    
    # Test different thresholds
    thresholds = np.percentile(scores, np.linspace(50, 99.9, 100))
    
    best_threshold = None
    best_score = -np.inf
    results = []
    
    for threshold in thresholds:
        predictions = (scores > threshold).astype(int)
        
        # Calculate metrics
        try:
            if metric == 'f1':
                score = f1_score(y_val, predictions, average='binary', zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_val, predictions, average='binary', zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_val, predictions, average='binary', zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            results.append({
                'threshold': threshold,
                'score': score,
                'precision': precision_score(y_val, predictions, average='binary', zero_division=0),
                'recall': recall_score(y_val, predictions, average='binary', zero_division=0),
                'f1': f1_score(y_val, predictions, average='binary', zero_division=0)
            })
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                
        except Exception as e:
            logger.warning(f"Error calculating metrics for threshold {threshold}: {str(e)}")
    
    # Get best result
    best_result = next(r for r in results if r['threshold'] == best_threshold)
    
    logger.info(f"Best threshold: {best_threshold:.4f} with {metric}: {best_score:.4f}")
    
    return best_threshold, best_result

def create_kick_detection_pipeline(model_type: str = 'ensemble') -> BaseModel:
    """Create kick detection pipeline"""
    
    if model_type == 'pca':
        return PCAKickDetection()
    elif model_type == 'isolation_forest':
        return IsolationForestKickDetection()
    elif model_type == 'one_class_svm':
        return OneClassSVMKickDetection()
    elif model_type == 'ensemble':
        return EnsembleKickDetection(['pca', 'isolation_forest'])
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    # Test kick detection models
    from data.data_loader import DataLoader
    from data.data_preprocessor import DataPreprocessor
    
    try:
        # Load and prepare data
        loader = DataLoader()
        kick_data = loader.load_kick_data()
        
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.prepare_kick_detection_data(kick_data)
        
        # Select features (exclude target)
        feature_cols = [col for col in processed_data.columns if col != 'ActiveGL']
        X = processed_data[feature_cols]
        
        # Create synthetic labels for testing (in real scenario, you'd have actual kick labels)
        y = (processed_data['ActiveGL'] > processed_data['ActiveGL'].quantile(0.95)).astype(int)
        
        print(f"Testing kick detection models...")
        print(f"Data shape: {X.shape}")
        print(f"Kick rate: {np.mean(y):.4f}")
        
        # Test PCA model
        print("\n--- Testing PCA Model ---")
        pca_model = PCAKickDetection()
        pca_metrics = pca_model.train(X, y)
        print(f"PCA Anomaly Rate: {pca_metrics['val_anomaly_rate']:.4f}")
        print(f"PCA Explained Variance: {pca_metrics['explained_variance_ratio']:.4f}")
        
        # Test Isolation Forest
        print("\n--- Testing Isolation Forest ---")
        iso_model = IsolationForestKickDetection()
        iso_metrics = iso_model.train(X, y)
        print(f"Isolation Forest Anomaly Rate: {iso_metrics['val_anomaly_rate']:.4f}")
        
        # Test Ensemble
        print("\n--- Testing Ensemble ---")
        ensemble_model = EnsembleKickDetection(['pca', 'isolation_forest'])
        ensemble_metrics = ensemble_model.train(X, y)
        print(f"Ensemble Metrics: {ensemble_metrics}")
        
        # Compare models
        models = {
            'PCA': pca_model,
            'IsolationForest': iso_model,
            'Ensemble': ensemble_model
        }
        
        X_test = X.tail(200)
        y_test = y.tail(200)
        
        comparison = KickDetectionAnalyzer.compare_detection_models(models, X_test, y_test)
        print("\n--- Model Comparison ---")
        print(comparison)
        
    except Exception as e:
        print(f"Error testing kick detection models: {str(e)}")
        
        # Create sample data for testing
        np.random.seed(42)
        n_samples = 1000
        
        # Normal drilling operations
        X_normal = pd.DataFrame({
            'FRate': np.random.normal(300, 30, n_samples),
            'ActiveGL': np.random.normal(100, 10, n_samples),
            'SMSpeed': np.random.normal(50, 5, n_samples),
            'FIn': np.random.normal(280, 20, n_samples),
            'FOut': np.random.normal(285, 20, n_samples),
            'WBoPress': np.random.normal(200, 15, n_samples),
        })
        
        # Add some anomalies (kicks)
        n_anomalies = 50
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        
        X_sample = X_normal.copy()
        X_sample.loc[anomaly_indices, 'ActiveGL'] += np.random.normal(50, 10, n_anomalies)  # Pit gain
        X_sample.loc[anomaly_indices, 'FOut'] += np.random.normal(20, 5, n_anomalies)      # Flow increase
        
        # Create labels
        y_sample = np.zeros(n_samples)
        y_sample[anomaly_indices] = 1
        
        print("Testing with sample data...")
        
        # Test PCA with sample data
        pca_model = PCAKickDetection()
        metrics = pca_model.train(X_sample, pd.Series(y_sample))
        
        print(f"Sample PCA model trained successfully!")
        print(f"Anomaly detection rate: {metrics['val_anomaly_rate']:.4f}")
        print(f"SPE threshold: {metrics['spe_threshold']:.4f}")
        
        # Test predictions
        sample_predictions = pca_model.predict(X_sample.head(10))
        print(f"Sample predictions: {sample_predictions}")
        # Negative for intuitive scores

class OneClassSVMKickDetection(BaseModel):
    """One-Class SVM for kick detection"""
    
    def __init__(self, nu: float = 0.1):
        super().__init__("OneClassSVM_Kick_Detection", "classification")
        self.nu = nu
        self.scaler = None
    
    def _build_model(self, **kwargs) -> OneClassSVM:
        """Build One-Class SVM model"""
        
        nu = kwargs.get('nu', self.nu)
        self.scaler = StandardScaler()
        
        return OneClassSVM(
            nu=nu,
            kernel='rbf',
            gamma='scale'
        )
    
    def _get_default_params(self) -> Dict[str, Any]:
        return {'nu': self.nu}
    
    def train(self, X: pd.DataFrame, y: pd.Series = None,
             validation_split: float = 0.2,
             **model_params) -> Dict[str, float]:
        """Train One-Class SVM model"""
        
        logger.info("Starting One-Class SVM Kick Detection training...")
        
        # Prepare data
        X_train, X_val, _, _ = self.prepare_data(X, y if y is not None else pd.Series([0]*len(X)), validation_split)
        
        # Scale data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Build and train model
        self.model = self._build_model(**model_params)
        self.model.fit(X_train_scaled)
        
        self.is_fitted = True
        
        # Make predictions
        train_pred = self.model.predict(X_train_scaled)
        val_pred = self.model.predict(X_val_scaled)
        
        # Convert to binary
        train_pred_binary = (train_pred == -1).astype(int)
        val_pred_binary = (val_pred == -1).astype(int)
        
        # Get decision scores
        train_scores = self.model.decision_function(X_train_scaled)
        val_scores = self.model.decision_function(X_val_scaled)
        
        self.metrics = {
            'train_anomaly_rate': np.mean(train_pred_binary),
            'val_anomaly_rate': np.mean(val_pred_binary),
            'train_score_mean': np.mean(train_scores),
            'val_score_mean': np.mean(val_scores),
            'nu': model_params.get('nu', self.nu)
        }
        
        logger.info(f"One-Class SVM training completed.")
        logger.info(f"Train anomaly rate: {self.metrics['train_anomaly_rate']:.4f}")
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict kicks using One-Class SVM"""
        
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_pred = X[self.feature_columns].fillna(X[self.feature_columns].mean())
        X_scaled = self.scaler.transform(X_pred)
        
        predictions = self.model.predict(X_scaled)
        return (predictions == -1).astype(int)