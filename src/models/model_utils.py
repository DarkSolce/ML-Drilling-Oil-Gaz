"""
Model Utilities for ML Drilling Project
=======================================

Utilitaires communs pour tous les modèles de machine learning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Classe pour évaluer les performances des modèles
    """
    
    @staticmethod
    def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Évalue un modèle de régression
        
        Args:
            y_true: Valeurs réelles
            y_pred: Prédictions
            
        Returns:
            Dictionnaire avec les métriques
        """
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    @staticmethod
    def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, 
                              y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Évalue un modèle de classification
        
        Args:
            y_true: Valeurs réelles
            y_pred: Prédictions
            y_proba: Probabilités (optionnel)
            
        Returns:
            Dictionnaire avec les métriques
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_proba is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_proba)
            except ValueError:
                logger.warning("Impossible de calculer l'AUC")
        
        return metrics

class ModelSelector:
    """
    Classe pour la sélection et l'optimisation des modèles
    """
    
    def __init__(self, models: Dict[str, Any], param_grids: Dict[str, Dict]):
        """
        Initialise le sélecteur de modèles
        
        Args:
            models: Dictionnaire des modèles {nom: instance}
            param_grids: Grilles de paramètres pour chaque modèle
        """
        self.models = models
        self.param_grids = param_grids
        self.results = {}
    
    def compare_models(self, X_train: np.ndarray, y_train: np.ndarray,
                      cv: int = 5, scoring: str = 'neg_mean_squared_error') -> pd.DataFrame:
        """
        Compare les modèles avec validation croisée
        
        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement
            cv: Nombre de folds pour la CV
            scoring: Métrique de scoring
            
        Returns:
            DataFrame avec les résultats de comparaison
        """
        results = []
        
        for name, model in self.models.items():
            logger.info(f"Évaluation du modèle: {name}")
            
            try:
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
                
                results.append({
                    'Model': name,
                    'Mean_Score': scores.mean(),
                    'Std_Score': scores.std(),
                    'Min_Score': scores.min(),
                    'Max_Score': scores.max()
                })
                
                self.results[name] = scores
                
            except Exception as e:
                logger.error(f"Erreur avec le modèle {name}: {e}")
                continue
        
        return pd.DataFrame(results).sort_values('Mean_Score', ascending=False)
    
    def optimize_hyperparameters(self, model_name: str, X_train: np.ndarray,
                                y_train: np.ndarray, method: str = 'grid',
                                cv: int = 5, n_iter: int = 100) -> Tuple[Any, Dict]:
        """
        Optimise les hyperparamètres d'un modèle
        
        Args:
            model_name: Nom du modèle à optimiser
            X_train: Features d'entraînement
            y_train: Target d'entraînement
            method: 'grid' ou 'random'
            cv: Nombre de folds
            n_iter: Nombre d'itérations pour RandomizedSearch
            
        Returns:
            Meilleur modèle et ses paramètres
        """
        if model_name not in self.models:
            raise ValueError(f"Modèle {model_name} non trouvé")
        
        model = self.models[model_name]
        param_grid = self.param_grids.get(model_name, {})
        
        if method == 'grid':
            search = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1)
        else:
            search = RandomizedSearchCV(model, param_grid, n_iter=n_iter,
                                      cv=cv, n_jobs=-1, random_state=42)
        
        logger.info(f"Optimisation des hyperparamètres pour {model_name}")
        search.fit(X_train, y_train)
        
        return search.best_estimator_, search.best_params_

class ModelPersistence:
    """
    Classe pour sauvegarder et charger les modèles
    """
    
    @staticmethod
    def save_model(model: Any, filepath: str, metadata: Optional[Dict] = None):
        """
        Sauvegarde un modèle
        
        Args:
            model: Modèle à sauvegarder
            filepath: Chemin de sauvegarde
            metadata: Métadonnées optionnelles
        """
        # Créer le dossier si nécessaire
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder le modèle
        joblib.dump(model, filepath)
        
        # Sauvegarder les métadonnées
        if metadata:
            metadata_path = str(filepath).replace('.pkl', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Modèle sauvegardé: {filepath}")
    
    @staticmethod
    def load_model(filepath: str) -> Tuple[Any, Optional[Dict]]:
        """
        Charge un modèle
        
        Args:
            filepath: Chemin du modèle
            
        Returns:
            Modèle et métadonnées
        """
        # Charger le modèle
        model = joblib.load(filepath)
        
        # Charger les métadonnées si elles existent
        metadata = None
        metadata_path = str(filepath).replace('.pkl', '_metadata.json')
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        logger.info(f"Modèle chargé: {filepath}")
        return model, metadata

class FeatureImportanceAnalyzer:
    """
    Analyse l'importance des features
    """
    
    @staticmethod
    def get_feature_importance(model: Any, feature_names: List[str]) -> pd.DataFrame:
        """
        Extrait l'importance des features
        
        Args:
            model: Modèle entraîné
            feature_names: Noms des features
            
        Returns:
            DataFrame avec l'importance des features
        """
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).flatten()
        else:
            logger.warning("Modèle ne supporte pas l'extraction d'importance")
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    @staticmethod
    def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20):
        """
        Visualise l'importance des features
        
        Args:
            importance_df: DataFrame avec l'importance
            top_n: Nombre de features à afficher
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Features les plus importantes')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

class ModelMonitor:
    """
    Classe pour monitorer les performances des modèles en production
    """
    
    def __init__(self):
        self.predictions_log = []
        self.performance_log = []
    
    def log_prediction(self, input_data: Dict, prediction: Any, 
                      actual: Optional[Any] = None, timestamp: Optional[str] = None):
        """
        Log une prédiction
        
        Args:
            input_data: Données d'entrée
            prediction: Prédiction du modèle
            actual: Valeur réelle (si disponible)
            timestamp: Timestamp (optionnel)
        """
        log_entry = {
            'timestamp': timestamp or pd.Timestamp.now().isoformat(),
            'input_data': input_data,
            'prediction': prediction,
            'actual': actual
        }
        
        self.predictions_log.append(log_entry)
    
    def calculate_drift(self, reference_data: np.ndarray, 
                       current_data: np.ndarray, threshold: float = 0.1) -> Dict:
        """
        Calcule la dérive des données
        
        Args:
            reference_data: Données de référence
            current_data: Données actuelles
            threshold: Seuil de dérive
            
        Returns:
            Dictionnaire avec les métriques de dérive
        """
        from scipy.stats import ks_2samp
        
        # Test de Kolmogorov-Smirnov
        ks_stat, p_value = ks_2samp(reference_data.flatten(), 
                                   current_data.flatten())
        
        drift_detected = ks_stat > threshold
        
        return {
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'drift_detected': drift_detected,
            'threshold': threshold
        }

# Fonctions utilitaires
def create_model_registry():
    """
    Crée un registre des modèles disponibles
    
    Returns:
        Dictionnaire avec les modèles et paramètres
    """
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.svm import SVR, SVC
    from sklearn.linear_model import LinearRegression, LogisticRegression
    
    models_regression = {
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'SVR': SVR(),
        'LinearRegression': LinearRegression()
    }
    
    models_classification = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'SVC': SVC(probability=True, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42)
    }
    
    param_grids_regression = {
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10]
        },
        'GradientBoosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'SVR': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }
    }
    
    param_grids_classification = {
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10]
        },
        'GradientBoosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'SVC': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }
    }
    
    return {
        'regression': {
            'models': models_regression,
            'param_grids': param_grids_regression
        },
        'classification': {
            'models': models_classification,
            'param_grids': param_grids_classification
        }
    }