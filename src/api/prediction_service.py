"""
Prediction Service for ML Drilling API
======================================

Service de prédiction pour les modèles de forage
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Imports des modules du projet
try:
    from src.data.data_preprocessor import DrillingDataPreprocessor
    from src.models.base_model import BaseDrillingModel
    from src.utils.logging_utils import APILogger
except ImportError:
    # Fallback pour développement
    import sys
    sys.path.append('.')

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Gestionnaire des modèles chargés en mémoire
    """
    
    def __init__(self, models_dir: str = 'models'):
        """
        Initialise le gestionnaire de modèles
        
        Args:
            models_dir: Dossier contenant les modèles sauvegardés
        """
        self.models_dir = Path(models_dir)
        self.loaded_models = {}
        self.model_metadata = {}
        self.preprocessors = {}
        
        # Logger API
        self.api_logger = APILogger()
        
        logger.info(f"ModelManager initialisé avec models_dir: {self.models_dir}")
    
    def load_model(self, model_type: str, model_name: str) -> bool:
        """
        Charge un modèle en mémoire
        
        Args:
            model_type: Type de modèle (formation_pressure, kick_detection)
            model_name: Nom du modèle
            
        Returns:
            True si le chargement a réussi
        """
        try:
            model_path = self.models_dir / model_type / f"{model_name}.pkl"
            metadata_path = self.models_dir / model_type / f"{model_name}_metadata.json"
            preprocessor_path = self.models_dir / model_type / f"{model_name}_preprocessor.pkl"
            
            if not model_path.exists():
                logger.error(f"Modèle non trouvé: {model_path}")
                return False
            
            # Charger le modèle
            model = joblib.load(model_path)
            model_key = f"{model_type}_{model_name}"
            self.loaded_models[model_key] = model
            
            # Charger les métadonnées
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.model_metadata[model_key] = json.load(f)
            
            # Charger le preprocessor
            if preprocessor_path.exists():
                preprocessor = joblib.load(preprocessor_path)
                self.preprocessors[model_key] = preprocessor
            
            logger.info(f"Modèle chargé avec succès: {model_key}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle {model_type}/{model_name}: {e}")
            return False
    
    def get_model(self, model_type: str, model_name: str) -> Optional[Any]:
        """
        Récupère un modèle chargé
        
        Args:
            model_type: Type de modèle
            model_name: Nom du modèle
            
        Returns:
            Modèle ou None si non trouvé
        """
        model_key = f"{model_type}_{model_name}"
        return self.loaded_models.get(model_key)
    
    def get_preprocessor(self, model_type: str, model_name: str) -> Optional[Any]:
        """
        Récupère le preprocessor d'un modèle
        
        Args:
            model_type: Type de modèle
            model_name: Nom du modèle
            
        Returns:
            Preprocessor ou None si non trouvé
        """
        model_key = f"{model_type}_{model_name}"
        return self.preprocessors.get(model_key)
    
    def get_model_info(self, model_type: str, model_name: str) -> Optional[Dict]:
        """
        Récupère les informations d'un modèle
        
        Args:
            model_type: Type de modèle
            model_name: Nom du modèle
            
        Returns:
            Métadonnées du modèle
        """
        model_key = f"{model_type}_{model_name}"
        return self.model_metadata.get(model_key)
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """
        Liste tous les modèles disponibles
        
        Returns:
            Dictionnaire {type: [liste_des_modèles]}
        """
        available_models = {}
        
        if not self.models_dir.exists():
            return available_models
        
        for model_type_dir in self.models_dir.iterdir():
            if model_type_dir.is_dir():
                model_type = model_type_dir.name
                models = []
                
                for model_file in model_type_dir.glob("*.pkl"):
                    if not model_file.name.endswith(('_metadata.pkl', '_preprocessor.pkl')):
                        model_name = model_file.stem
                        models.append(model_name)
                
                available_models[model_type] = models
        
        return available_models

class PredictionService:
    """
    Service principal de prédiction
    """
    
    def __init__(self, models_dir: str = 'models'):
        """
        Initialise le service de prédiction
        
        Args:
            models_dir: Dossier des modèles
        """
        self.model_manager = ModelManager(models_dir)
        self.prediction_cache = {}
        self.cache_max_size = 1000
        
        # Charger les modèles par défaut
        self._load_default_models()
        
        logger.info("PredictionService initialisé")
    
    def _load_default_models(self):
        """Charge les modèles par défaut au démarrage"""
        available_models = self.model_manager.list_available_models()
        
        for model_type, models in available_models.items():
            if models:
                # Charger le premier modèle de chaque type
                default_model = models[0]
                self.model_manager.load_model(model_type, default_model)
                logger.info(f"Modèle par défaut chargé: {model_type}/{default_model}")
    
    def predict_formation_pressure(self, input_data: Dict[str, Any], 
                                 model_name: str = None) -> Dict[str, Any]:
        """
        Prédiction de la pression de formation
        
        Args:
            input_data: Données d'entrée
            model_name: Nom du modèle à utiliser
            
        Returns:
            Résultat de la prédiction
        """
        start_time = datetime.now()
        model_type = 'formation_pressure'
        
        try:
            # Sélectionner le modèle
            if model_name is None:
                available_models = self.model_manager.list_available_models()
                if model_type in available_models and available_models[model_type]:
                    model_name = available_models[model_type][0]
                else:
                    raise ValueError(f"Aucun modèle {model_type} disponible")
            
            # Récupérer le modèle
            model = self.model_manager.get_model(model_type, model_name)
            if model is None:
                # Essayer de charger le modèle
                if not self.model_manager.load_model(model_type, model_name):
                    raise ValueError(f"Impossible de charger le modèle {model_type}/{model_name}")
                model = self.model_manager.get_model(model_type, model_name)
            
            # Préparer les données
            features_df = self._prepare_formation_pressure_features(input_data)
            
            # Préprocessing si disponible
            preprocessor = self.model_manager.get_preprocessor(model_type, model_name)
            if preprocessor:
                features_array = preprocessor.transform(features_df)
            else:
                features_array = features_df.values
            
            # Prédiction
            prediction = model.predict(features_array)[0]
            
            # Calculer la confiance si le modèle le supporte
            confidence = None
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(features_array)[0]
                    confidence = float(np.max(proba))
                except:
                    pass
            
            # Métadonnées du modèle
            model_info = self.model_manager.get_model_info(model_type, model_name)
            
            # Calculer le temps de traitement
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'prediction': float(prediction),
                'confidence': confidence,
                'model_name': model_name,
                'model_type': model_type,
                'processing_time_seconds': processing_time,
                'timestamp': datetime.now().isoformat(),
                'model_info': model_info
            }
            
            # Logger la prédiction
            self.model_manager.api_logger.log_prediction_request(
                model_type, input_data, prediction, processing_time
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction de pression: {e}")
            return {
                'error': str(e),
                'model_type': model_type,
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_kick_detection(self, input_data: Dict[str, Any], 
                             model_name: str = None) -> Dict[str, Any]:
        """
        Prédiction de détection de kick
        
        Args:
            input_data: Données d'entrée
            model_name: Nom du modèle à utiliser
            
        Returns:
            Résultat de la prédiction
        """
        start_time = datetime.now()
        model_type = 'kick_detection'
        
        try:
            # Sélectionner le modèle
            if model_name is None:
                available_models = self.model_manager.list_available_models()
                if model_type in available_models and available_models[model_type]:
                    model_name = available_models[model_type][0]
                else:
                    raise ValueError(f"Aucun modèle {model_type} disponible")
            
            # Récupérer le modèle
            model = self.model_manager.get_model(model_type, model_name)
            if model is None:
                # Essayer de charger le modèle
                if not self.model_manager.load_model(model_type, model_name):
                    raise ValueError(f"Impossible de charger le modèle {model_type}/{model_name}")
                model = self.model_manager.get_model(model_type, model_name)
            
            # Préparer les données
            features_df = self._prepare_kick_detection_features(input_data)
            
            # Préprocessing si disponible
            preprocessor = self.model_manager.get_preprocessor(model_type, model_name)
            if preprocessor:
                features_array = preprocessor.transform(features_df)
            else:
                features_array = features_df.values
            
            # Prédiction
            prediction = model.predict(features_array)[0]
            
            # Calculer les probabilités
            kick_probability = None
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(features_array)[0]
                    if len(probabilities) == 2:  # Classification binaire
                        kick_probability = float(probabilities[1])  # Probabilité de kick
                except:
                    pass
            
            # Métadonnées du modèle
            model_info = self.model_manager.get_model_info(model_type, model_name)
            
            # Calculer le temps de traitement
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Déterminer le niveau d'alerte
            alert_level = self._determine_alert_level(prediction, kick_probability)
            
            result = {
                'prediction': int(prediction),  # 0: Normal, 1: Kick détecté
                'kick_probability': kick_probability,
                'alert_level': alert_level,
                'model_name': model_name,
                'model_type': model_type,
                'processing_time_seconds': processing_time,
                'timestamp': datetime.now().isoformat(),
                'model_info': model_info
            }
            
            # Logger la prédiction
            self.model_manager.api_logger.log_prediction_request(
                model_type, input_data, prediction, processing_time
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de la détection de kick: {e}")
            return {
                'error': str(e),
                'model_type': model_type,
                'timestamp': datetime.now().isoformat()
            }
    
    def _prepare_formation_pressure_features(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prépare les features pour la prédiction de pression de formation
        
        Args:
            input_data: Données d'entrée
            
        Returns:
            DataFrame avec les features préparées
        """
        # Features attendues pour la prédiction de pression
        expected_features = [
            'Depth', 'MudWeight', 'Temperature', 'Porosity', 'Permeability'
        ]
        
        # Créer le DataFrame avec les features disponibles
        features = {}
        for feature in expected_features:
            if feature in input_data:
                features[feature] = [input_data[feature]]
            else:
                # Valeurs par défaut si la feature est manquante
                default_values = {
                    'Depth': 1000.0,
                    'MudWeight': 1.2,
                    'Temperature': 25.0,
                    'Porosity': 0.15,
                    'Permeability': 100.0
                }
                features[feature] = [default_values.get(feature, 0.0)]
                logger.warning(f"Feature manquante: {feature}, utilisation de la valeur par défaut")
        
        return pd.DataFrame(features)
    
    def _prepare_kick_detection_features(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prépare les features pour la détection de kick
        
        Args:
            input_data: Données d'entrée
            
        Returns:
            DataFrame avec les features préparées
        """
        # Features attendues pour la détection de kick
        expected_features = [
            'FlowRateIn', 'FlowRateOut', 'StandpipePressure', 'CasingPressure',
            'MudWeight', 'HookLoad', 'RPM', 'Torque', 'ROP'
        ]
        
        # Créer le DataFrame avec les features disponibles
        features = {}
        for feature in expected_features:
            if feature in input_data:
                features[feature] = [input_data[feature]]
            else:
                # Valeurs par défaut si la feature est manquante
                default_values = {
                    'FlowRateIn': 350.0,
                    'FlowRateOut': 350.0,
                    'StandpipePressure': 200.0,
                    'CasingPressure': 50.0,
                    'MudWeight': 1.2,
                    'HookLoad': 180.0,
                    'RPM': 120.0,
                    'Torque': 15.0,
                    'ROP': 10.0
                }
                features[feature] = [default_values.get(feature, 0.0)]
                logger.warning(f"Feature manquante: {feature}, utilisation de la valeur par défaut")
        
        return pd.DataFrame(features)
    
    def _determine_alert_level(self, prediction: int, kick_probability: Optional[float]) -> str:
        """
        Détermine le niveau d'alerte basé sur la prédiction
        
        Args:
            prediction: Prédiction binaire (0 ou 1)
            kick_probability: Probabilité de kick
            
        Returns:
            Niveau d'alerte ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
        """
        if prediction == 0:  # Pas de kick prédit
            if kick_probability and kick_probability > 0.3:
                return 'MEDIUM'  # Probabilité élevée malgré prédiction négative
            else:
                return 'LOW'
        else:  # Kick prédit
            if kick_probability:
                if kick_probability >= 0.8:
                    return 'CRITICAL'
                elif kick_probability >= 0.6:
                    return 'HIGH'
                else:
                    return 'MEDIUM'
            else:
                return 'HIGH'  # Par défaut si pas de probabilité
    
    def batch_predict(self, batch_data: List[Dict[str, Any]], 
                     model_type: str, model_name: str = None) -> List[Dict[str, Any]]:
        """
        Prédictions en lot
        
        Args:
            batch_data: Liste des données d'entrée
            model_type: Type de modèle
            model_name: Nom du modèle
            
        Returns:
            Liste des résultats de prédiction
        """
        results = []
        
        for data in batch_data:
            if model_type == 'formation_pressure':
                result = self.predict_formation_pressure(data, model_name)
            elif model_type == 'kick_detection':
                result = self.predict_kick_detection(data, model_name)
            else:
                result = {'error': f'Type de modèle non supporté: {model_type}'}
            
            results.append(result)
        
        return results
    
    def get_model_health(self) -> Dict[str, Any]:
        """
        Vérifie la santé des modèles chargés
        
        Returns:
            Statut de santé des modèles
        """
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'loaded_models': len(self.model_manager.loaded_models),
            'available_models': self.model_manager.list_available_models(),
            'cache_size': len(self.prediction_cache),
            'status': 'healthy'
        }
        
        # Vérifier que chaque type de modèle a au moins un modèle chargé
        required_types = ['formation_pressure', 'kick_detection']
        missing_types = []
        
        for model_type in required_types:
            loaded = any(key.startswith(model_type) for key in self.model_manager.loaded_models.keys())
            if not loaded:
                missing_types.append(model_type)
        
        if missing_types:
            health_status['status'] = 'warning'
            health_status['missing_model_types'] = missing_types
        
        return health_status
    
    def reload_models(self) -> Dict[str, Any]:
        """
        Recharge tous les modèles
        
        Returns:
            Résultat du rechargement
        """
        try:
            # Vider le cache
            self.model_manager.loaded_models.clear()
            self.model_manager.model_metadata.clear()
            self.model_manager.preprocessors.clear()
            self.prediction_cache.clear()
            
            # Recharger les modèles par défaut
            self._load_default_models()
            
            return {
                'status': 'success',
                'message': 'Modèles rechargés avec succès',
                'timestamp': datetime.now().isoformat(),
                'loaded_models': len(self.model_manager.loaded_models)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du rechargement des modèles: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """
        Retourne les statistiques de prédiction
        
        Returns:
            Statistiques de prédiction
        """
        # Cette méthode nécessiterait une base de données pour stocker
        # l'historique des prédictions en production
        return {
            'total_predictions': 0,
            'predictions_by_type': {},
            'average_processing_time': 0.0,
            'error_rate': 0.0,
            'timestamp': datetime.now().isoformat()
        }
    
    def validate_input_data(self, input_data: Dict[str, Any], model_type: str) -> Tuple[bool, str]:
        """
        Valide les données d'entrée pour un type de modèle
        
        Args:
            input_data: Données à valider
            model_type: Type de modèle
            
        Returns:
            (is_valid, error_message)
        """
        if model_type == 'formation_pressure':
            required_fields = ['Depth', 'MudWeight']
            for field in required_fields:
                if field not in input_data:
                    return False, f"Champ obligatoire manquant: {field}"
                if not isinstance(input_data[field], (int, float)):
                    return False, f"Type invalide pour {field}, attendu: nombre"
                if input_data[field] < 0:
                    return False, f"Valeur négative non autorisée pour {field}"
            
            # Validations spécifiques
            if input_data['Depth'] > 10000:  # Profondeur max 10km
                return False, "Profondeur trop importante (max 10000m)"
            if input_data['MudWeight'] < 0.8 or input_data['MudWeight'] > 3.0:
                return False, "Densité de boue hors limites (0.8-3.0)"
        
        elif model_type == 'kick_detection':
            required_fields = ['FlowRateIn', 'FlowRateOut', 'StandpipePressure']
            for field in required_fields:
                if field not in input_data:
                    return False, f"Champ obligatoire manquant: {field}"
                if not isinstance(input_data[field], (int, float)):
                    return False, f"Type invalide pour {field}, attendu: nombre"
                if input_data[field] < 0:
                    return False, f"Valeur négative non autorisée pour {field}"
            
            # Validations spécifiques
            if input_data['FlowRateIn'] > 1000:  # Débit max 1000 L/min
                return False, "Débit d'entrée trop élevé (max 1000 L/min)"
            if input_data['StandpipePressure'] > 500:  # Pression max 500 bar
                return False, "Pression standpipe trop élevée (max 500 bar)"
        
        else:
            return False, f"Type de modèle non supporté: {model_type}"
        
        return True, ""
    
    def get_feature_importance(self, model_type: str, model_name: str = None) -> Dict[str, Any]:
        """
        Récupère l'importance des features d'un modèle
        
        Args:
            model_type: Type de modèle
            model_name: Nom du modèle
            
        Returns:
            Importance des features
        """
        try:
            if model_name is None:
                available_models = self.model_manager.list_available_models()
                if model_type in available_models and available_models[model_type]:
                    model_name = available_models[model_type][0]
                else:
                    return {'error': f'Aucun modèle {model_type} disponible'}
            
            model = self.model_manager.get_model(model_type, model_name)
            if model is None:
                return {'error': f'Modèle {model_type}/{model_name} non trouvé'}
            
            # Récupérer les noms de features
            if model_type == 'formation_pressure':
                feature_names = ['Depth', 'MudWeight', 'Temperature', 'Porosity', 'Permeability']
            elif model_type == 'kick_detection':
                feature_names = ['FlowRateIn', 'FlowRateOut', 'StandpipePressure', 
                               'CasingPressure', 'MudWeight', 'HookLoad', 'RPM', 'Torque', 'ROP']
            else:
                return {'error': f'Type de modèle non supporté: {model_type}'}
            
            # Extraire l'importance des features
            importance = None
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_).flatten()
            else:
                return {'error': 'Modèle ne supporte pas l\'extraction d\'importance'}
            
            # Créer le dictionnaire importance
            feature_importance = {}
            for i, name in enumerate(feature_names):
                if i < len(importance):
                    feature_importance[name] = float(importance[i])
            
            # Trier par importance décroissante
            sorted_importance = dict(sorted(feature_importance.items(), 
                                          key=lambda x: x[1], reverse=True))
            
            return {
                'model_type': model_type,
                'model_name': model_name,
                'feature_importance': sorted_importance,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction d'importance: {e}")
            return {'error': str(e)}
    
    def predict_with_uncertainty(self, input_data: Dict[str, Any], 
                                model_type: str, model_name: str = None,
                                n_samples: int = 100) -> Dict[str, Any]:
        """
        Prédiction avec estimation d'incertitude (si le modèle le supporte)
        
        Args:
            input_data: Données d'entrée
            model_type: Type de modèle
            model_name: Nom du modèle
            n_samples: Nombre d'échantillons pour l'estimation d'incertitude
            
        Returns:
            Prédiction avec intervalles de confiance
        """
        try:
            # Prédiction normale
            if model_type == 'formation_pressure':
                base_result = self.predict_formation_pressure(input_data, model_name)
            elif model_type == 'kick_detection':
                base_result = self.predict_kick_detection(input_data, model_name)
            else:
                return {'error': f'Type de modèle non supporté: {model_type}'}
            
            if 'error' in base_result:
                return base_result
            
            # Pour des modèles probabilistes, on peut estimer l'incertitude
            model = self.model_manager.get_model(model_type, model_name or base_result['model_name'])
            
            if hasattr(model, 'predict_proba') and model_type == 'kick_detection':
                # Pour la classification, l'incertitude peut être estimée via l'entropie
                if model_type == 'kick_detection':
                    features_df = self._prepare_kick_detection_features(input_data)
                else:
                    features_df = self._prepare_formation_pressure_features(input_data)
                
                preprocessor = self.model_manager.get_preprocessor(model_type, base_result['model_name'])
                if preprocessor:
                    features_array = preprocessor.transform(features_df)
                else:
                    features_array = features_df.values
                
                probabilities = model.predict_proba(features_array)[0]
                
                # Calculer l'entropie comme mesure d'incertitude
                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-8))
                max_entropy = np.log2(len(probabilities))
                uncertainty = entropy / max_entropy
                
                base_result['uncertainty'] = float(uncertainty)
                base_result['entropy'] = float(entropy)
                base_result['probabilities'] = probabilities.tolist()
            
            elif model_type == 'formation_pressure':
                # Pour la régression, on peut essayer de faire du bootstrap si possible
                base_result['uncertainty_note'] = "Estimation d'incertitude non disponible pour ce modèle"
            
            return base_result
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction avec incertitude: {e}")
            return {'error': str(e)}


class AlertManager:
    """
    Gestionnaire d'alertes pour les prédictions critiques
    """
    
    def __init__(self):
        """Initialise le gestionnaire d'alertes"""
        self.alert_threshold = 0.7  # Seuil de probabilité pour déclencher une alerte
        self.alert_history = []
        self.max_history = 1000
        self.alert_counts = {
            'KICK_DETECTION': 0,
            'HIGH_PRESSURE': 0,
            'LOW_CONFIDENCE': 0
        }
    
    def check_alert_conditions(self, prediction_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Vérifie si une alerte doit être déclenchée
        
        Args:
            prediction_result: Résultat de prédiction
            
        Returns:
            Alerte si conditions remplies, None sinon
        """
        alert = None
        
        # Alerte pour kick detection
        if prediction_result.get('model_type') == 'kick_detection':
            kick_prob = prediction_result.get('kick_probability', 0)
            if kick_prob >= self.alert_threshold:
                alert = {
                    'type': 'KICK_DETECTION',
                    'severity': 'CRITICAL' if kick_prob >= 0.9 else 'HIGH',
                    'message': f'Kick détecté avec probabilité {kick_prob:.2%}',
                    'timestamp': datetime.now().isoformat(),
                    'data': prediction_result
                }
                self.alert_counts['KICK_DETECTION'] += 1
        
        # Alerte pour pression anormale
        elif prediction_result.get('model_type') == 'formation_pressure':
            predicted_pressure = prediction_result.get('prediction', 0)
            if predicted_pressure > 5000:  # Exemple de seuil
                alert = {
                    'type': 'HIGH_PRESSURE',
                    'severity': 'HIGH',
                    'message': f'Pression élevée détectée: {predicted_pressure:.0f} psi',
                    'timestamp': datetime.now().isoformat(),
                    'data': prediction_result
                }
                self.alert_counts['HIGH_PRESSURE'] += 1
        
        # Alerte pour faible confiance
        confidence = prediction_result.get('confidence')
        if confidence and confidence < 0.6:
            alert = {
                'type': 'LOW_CONFIDENCE',
                'severity': 'MEDIUM',
                'message': f'Prédiction avec faible confiance: {confidence:.2%}',
                'timestamp': datetime.now().isoformat(),
                'data': prediction_result
            }
            self.alert_counts['LOW_CONFIDENCE'] += 1
        
        # Enregistrer l'alerte
        if alert:
            self.alert_history.append(alert)
            if len(self.alert_history) > self.max_history:
                self.alert_history = self.alert_history[-self.max_history:]
            
            logger.warning(f"ALERTE GÉNÉRÉE: {alert['message']}")
        
        return alert
    
    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retourne les alertes récentes
        
        Args:
            limit: Nombre maximum d'alertes à retourner
            
        Returns:
            Liste des alertes récentes
        """
        return self.alert_history[-limit:]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """
        Retourne les statistiques d'alertes
        
        Returns:
            Statistiques des alertes
        """
        total_alerts = sum(self.alert_counts.values())
        
        return {
            'total_alerts': total_alerts,
            'alert_counts_by_type': self.alert_counts.copy(),
            'recent_alerts_count': len(self.alert_history),
            'alert_rates': {
                alert_type: count / max(total_alerts, 1)
                for alert_type, count in self.alert_counts.items()
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def clear_alerts(self, alert_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Efface les alertes
        
        Args:
            alert_type: Type d'alerte à effacer, ou None pour tout effacer
            
        Returns:
            Résultat de l'opération
        """
        if alert_type:
            # Effacer seulement un type d'alerte
            self.alert_counts[alert_type] = 0
            self.alert_history = [
                alert for alert in self.alert_history
                if alert['type'] != alert_type
            ]
            message = f"Alertes de type {alert_type} effacées"
        else:
            # Effacer toutes les alertes
            self.alert_counts = {key: 0 for key in self.alert_counts}
            self.alert_history.clear()
            message = "Toutes les alertes effacées"
        
        return {
            'status': 'success',
            'message': message,
            'timestamp': datetime.now().isoformat()
        }


class PredictionCache:
    """
    Cache pour les prédictions fréquentes
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """
        Initialise le cache
        
        Args:
            max_size: Taille maximale du cache
            ttl_seconds: Durée de vie des entrées (TTL)
        """
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def _generate_cache_key(self, model_type: str, model_name: str, 
                           input_data: Dict[str, Any]) -> str:
        """
        Génère une clé de cache unique
        
        Args:
            model_type: Type de modèle
            model_name: Nom du modèle
            input_data: Données d'entrée
            
        Returns:
            Clé de cache
        """
        import hashlib
        
        # Créer une représentation stable des données
        data_str = f"{model_type}:{model_name}:{sorted(input_data.items())}"
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get(self, model_type: str, model_name: str, 
            input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Récupère une prédiction du cache
        
        Args:
            model_type: Type de modèle
            model_name: Nom du modèle
            input_data: Données d'entrée
            
        Returns:
            Prédiction cachée ou None
        """
        cache_key = self._generate_cache_key(model_type, model_name, input_data)
        
        if cache_key in self.cache:
            # Vérifier si l'entrée n'a pas expiré
            cache_time = self.access_times[cache_key]
            if (datetime.now() - cache_time).total_seconds() <= self.ttl_seconds:
                # Mettre à jour le temps d'accès
                self.access_times[cache_key] = datetime.now()
                return self.cache[cache_key]
            else:
                # Supprimer l'entrée expirée
                del self.cache[cache_key]
                del self.access_times[cache_key]
        
        return None
    
    def put(self, model_type: str, model_name: str, 
            input_data: Dict[str, Any], prediction: Dict[str, Any]):
        """
        Stocke une prédiction dans le cache
        
        Args:
            model_type: Type de modèle
            model_name: Nom du modèle
            input_data: Données d'entrée
            prediction: Résultat de prédiction
        """
        cache_key = self._generate_cache_key(model_type, model_name, input_data)
        
        # Si le cache est plein, supprimer l'entrée la plus ancienne
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        # Ajouter la nouvelle entrée
        self.cache[cache_key] = prediction.copy()
        self.access_times[cache_key] = datetime.now()
    
    def clear(self):
        """Vide le cache"""
        self.cache.clear()
        self.access_times.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retourne les statistiques du cache
        
        Returns:
            Statistiques du cache
        """
        return {
            'cache_size': len(self.cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds,
            'hit_rate': 0.0,  # À implémenter avec des compteurs
            'oldest_entry': min(self.access_times.values()) if self.access_times else None,
            'newest_entry': max(self.access_times.values()) if self.access_times else None
        }


# Extensions du PredictionService avec cache et alertes
class EnhancedPredictionService(PredictionService):
    """
    Version améliorée du service de prédiction avec cache et alertes
    """
    
    def __init__(self, models_dir: str = 'models', enable_cache: bool = True,
                 enable_alerts: bool = True):
        """
        Initialise le service amélioré
        
        Args:
            models_dir: Dossier des modèles
            enable_cache: Activer le cache
            enable_alerts: Activer les alertes
        """
        super().__init__(models_dir)
        
        self.enable_cache = enable_cache
        self.enable_alerts = enable_alerts
        
        if enable_cache:
            self.cache = PredictionCache()
        
        if enable_alerts:
            self.alert_manager = AlertManager()
        
        logger.info(f"EnhancedPredictionService initialisé - Cache: {enable_cache}, Alertes: {enable_alerts}")
    
    def predict_formation_pressure(self, input_data: Dict[str, Any], 
                                 model_name: str = None) -> Dict[str, Any]:
        """Version avec cache de la prédiction de pression"""
        model_type = 'formation_pressure'
        
        # Vérifier le cache si activé
        if self.enable_cache and model_name:
            cached_result = self.cache.get(model_type, model_name, input_data)
            if cached_result:
                cached_result['from_cache'] = True
                return cached_result
        
        # Prédiction normale
        result = super().predict_formation_pressure(input_data, model_name)
        
        # Stocker dans le cache si activé
        if self.enable_cache and 'error' not in result:
            self.cache.put(model_type, result['model_name'], input_data, result)
        
        # Vérifier les alertes si activé
        if self.enable_alerts and 'error' not in result:
            alert = self.alert_manager.check_alert_conditions(result)
            if alert:
                result['alert'] = alert
        
        result['from_cache'] = False
        return result
    
    def predict_kick_detection(self, input_data: Dict[str, Any], 
                             model_name: str = None) -> Dict[str, Any]:
        """Version avec cache de la détection de kick"""
        model_type = 'kick_detection'
        
        # Vérifier le cache si activé
        if self.enable_cache and model_name:
            cached_result = self.cache.get(model_type, model_name, input_data)
            if cached_result:
                cached_result['from_cache'] = True
                return cached_result
        
        # Prédiction normale
        result = super().predict_kick_detection(input_data, model_name)
        
        # Stocker dans le cache si activé
        if self.enable_cache and 'error' not in result:
            self.cache.put(model_type, result['model_name'], input_data, result)
        
        # Vérifier les alertes si activé
        if self.enable_alerts and 'error' not in result:
            alert = self.alert_manager.check_alert_conditions(result)
            if alert:
                result['alert'] = alert
        
        result['from_cache'] = False
        return result
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """
        Retourne les statistiques complètes du service
        
        Returns:
            Statistiques du service
        """
        stats = {
            'timestamp': datetime.now().isoformat(),
            'model_health': self.get_model_health(),
            'prediction_stats': self.get_prediction_statistics()
        }
        
        if self.enable_cache:
            stats['cache_stats'] = self.cache.get_statistics()
        
        if self.enable_alerts:
            stats['alert_stats'] = self.alert_manager.get_alert_statistics()
        
        return stats
    
    def clear_cache(self) -> Dict[str, Any]:
        """
        Vide le cache de prédictions
        
        Returns:
            Résultat de l'opération
        """
        if not self.enable_cache:
            return {'status': 'error', 'message': 'Cache non activé'}
        
        self.cache.clear()
        return {
            'status': 'success',
            'message': 'Cache vidé avec succès',
            'timestamp': datetime.now().isoformat()
        }
    
    def get_alerts(self, limit: int = 50) -> Dict[str, Any]:
        """
        Retourne les alertes récentes
        
        Args:
            limit: Nombre maximum d'alertes
            
        Returns:
            Alertes récentes
        """
        if not self.enable_alerts:
            return {'status': 'error', 'message': 'Alertes non activées'}
        
        return {
            'status': 'success',
            'alerts': self.alert_manager.get_recent_alerts(limit),
            'statistics': self.alert_manager.get_alert_statistics(),
            'timestamp': datetime.now().isoformat()
        }


# Fonctions utilitaires
def create_prediction_service(config: Optional[Dict[str, Any]] = None) -> EnhancedPredictionService:
    """
    Factory function pour créer un service de prédiction
    
    Args:
        config: Configuration du service
        
    Returns:
        Service de prédiction configuré
    """
    if config is None:
        config = {}
    
    models_dir = config.get('models_dir', 'models')
    enable_cache = config.get('enable_cache', True)
    enable_alerts = config.get('enable_alerts', True)
    
    return EnhancedPredictionService(
        models_dir=models_dir,
        enable_cache=enable_cache,
        enable_alerts=enable_alerts
    )

def validate_prediction_input(data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validation générale des données d'entrée
    
    Args:
        data: Données à valider
        
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(data, dict):
        return False, "Les données doivent être un dictionnaire"
    
    if not data:
        return False, "Données vides"
    
    # Vérifier que toutes les valeurs sont des nombres
    for key, value in data.items():
        if not isinstance(value, (int, float)):
            return False, f"Valeur non numérique pour {key}: {value}"
        
        if np.isnan(value) or np.isinf(value):
            return False, f"Valeur invalide pour {key}: {value}"
    
    return True, ""

# Point d'entrée pour tests
if __name__ == "__main__":
    # Test basique du service
    service = create_prediction_service()
    
    # Test de prédiction de pression
    formation_data = {
        'Depth': 2000.0,
        'MudWeight': 1.3,
        'Temperature': 45.0,
        'Porosity': 0.18,
        'Permeability': 150.0
    }
    
    print("Test prédiction pression de formation:")
    result = service.predict_formation_pressure(formation_data)
    print(json.dumps(result, indent=2))
    
    # Test de détection de kick
    kick_data = {
        'FlowRateIn': 360.0,
        'FlowRateOut': 350.0,
        'StandpipePressure': 210.0,
        'CasingPressure': 55.0,
        'MudWeight': 1.3,
        'HookLoad': 185.0,
        'RPM': 125.0,
        'Torque': 18.0,
        'ROP': 12.0
    }
    
    print("\nTest détection de kick:")
    result = service.predict_kick_detection(kick_data)
    print(json.dumps(result, indent=2))
    
    # Test des statistiques
    print("\nStatistiques du service:")
    stats = service.get_service_statistics()
    print(json.dumps(stats, indent=2))