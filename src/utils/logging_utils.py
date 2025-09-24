"""
Logging Utilities for ML Drilling Project
=========================================

Module de gestion des logs avec différents niveaux et formats
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json


class CustomFormatter(logging.Formatter):
    """
    Formatter personnalisé avec des couleurs pour la console
    """
    
    # Codes couleur ANSI
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Vert
        'WARNING': '\033[33m',    # Jaune
        'ERROR': '\033[31m',      # Rouge
        'CRITICAL': '\033[35m'    # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Ajouter la couleur au niveau de log
        if hasattr(record, 'levelname'):
            color = self.COLORS.get(record.levelname, '')
            record.colored_levelname = f"{color}{record.levelname}{self.RESET}"
        
        return super().format(record)


class DrillingLogger:
    """
    Classe principale pour la gestion des logs du projet de forage
    """
    
    def __init__(self, name: str, log_level: str = 'INFO', log_dir: str = 'outputs/logs'):
        """
        Initialise le logger
        
        Args:
            name: Nom du logger
            log_level: Niveau de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Dossier pour les fichiers de log
        """
        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Créer le logger principal
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        
        # Éviter les doublons de handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Configure les handlers pour console et fichier"""
        
        # Handler pour la console avec couleurs
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        
        console_format = CustomFormatter(
            '%(colored_levelname)s - %(name)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        
        # Handler pour fichier principal
        log_file = self.log_dir / f"{self.name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(self.log_level)
        
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        
        # Handler pour les erreurs (fichier séparé)
        error_file = self.log_dir / f"{self.name}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file, maxBytes=5*1024*1024, backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_format)
        
        # Ajouter les handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
    
    def get_logger(self) -> logging.Logger:
        """Retourne l'instance du logger"""
        return self.logger
    
    def log_model_training(self, model_name: str, metrics: Dict[str, float], 
                          duration: float, params: Dict[str, Any]):
        """
        Log spécialisé pour l'entraînement des modèles
        
        Args:
            model_name: Nom du modèle
            metrics: Métriques de performance
            duration: Durée d'entraînement
            params: Paramètres du modèle
        """
        self.logger.info(f"🚀 Début entraînement modèle: {model_name}")
        self.logger.info(f"⚙️  Paramètres: {params}")
        self.logger.info(f"⏱️  Durée d'entraînement: {duration:.2f}s")
        self.logger.info(f"📊 Métriques: {metrics}")
    
    def log_data_processing(self, operation: str, input_shape: tuple, 
                           output_shape: tuple, duration: float):
        """
        Log pour le traitement des données
        
        Args:
            operation: Type d'opération
            input_shape: Forme des données d'entrée
            output_shape: Forme des données de sortie
            duration: Durée de l'opération
        """
        self.logger.info(f"🔄 {operation}")
        self.logger.info(f"📥 Input shape: {input_shape}")
        self.logger.info(f"📤 Output shape: {output_shape}")
        self.logger.info(f"⏱️  Durée: {duration:.2f}s")
    
    def log_prediction(self, model_name: str, input_data: Dict[str, Any], 
                      prediction: Any, confidence: Optional[float] = None):
        """
        Log pour les prédictions
        
        Args:
            model_name: Nom du modèle
            input_data: Données d'entrée (échantillon)
            prediction: Prédiction
            confidence: Niveau de confiance
        """
        self.logger.info(f"🎯 Prédiction - Modèle: {model_name}")
        self.logger.debug(f"📊 Input: {input_data}")
        self.logger.info(f"🔮 Prédiction: {prediction}")
        if confidence:
            self.logger.info(f"🎲 Confiance: {confidence:.3f}")
    
    def log_alert(self, alert_type: str, message: str, severity: str = 'WARNING'):
        """
        Log pour les alertes système
        
        Args:
            alert_type: Type d'alerte
            message: Message d'alerte
            severity: Niveau de sévérité
        """
        level = getattr(logging, severity.upper())
        self.logger.log(level, f"🚨 ALERTE [{alert_type}]: {message}")


class ModelTrainingLogger:
    """
    Logger spécialisé pour le suivi de l'entraînement des modèles
    """
    
    def __init__(self, experiment_name: str, log_dir: str = 'outputs/logs/experiments'):
        """
        Initialise le logger d'expérimentations
        
        Args:
            experiment_name: Nom de l'expérience
            log_dir: Dossier des logs d'expériences
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamp pour identifier l'expérience
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name}_{self.timestamp}"
        
        # Fichier de log JSON pour les métriques
        self.metrics_file = self.log_dir / f"{self.experiment_id}_metrics.json"
        self.metrics_data = {
            'experiment_id': self.experiment_id,
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'models': {},
            'best_model': None
        }
    
    def log_model_experiment(self, model_name: str, hyperparams: Dict[str, Any],
                           train_metrics: Dict[str, float], val_metrics: Dict[str, float],
                           test_metrics: Optional[Dict[str, float]] = None,
                           training_time: Optional[float] = None):
        """
        Log une expérience de modèle complète
        
        Args:
            model_name: Nom du modèle
            hyperparams: Hyperparamètres utilisés
            train_metrics: Métriques sur l'ensemble d'entraînement
            val_metrics: Métriques sur l'ensemble de validation
            test_metrics: Métriques sur l'ensemble de test
            training_time: Temps d'entraînement
        """
        model_data = {
            'hyperparameters': hyperparams,
            'training_metrics': train_metrics,
            'validation_metrics': val_metrics,
            'training_time': training_time,
            'timestamp': datetime.now().isoformat()
        }
        
        if test_metrics:
            model_data['test_metrics'] = test_metrics
        
        self.metrics_data['models'][model_name] = model_data
        
        # Sauvegarder après chaque ajout
        self._save_metrics()
    
    def set_best_model(self, model_name: str, metric: str, value: float):
        """
        Marque un modèle comme le meilleur
        
        Args:
            model_name: Nom du meilleur modèle
            metric: Métrique utilisée pour la sélection
            value: Valeur de la métrique
        """
        self.metrics_data['best_model'] = {
            'model_name': model_name,
            'selection_metric': metric,
            'selection_value': value,
            'selected_at': datetime.now().isoformat()
        }
        self._save_metrics()
    
    def _save_metrics(self):
        """Sauvegarde les métriques en JSON"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_data, f, indent=2)
    
    def generate_experiment_summary(self) -> str:
        """
        Génère un résumé de l'expérience
        
        Returns:
            Résumé formaté
        """
        summary = f"""
📊 RÉSUMÉ DE L'EXPÉRIENCE: {self.experiment_name}
{'='*50}
🆔 ID: {self.experiment_id}
⏰ Démarré le: {self.metrics_data['start_time']}
🔬 Nombre de modèles testés: {len(self.metrics_data['models'])}

🏆 MEILLEUR MODÈLE:
"""
        if self.metrics_data['best_model']:
            best = self.metrics_data['best_model']
            summary += f"""   Nom: {best['model_name']}
   Métrique: {best['selection_metric']} = {best['selection_value']:.4f}
"""
        else:
            summary += "   Aucun modèle sélectionné\n"
        
        summary += "\n📋 MODÈLES TESTÉS:\n"
        for model_name, data in self.metrics_data['models'].items():
            val_score = data.get('validation_metrics', {}).get('r2', 'N/A')
            train_time = data.get('training_time', 'N/A')
            summary += f"   • {model_name}: Val R² = {val_score}, Temps = {train_time}s\n"
        
        return summary


class APILogger:
    """
    Logger spécialisé pour l'API de prédiction
    """
    
    def __init__(self, log_dir: str = 'outputs/logs/api'):
        """
        Initialise le logger API
        
        Args:
            log_dir: Dossier des logs API
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Logger principal
        self.logger = logging.getLogger('drilling_api')
        self.logger.setLevel(logging.INFO)
        
        # Handler pour fichier API
        api_log_file = self.log_dir / 'api_requests.log'
        handler = logging.handlers.RotatingFileHandler(
            api_log_file, maxBytes=20*1024*1024, backupCount=10
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - API - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(handler)
    
    def log_api_request(self, endpoint: str, method: str, client_ip: str,
                       input_data: Dict[str, Any], response_time: float,
                       status_code: int = 200):
        """
        Log une requête API
        
        Args:
            endpoint: Point de terminaison appelé
            method: Méthode HTTP
            client_ip: IP du client
            input_data: Données reçues
            response_time: Temps de réponse
            status_code: Code de statut HTTP
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'endpoint': endpoint,
            'method': method,
            'client_ip': client_ip,
            'input_size': len(str(input_data)),
            'response_time': response_time,
            'status_code': status_code
        }
        
        self.logger.info(json.dumps(log_entry))
    
    def log_prediction_request(self, model_type: str, input_features: Dict[str, Any],
                              prediction_result: Any, processing_time: float):
        """
        Log une requête de prédiction
        
        Args:
            model_type: Type de modèle utilisé
            input_features: Features d'entrée
            prediction_result: Résultat de la prédiction
            processing_time: Temps de traitement
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'prediction',
            'model_type': model_type,
            'input_features_count': len(input_features),
            'prediction': str(prediction_result),
            'processing_time': processing_time
        }
        
        self.logger.info(json.dumps(log_entry))


def setup_project_logging(log_level: str = 'INFO', log_dir: str = 'outputs/logs') -> Dict[str, DrillingLogger]:
    """
    Configure tous les loggers du projet
    
    Args:
        log_level: Niveau de log global
        log_dir: Dossier racine des logs
        
    Returns:
        Dictionnaire avec tous les loggers configurés
    """
    loggers = {}
    
    # Logger principal
    loggers['main'] = DrillingLogger('drilling_main', log_level, log_dir)
    
    # Logger pour les données
    loggers['data'] = DrillingLogger('drilling_data', log_level, log_dir)
    
    # Logger pour les modèles
    loggers['models'] = DrillingLogger('drilling_models', log_level, log_dir)
    
    # Logger pour l'API
    loggers['api'] = APILogger(f"{log_dir}/api")
    
    return loggers


def create_performance_log(operation: str, start_time: datetime, 
                          end_time: datetime, details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Crée un log de performance formaté
    
    Args:
        operation: Nom de l'opération
        start_time: Heure de début
        end_time: Heure de fin
        details: Détails additionnels
        
    Returns:
        Dictionnaire avec les infos de performance
    """
    duration = (end_time - start_time).total_seconds()
    
    return {
        'operation': operation,
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': duration,
        'details': details
    }