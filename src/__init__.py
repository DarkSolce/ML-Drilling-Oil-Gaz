"""
ML-Drilling-Oil-Gas Package
===========================

Package principal pour le machine learning appliqué au forage pétrolier et gazier.
Ce package contient tous les modules nécessaires pour:
- Charger et prétraiter les données de forage
- Entraîner des modèles de prédiction de pression de formation
- Détecter les kicks en temps réel
- Visualiser les données et résultats
- Déployer des modèles via une API REST
"""

__version__ = "1.0.0"
__author__ = "Skander Chebbi"
__email__ = "skanderchbb@gmail.com"
__description__ = "Machine Learning pour opérations de forage pétrolier et gazier"

# Imports principaux pour faciliter l'utilisation
try:
    from .data.data_loader import DrillingDataLoader
    from .data.data_preprocessor import DrillingDataPreprocessor  
    from .data.feature_engineering import FeatureEngineer
    
    from .models.base_model import BaseDrillingModel
    from .models.formation_pressure import FormationPressureModel
    from .models.kick_detection import KickDetectionModel
    
    from .utils.config import load_config, save_config
    from .utils.logging_utils import DrillingLogger, setup_project_logging
    from .utils.metrics import RegressionMetrics, ClassificationMetrics
    
    from .api.prediction_service import PredictionService, EnhancedPredictionService
    
    # Classes principales exportées
    __all__ = [
        # Data
        'DrillingDataLoader',
        'DrillingDataPreprocessor', 
        'FeatureEngineer',
        
        # Models  
        'BaseDrillingModel',
        'FormationPressureModel',
        'KickDetectionModel',
        
        # Utils
        'load_config',
        'save_config', 
        'DrillingLogger',
        'setup_project_logging',
        'RegressionMetrics',
        'ClassificationMetrics',
        
        # API
        'PredictionService',
        'EnhancedPredictionService'
    ]
    
except ImportError as e:
    # En cas d'import manquant, continuer sans erreur
    import warnings
    warnings.warn(f"Certains modules ne sont pas disponibles: {e}")
    __all__ = []

# Configuration des logs par défaut
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Métadonnées du package
__metadata__ = {
    'name': 'ml-drilling-oil-gas',
    'version': __version__,
    'author': __author__,
    'email': __email__,
    'description': __description__,
    'license': 'MIT',
    'url': 'https://github.com/your-org/ML-Drilling-Oil-Gas',
    'keywords': ['machine learning', 'drilling', 'oil and gas', 'petroleum engineering'],
    'python_requires': '>=3.8',
}

def get_version():
    """Retourne la version du package"""
    return __version__

def get_info():
    """Retourne les informations du package"""
    return __metadata__.copy()

def setup():
    """
    Configuration initiale du package
    Appelle cette fonction après l'installation pour configurer l'environnement
    """
    import os
    from pathlib import Path
    
    # Créer les dossiers nécessaires
    directories = [
        'data/raw',
        'data/processed', 
        'data/external',
        'models/formation_pressure',
        'models/kick_detection',
        'outputs/logs',
        'outputs/figures', 
        'outputs/reports'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Structure de dossiers créée")
    
    # Configurer les logs par défaut
    try:
        from .utils.logging_utils import setup_project_logging
        loggers = setup_project_logging()
        print("✅ Logging configuré")
    except ImportError:
        print("⚠️ Module de logging non disponible")
    
    print("🎉 Package ML-Drilling-Oil-Gas configuré avec succès!")

# Fonction pour vérifier l'installation
def check_installation():
    """
    Vérifie que tous les composants nécessaires sont installés
    
    Returns:
        bool: True si l'installation est complète
    """
    required_modules = [
        'numpy', 'pandas', 'scikit-learn', 'matplotlib', 
        'seaborn', 'plotly', 'fastapi', 'streamlit'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Modules manquants: {', '.join(missing_modules)}")
        print("💡 Installez-les avec: pip install -r requirements.txt")
        return False
    else:
        print("✅ Tous les modules requis sont installés")
        return True

# Message d'accueil lors de l'import
def _welcome_message():
    """Message d'accueil"""
    print(f"""
🛢️  ML-Drilling-Oil-Gas v{__version__}
{'='*40}
Machine Learning pour le forage pétrolier

Modules disponibles:
• Data: DrillingDataLoader, DataPreprocessor, FeatureEngineer  
• Models: FormationPressureModel, KickDetectionModel
• API: PredictionService
• Utils: Config, Logging, Metrics

Commandes utiles:
• ml_drilling.setup() - Configuration initiale
• ml_drilling.check_installation() - Vérifier l'installation
""")

# Afficher le message seulement si on importe le package principal
if __name__ != "__main__":
    _welcome_message()