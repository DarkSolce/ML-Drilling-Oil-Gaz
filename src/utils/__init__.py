"""
Utilities Module for ML-Drilling-Oil-Gas
========================================

Module utilitaire contenant les fonctions et classes communes
pour la configuration, le logging, les métriques et autres outils.
"""

__version__ = "1.0.0"

# Imports principaux
try:
    from .config import (
        load_config, 
        save_config, 
        merge_configs,
        validate_config,
        get_default_config
    )
    
    from .logging_utils import (
        DrillingLogger,
        ModelTrainingLogger, 
        APILogger,
        setup_project_logging,
        create_performance_log
    )
    
    from .metrics import (
        RegressionMetrics,
        ClassificationMetrics,
        DrillingSpecificMetrics,
        ModelComparisonMetrics,
        MetricsReporter,
        create_metrics_dashboard_data
    )
    
    __all__ = [
        # Configuration
        'load_config',
        'save_config', 
        'merge_configs',
        'validate_config',
        'get_default_config',
        
        # Logging
        'DrillingLogger',
        'ModelTrainingLogger',
        'APILogger', 
        'setup_project_logging',
        'create_performance_log',
        
        # Métriques
        'RegressionMetrics',
        'ClassificationMetrics',
        'DrillingSpecificMetrics', 
        'ModelComparisonMetrics',
        'MetricsReporter',
        'create_metrics_dashboard_data'
    ]
    
except ImportError as e:
    import warnings
    warnings.warn(f"Erreur d'import dans le module utils: {e}")
    __all__ = []

# Configuration par défaut du module
DEFAULT_UTILS_CONFIG = {
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_rotation': True,
        'max_bytes': 10485760,  # 10MB
        'backup_count': 5
    },
    'metrics': {
        'precision_decimals': 4,
        'percentage_decimals': 2,
        'auto_generate_reports': True
    },
    'config': {
        'auto_backup': True,
        'validation_strict': False,
        'merge_strategy': 'update'
    }
}

def get_utils_info():
    """
    Retourne les informations sur le module utils
    
    Returns:
        dict: Informations sur le module
    """
    return {
        'name': 'ML Drilling Utils',
        'version': __version__,
        'description': 'Utilitaires pour le projet ML de forage',
        'components': {
            'config': 'Gestion de la configuration YAML/JSON',
            'logging_utils': 'Système de logging avancé avec rotation',
            'metrics': 'Métriques spécialisées pour le forage'
        },
        'features': [
            'Configuration hiérarchique avec validation',
            'Logging multi-niveaux avec couleurs',
            'Métriques personnalisées pour le domaine',
            'Rapports automatiques',
            'Monitoring des performances'
        ]
    }

def setup_utils(config=None):
    """
    Configuration initiale du module utils
    
    Args:
        config: Configuration personnalisée (optionnel)
    """
    if config is None:
        config = DEFAULT_UTILS_CONFIG
    
    print("🔧 Configuration du module Utils...")
    
    # Setup logging
    try:
        loggers = setup_project_logging(
            log_level=config['logging']['level'],
            log_dir='outputs/logs'
        )
        print(f"✅ Logging configuré - {len(loggers)} loggers créés")
    except Exception as e:
        print(f"⚠️ Erreur configuration logging: {e}")
    
    # Validation de la configuration
    try:
        if validate_config(config):
            print("✅ Configuration validée")
        else:
            print("⚠️ Configuration avec des warnings")
    except Exception as e:
        print(f"⚠️ Erreur validation config: {e}")
    
    print("✅ Module Utils configuré")

# Fonctions utilitaires communes
def format_number(value, precision=4, as_percentage=False):
    """
    Formate un nombre avec la précision spécifiée
    
    Args:
        value: Valeur à formater
        precision: Nombre de décimales
        as_percentage: Si True, formate en pourcentage
        
    Returns:
        str: Nombre formaté
    """
    if value is None or (hasattr(value, '__len__') and len(value) == 0):
        return "N/A"
    
    try:
        if as_percentage:
            return f"{float(value) * 100:.{precision}f}%"
        else:
            return f"{float(value):.{precision}f}"
    except (ValueError, TypeError):
        return str(value)

def safe_divide(numerator, denominator, default=0):
    """
    Division sécurisée qui évite la division par zéro
    
    Args:
        numerator: Numérateur
        denominator: Dénominateur  
        default: Valeur par défaut si division par zéro
        
    Returns:
        Résultat de la division ou valeur par défaut
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default

def ensure_directory(path):
    """
    S'assure qu'un dossier existe, le crée sinon
    
    Args:
        path: Chemin du dossier
        
    Returns:
        pathlib.Path: Chemin du dossier créé
    """
    from pathlib import Path
    
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_timestamp(format_string="%Y%m%d_%H%M%S"):
    """
    Génère un timestamp formaté
    
    Args:
        format_string: Format du timestamp
        
    Returns:
        str: Timestamp formaté
    """
    from datetime import datetime
    return datetime.now().strftime(format_string)

def deep_merge_dict(dict1, dict2):
    """
    Fusion profonde de deux dictionnaires
    
    Args:
        dict1: Premier dictionnaire
        dict2: Second dictionnaire (prioritaire)
        
    Returns:
        dict: Dictionnaire fusionné
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = value
    
    return result

def calculate_percentage_change(old_value, new_value):
    """
    Calcule le pourcentage de changement entre deux valeurs
    
    Args:
        old_value: Ancienne valeur
        new_value: Nouvelle valeur
        
    Returns:
        float: Pourcentage de changement
    """
    if old_value == 0:
        return float('inf') if new_value != 0 else 0
    
    return ((new_value - old_value) / abs(old_value)) * 100

def validate_numeric_range(value, min_val=None, max_val=None, 
                          name="value"):
    """
    Valide qu'une valeur numérique est dans une plage donnée
    
    Args:
        value: Valeur à valider
        min_val: Valeur minimale (optionnel)
        max_val: Valeur maximale (optionnel)
        name: Nom de la valeur pour les messages d'erreur
        
    Returns:
        bool: True si valide
        
    Raises:
        ValueError: Si la valeur est hors limites
    """
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} doit être numérique")
    
    if min_val is not None and value < min_val:
        raise ValueError(f"{name} ({value}) doit être >= {min_val}")
    
    if max_val is not None and value > max_val:
        raise ValueError(f"{name} ({value}) doit être <= {max_val}")
    
    return True

def memory_usage_mb():
    """
    Retourne l'usage mémoire actuel en MB
    
    Returns:
        float: Usage mémoire en MB
    """
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0

def disk_usage_gb(path="."):
    """
    Retourne l'usage disque d'un dossier en GB
    
    Args:
        path: Chemin du dossier
        
    Returns:
        dict: Usage disque (total, used, free en GB)
    """
    try:
        import shutil
        
        total, used, free = shutil.disk_usage(path)
        
        return {
            'total_gb': total / (1024**3),
            'used_gb': used / (1024**3), 
            'free_gb': free / (1024**3),
            'usage_percent': (used / total) * 100
        }
    except:
        return {'total_gb': 0, 'used_gb': 0, 'free_gb': 0, 'usage_percent': 0}

class Timer:
    """
    Gestionnaire de contexte pour mesurer le temps d'exécution
    """
    
    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        print(f"⏱️  {self.name} démarré...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"✅ {self.name} terminé en {duration:.2f} secondes")
    
    @property
    def duration(self):
        """Durée en secondes"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0

class ProgressTracker:
    """
    Tracker de progression pour les opérations longues
    """
    
    def __init__(self, total_items, name="Progress"):
        self.total_items = total_items
        self.name = name
        self.current = 0
        self.start_time = None
    
    def start(self):
        """Démarre le tracking"""
        import time
        self.start_time = time.time()
        print(f"🚀 {self.name} - 0/{self.total_items} (0%)")
    
    def update(self, increment=1):
        """Met à jour la progression"""
        import time
        
        self.current += increment
        percentage = (self.current / self.total_items) * 100
        
        if self.start_time:
            elapsed = time.time() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total_items - self.current) / rate if rate > 0 else 0
            
            print(f"📊 {self.name} - {self.current}/{self.total_items} "
                  f"({percentage:.1f}%) - ETA: {eta:.0f}s")
        else:
            print(f"📊 {self.name} - {self.current}/{self.total_items} ({percentage:.1f}%)")
    
    def finish(self):
        """Termine le tracking"""
        import time
        
        if self.start_time:
            total_time = time.time() - self.start_time
            print(f"✅ {self.name} terminé - {self.total_items} items en {total_time:.1f}s")
        else:
            print(f"✅ {self.name} terminé - {self.total_items} items")

def create_hash(data, algorithm='md5'):
    """
    Crée un hash des données
    
    Args:
        data: Données à hasher (str, dict, ou bytes)
        algorithm: Algorithme de hash ('md5', 'sha256')
        
    Returns:
        str: Hash hexadécimal
    """
    import hashlib
    import json
    
    # Convertir les données en bytes si nécessaire
    if isinstance(data, dict):
        data_str = json.dumps(data, sort_keys=True)
        data_bytes = data_str.encode('utf-8')
    elif isinstance(data, str):
        data_bytes = data.encode('utf-8')
    else:
        data_bytes = data
    
    # Choisir l'algorithme
    if algorithm == 'md5':
        hasher = hashlib.md5()
    elif algorithm == 'sha256':
        hasher = hashlib.sha256()
    else:
        raise ValueError(f"Algorithme non supporté: {algorithm}")
    
    hasher.update(data_bytes)
    return hasher.hexdigest()

def retry_on_failure(max_attempts=3, delay=1, exceptions=(Exception,)):
    """
    Décorateur pour retry automatique en cas d'échec
    
    Args:
        max_attempts: Nombre maximum de tentatives
        delay: Délai entre les tentatives (secondes)
        exceptions: Types d'exceptions à retry
        
    Returns:
        Décorateur
    """
    def decorator(func):
        import functools
        import time
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        print(f"⚠️ Tentative {attempt + 1} échouée: {e}. Retry dans {delay}s...")
                        time.sleep(delay)
                    else:
                        print(f"❌ Échec après {max_attempts} tentatives")
            
            raise last_exception
        
        return wrapper
    return decorator

def compress_data(data, method='gzip'):
    """
    Compresse des données
    
    Args:
        data: Données à comprimer (str ou bytes)
        method: Méthode de compression ('gzip', 'bz2', 'lzma')
        
    Returns:
        bytes: Données compressées
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    if method == 'gzip':
        import gzip
        return gzip.compress(data)
    elif method == 'bz2':
        import bz2
        return bz2.compress(data)
    elif method == 'lzma':
        import lzma
        return lzma.compress(data)
    else:
        raise ValueError(f"Méthode de compression non supportée: {method}")

def decompress_data(compressed_data, method='gzip', encoding='utf-8'):
    """
    Décompresse des données
    
    Args:
        compressed_data: Données compressées
        method: Méthode de décompression
        encoding: Encodage pour retourner une string (None pour bytes)
        
    Returns:
        str ou bytes: Données décompressées
    """
    if method == 'gzip':
        import gzip
        data = gzip.decompress(compressed_data)
    elif method == 'bz2':
        import bz2
        data = bz2.decompress(compressed_data)
    elif method == 'lzma':
        import lzma
        data = lzma.decompress(compressed_data)
    else:
        raise ValueError(f"Méthode de décompression non supportée: {method}")
    
    if encoding:
        return data.decode(encoding)
    return data

def get_system_info():
    """
    Retourne les informations système
    
    Returns:
        dict: Informations système
    """
    import platform
    import sys
    
    try:
        import psutil
        
        info = {
            'platform': platform.platform(),
            'python_version': sys.version,
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'disk_usage': disk_usage_gb(),
            'memory_usage_mb': memory_usage_mb()
        }
    except ImportError:
        info = {
            'platform': platform.platform(),
            'python_version': sys.version,
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'cpu_count': 'N/A',
            'memory_gb': 'N/A',
            'disk_usage': 'N/A',
            'memory_usage_mb': 'N/A'
        }
    
    return info

# Constantes utiles
DRILLING_CONSTANTS = {
    'PRESSURE_GRADIENTS': {
        'freshwater_psi_ft': 0.433,
        'seawater_psi_ft': 0.445,
        'normal_pore_pressure_psi_ft': 0.465
    },
    'UNIT_CONVERSIONS': {
        'bar_to_psi': 14.5038,
        'psi_to_bar': 0.068948,
        'ft_to_m': 0.3048,
        'm_to_ft': 3.28084,
        'gpm_to_lpm': 3.78541,
        'lpm_to_gpm': 0.264172
    },
    'TYPICAL_RANGES': {
        'mud_weight_ppg': (8.0, 20.0),
        'rop_ft_hr': (5.0, 200.0),
        'wob_klbs': (5.0, 80.0),
        'rpm': (50.0, 250.0),
        'flow_rate_gpm': (100.0, 800.0)
    }
}

def convert_units(value, from_unit, to_unit):
    """
    Convertit une valeur d'une unité à une autre
    
    Args:
        value: Valeur à convertir
        from_unit: Unité source
        to_unit: Unité cible
        
    Returns:
        float: Valeur convertie
    """
    conversions = DRILLING_CONSTANTS['UNIT_CONVERSIONS']
    
    conversion_key = f"{from_unit}_to_{to_unit}"
    
    if conversion_key in conversions:
        return value * conversions[conversion_key]
    else:
        # Essayer la conversion inverse
        reverse_key = f"{to_unit}_to_{from_unit}"
        if reverse_key in conversions:
            return value / conversions[reverse_key]
        else:
            raise ValueError(f"Conversion {from_unit} -> {to_unit} non supportée")

def validate_drilling_parameter(parameter, value):
    """
    Valide qu'un paramètre de forage est dans une plage réaliste
    
    Args:
        parameter: Nom du paramètre
        value: Valeur à valider
        
    Returns:
        tuple: (is_valid, message)
    """
    ranges = DRILLING_CONSTANTS['TYPICAL_RANGES']
    
    # Mapping des paramètres
    param_mapping = {
        'mudweight': 'mud_weight_ppg',
        'mud_weight': 'mud_weight_ppg',
        'rop': 'rop_ft_hr',
        'rate_of_penetration': 'rop_ft_hr',
        'wob': 'wob_klbs',
        'weight_on_bit': 'wob_klbs',
        'rpm': 'rpm',
        'flowrate': 'flow_rate_gpm',
        'flow_rate': 'flow_rate_gpm'
    }
    
    param_key = param_mapping.get(parameter.lower())
    
    if not param_key:
        return True, f"Paramètre {parameter} non reconnu (validation ignorée)"
    
    min_val, max_val = ranges[param_key]
    
    if value < min_val:
        return False, f"{parameter} ({value}) en dessous de la plage normale ({min_val}-{max_val})"
    elif value > max_val:
        return False, f"{parameter} ({value}) au dessus de la plage normale ({min_val}-{max_val})"
    else:
        return True, f"{parameter} dans la plage normale"

# Classes d'exception personnalisées
class DrillingDataError(Exception):
    """Exception pour les erreurs de données de forage"""
    pass

class ConfigurationError(Exception):
    """Exception pour les erreurs de configuration"""
    pass

class ModelError(Exception):
    """Exception pour les erreurs de modèle"""
    pass

class ValidationError(Exception):
    """Exception pour les erreurs de validation"""
    pass

# Message d'information du module
def _utils_info():
    """Affiche les informations du module utils"""
    info = get_utils_info()
    print(f"""
🔧 {info['name']} v{info['version']}
{info['description']}

📦 Composants:""")
    for comp, desc in info['components'].items():
        print(f"  • {comp:<15} - {desc}")
    
    print(f"""
✨ Fonctionnalités principales:""")
    for feature in info['features']:
        print(f"  • {feature}")
    
    print(f"""
🛠️  Utilitaires disponibles:
  • Timer()                       - Mesure du temps d'exécution
  • ProgressTracker()             - Suivi de progression
  • retry_on_failure()            - Retry automatique
  • convert_units()               - Conversion d'unités
  • validate_drilling_parameter() - Validation paramètres forage
""")

# Afficher l'info lors de l'import (seulement si pas dans __main__)
if __name__ != "__main__":
    try:
        _utils_info()
    except:
        pass  # Silencieux si erreur d'affichage