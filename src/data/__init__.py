"""
ML Drilling Operations Package
Advanced Machine Learning for Oil & Gas Drilling Operations

This package provides comprehensive machine learning solutions for:
- Formation pressure prediction
- Kick detection and safety monitoring  
- Real-time drilling optimization
- Production-ready API and dashboard

Main modules:
- data: Data loading, preprocessing, and feature engineering
- models: ML models for formation pressure and kick detection
- api: REST API for production deployment
- visualization: Interactive dashboard and monitoring
- utils: Configuration and utilities

Example usage:
    from src.data.data_loader import DataLoader
    from src.models.formation_pressure import PCRFormationPressure
    
    # Load data
    loader = DataLoader()
    data = loader.load_formation_data()
    
    # Train model
    model = PCRFormationPressure()
    X, y = loader.split_features_target('formation')
    model.train(X, y)
    
    # Make prediction
    prediction = model.predict(X.head(1))
"""

__version__ = "1.0.0"
__author__ = "Skander Chebbi"
__email__ = "skanderchbb@gmail.com"

# Package metadata
__title__ = "ML Drilling Operations"
__description__ = "Advanced Machine Learning for Oil & Gas Drilling Operations"
__url__ = "https://github.com/DarkSolce/ML-Drilling-Oil-Gaz"
__license__ = "MIT"
__copyright__ = "2024 ML Drilling Team"

# Import main classes for easy access
from .data_loader import DataLoader
from .data_preprocessor import DataPreprocessor
from src.models.formation_pressure import (
    PCRFormationPressure,
    XGBoostFormationPressure, 
    EnsembleFormationPressure,
    create_formation_pressure_pipeline
)
from src.models.kick_detection import (
    PCAKickDetection,
    EnsembleKickDetection,
    create_kick_detection_pipeline
)
from src.utils.config import config

# Define what gets imported with "from src import *"
__all__ = [
    # Data classes
    'DataLoader',
    'DataPreprocessor',
    
    # Formation pressure models
    'PCRFormationPressure',
    'XGBoostFormationPressure',
    'EnsembleFormationPressure',
    'create_formation_pressure_pipeline',
    
    # Kick detection models
    'PCAKickDetection', 
    'EnsembleKickDetection',
    'create_kick_detection_pipeline',
    
    # Configuration
    'config',
    
    # Package info
    '__version__',
    '__author__',
    '__email__'
]

# Package initialization
def initialize_package():
    """Initialize package with default configuration"""
    try:
        # Create necessary directories
        config.create_directories()
        
        # Setup logging
        from src.utils.config import setup_logging
        logger = setup_logging()
        logger.info(f"ML Drilling Operations v{__version__} initialized")
        
        return True
    except Exception as e:
        print(f"Warning: Package initialization failed: {e}")
        return False

# Auto-initialize when package is imported
_initialized = initialize_package()

# Compatibility imports for different Python versions
import sys
if sys.version_info >= (3, 8):
    from typing import Literal, Protocol
else:
    try:
        from typing_extensions import Literal, Protocol
    except ImportError:
        # Fallback for older environments
        Literal = None
        Protocol = None

# Version check
def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'scipy',
        'xgboost', 'lightgbm', 'fastapi', 'streamlit'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        raise ImportError(
            f"Missing required packages: {', '.join(missing_packages)}. "
            f"Please install with: pip install {' '.join(missing_packages)}"
        )

# Run dependency check on import
try:
    check_dependencies()
except ImportError as e:
    print(f"Warning: {e}")