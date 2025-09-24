"""
ML Models module for Drilling Operations

This module contains specialized machine learning models for drilling operations:
- Formation pressure prediction models
- Kick detection and anomaly detection models
- Base model classes and utilities

Main components:
- Formation Pressure: PCR, XGBoost, Random Forest, Ensemble models
- Kick Detection: PCA, Isolation Forest, One-Class SVM, Ensemble models
- Base Model: Common functionality for all models

Example usage:
    from src.models import PCRFormationPressure, PCAKickDetection
    
    # Formation pressure prediction
    fp_model = PCRFormationPressure(n_components=4)
    fp_model.train(X_formation, y_formation)
    
    # Kick detection
    kd_model = PCAKickDetection()
    kd_model.train(X_kick, y_kick)
"""

from .base_model import BaseModel, ModelValidator, ModelEnsemble
from .formation_pressure import (
    PCRFormationPressure,
    XGBoostFormationPressure,
    RandomForestFormationPressure,
    PLSFormationPressure,
    EnsembleFormationPressure,
    FormationPressureAnalyzer,
    FormationPressureOptimizer,
    create_formation_pressure_pipeline,
    optimize_formation_pressure_model,
    batch_predict_formation_pressure
)
from .kick_detection import (
    PCAKickDetection,
    IsolationForestKickDetection,
    OneClassSVMKickDetection,
    EnsembleKickDetection,
    KickDetectionAnalyzer,
    optimize_kick_detection_threshold,
    create_kick_detection_pipeline
)

__all__ = [
    # Base classes
    'BaseModel',
    'ModelValidator',
    'ModelEnsemble',
    
    # Formation Pressure Models
    'PCRFormationPressure',
    'XGBoostFormationPressure', 
    'RandomForestFormationPressure',
    'PLSFormationPressure',
    'EnsembleFormationPressure',
    
    # Formation Pressure Utilities
    'FormationPressureAnalyzer',
    'FormationPressureOptimizer',
    'create_formation_pressure_pipeline',
    'optimize_formation_pressure_model',
    'batch_predict_formation_pressure',
    
    # Kick Detection Models
    'PCAKickDetection',
    'IsolationForestKickDetection',
    'OneClassSVMKickDetection', 
    'EnsembleKickDetection',
    
    # Kick Detection Utilities
    'KickDetectionAnalyzer',
    'optimize_kick_detection_threshold',
    'create_kick_detection_pipeline'
]

# Model registry for dynamic loading
MODEL_REGISTRY = {
    'formation_pressure': {
        'pcr': PCRFormationPressure,
        'xgboost': XGBoostFormationPressure,
        'random_forest': RandomForestFormationPressure,
        'pls': PLSFormationPressure,
        'ensemble': EnsembleFormationPressure
    },
    'kick_detection': {
        'pca': PCAKickDetection,
        'isolation_forest': IsolationForestKickDetection,
        'one_class_svm': OneClassSVMKickDetection,
        'ensemble': EnsembleKickDetection
    }
}

def get_model_class(model_category: str, model_type: str):
    """Get model class from registry"""
    if model_category not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model category: {model_category}")
    
    if model_type not in MODEL_REGISTRY[model_category]:
        available_types = list(MODEL_REGISTRY[model_category].keys())
        raise ValueError(f"Unknown model type '{model_type}' for category '{model_category}'. "
                        f"Available types: {available_types}")
    
    return MODEL_REGISTRY[model_category][model_type]

def list_available_models():
    """List all available model types"""
    return {
        category: list(models.keys()) 
        for category, models in MODEL_REGISTRY.items()
    }