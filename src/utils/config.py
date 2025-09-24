"""
Configuration module for ML Drilling project
Centralized configuration management for all project components
"""

import os
from pathlib import Path
from typing import Dict, List, Any
import yaml
from dataclasses import dataclass

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
CONFIG_DIR = ROOT_DIR / "configs"

@dataclass
class DataConfig:
    """Data configuration settings"""
    raw_data_path: str = str(DATA_DIR / "raw")
    processed_data_path: str = str(DATA_DIR / "processed")
    formation_data_file: str = "FormationChangeData.csv"
    kick_data_file: str = "Kick_Detection_Data2.csv"
    
    # Data processing parameters
    train_split: float = 0.7
    validation_split: float = 0.15
    test_split: float = 0.15
    
    # Feature engineering
    window_size: int = 10
    lag_features: int = 5
    moving_average_window: int = 50

@dataclass
class ModelConfig:
    """Model configuration settings"""
    # Formation Pressure Model
    formation_pressure_features: List[str] = None
    formation_n_components: int = 4
    formation_cv_folds: int = 10
    
    # Kick Detection Model  
    kick_detection_features: List[str] = None
    kick_pca_variance: float = 0.9
    kick_detection_threshold: float = 99.99
    
    # XGBoost parameters
    xgb_params: Dict[str, Any] = None
    
    # Random Forest parameters
    rf_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.formation_pressure_features is None:
            self.formation_pressure_features = [
                'WellDepth', 'BTBR', 'WBoPress', 'HLoad', 
                'WoBit', 'RoPen', 'DPPress'
            ]
        
        if self.kick_detection_features is None:
            self.kick_detection_features = [
                'FRate', 'SMSpeed', 'FIn', 'FOut', 'MRFlow', 
                'ActiveGL', 'ATVolume', 'ROPenetration', 'WOBit', 
                'HLoad', 'WBoPressure', 'BTBR', 'ATMPV', 'ATMYP',
                'WellDepth'
            ]
            
        if self.xgb_params is None:
            self.xgb_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
            
        if self.rf_params is None:
            self.rf_params = {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }

@dataclass
class APIConfig:
    """API configuration settings"""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1
    
    # Model endpoints
    formation_pressure_endpoint: str = "/predict/formation_pressure"
    kick_detection_endpoint: str = "/predict/kick_detection"
    health_endpoint: str = "/health"

@dataclass
class DashboardConfig:
    """Dashboard configuration settings"""
    title: str = "Drilling Operations ML Dashboard"
    port: int = 8501
    
    # Chart configurations
    chart_height: int = 400
    chart_width: int = 800
    
    # Real-time update interval (seconds)
    update_interval: int = 5
    
    # Display options
    show_correlation_matrix: bool = True
    show_feature_importance: bool = True
    show_prediction_intervals: bool = True

@dataclass
class LoggingConfig:
    """Logging configuration settings"""
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    log_file: str = str(OUTPUT_DIR / "logs" / "app.log")
    rotation: str = "1 day"
    retention: str = "30 days"

class Config:
    """Main configuration class"""
    
    def __init__(self, config_file: str = None):
        """Initialize configuration with optional YAML file"""
        self.data = DataConfig()
        self.model = ModelConfig()
        self.api = APIConfig()
        self.dashboard = DashboardConfig()
        self.logging = LoggingConfig()
        
        # Load from YAML if provided
        if config_file and os.path.exists(config_file):
            self.load_from_yaml(config_file)
    
    def load_from_yaml(self, config_file: str):
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            # Update configurations
            for section, values in yaml_config.items():
                if hasattr(self, section):
                    config_obj = getattr(self, section)
                    for key, value in values.items():
                        if hasattr(config_obj, key):
                            setattr(config_obj, key, value)
        
        except Exception as e:
            print(f"Warning: Could not load config from {config_file}: {e}")
    
    def save_to_yaml(self, config_file: str):
        """Save current configuration to YAML file"""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'api': self.api.__dict__,
            'dashboard': self.dashboard.__dict__,
            'logging': self.logging.__dict__
        }
        
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def get_data_paths(self) -> Dict[str, Path]:
        """Get all important data paths"""
        return {
            'raw_data': Path(self.data.raw_data_path),
            'processed_data': Path(self.data.processed_data_path),
            'formation_data': Path(self.data.raw_data_path) / self.data.formation_data_file,
            'kick_data': Path(self.data.raw_data_path) / self.data.kick_data_file,
            'models': MODEL_DIR,
            'outputs': OUTPUT_DIR,
            'logs': OUTPUT_DIR / "logs"
        }
    
    def create_directories(self):
        """Create all necessary directories"""
        paths = self.get_data_paths()
        for path in paths.values():
            if path.suffix == '':  # Directory, not file
                path.mkdir(parents=True, exist_ok=True)
            else:  # File, create parent directory
                path.parent.mkdir(parents=True, exist_ok=True)

# Global configuration instance
config = Config()

# Environment-specific configurations
def load_config(env: str = "development") -> Config:
    """Load configuration based on environment"""
    config_file = CONFIG_DIR / f"{env}_config.yaml"
    return Config(str(config_file) if config_file.exists() else None)

# Utility functions
def get_model_path(model_name: str) -> Path:
    """Get path for a specific model"""
    return MODEL_DIR / model_name

def get_output_path(output_type: str) -> Path:
    """Get path for specific output type (figures, reports, logs)"""
    return OUTPUT_DIR / output_type

def setup_logging():
    """Setup logging configuration"""
    from loguru import logger
    
    # Remove default handler
    logger.remove()
    
    # Add file handler
    log_path = Path(config.logging.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        str(log_path),
        format=config.logging.format,
        level=config.logging.level,
        rotation=config.logging.rotation,
        retention=config.logging.retention
    )
    
    # Add console handler
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
        level=config.logging.level
    )
    
    return logger

if __name__ == "__main__":
    # Create directories and save default config
    config.create_directories()
    
    # Save default configuration
    default_config_path = CONFIG_DIR / "default_config.yaml"
    config.save_to_yaml(str(default_config_path))
    
    print("Configuration initialized and directories created!")
    print(f"Default config saved to: {default_config_path}")
    
    # Display current configuration
    print("\nCurrent Configuration:")
    print(f"Data path: {config.data.raw_data_path}")
    print(f"Model features: {len(config.model.formation_pressure_features)} features")
    print(f"API port: {config.api.port}")
    print(f"Dashboard port: {config.dashboard.port}")