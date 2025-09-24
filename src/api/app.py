"""
FastAPI Application for Drilling Operations ML
REST API for formation pressure prediction and kick detection
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import uvicorn
from datetime import datetime, timedelta
import asyncio
import logging
from pathlib import Path
import joblib
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.formation_pressure import (
    PCRFormationPressure, XGBoostFormationPressure, EnsembleFormationPressure,
    FormationPressureAnalyzer
)
from models.kick_detection import (
    PCAKickDetection, EnsembleKickDetection, KickDetectionAnalyzer
)
from data.data_loader import DataLoader
from data.data_preprocessor import DataPreprocessor
from utils.config import config, setup_logging

# Setup logging
logger = setup_logging()

# Initialize FastAPI app
app = FastAPI(
    title="Drilling Operations ML API",
    description="Advanced Machine Learning API for Oil & Gas Drilling Operations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
MODELS = {
    'formation_pressure': {},
    'kick_detection': {}
}

MODEL_METADATA = {
    'formation_pressure': {},
    'kick_detection': {}
}

# Pydantic models for request/response validation
class DrillingParameters(BaseModel):
    """Base drilling parameters model"""
    well_depth: float = Field(..., ge=0, description="Well depth in feet")
    wob: float = Field(..., ge=0, description="Weight on bit in klbs")
    rop: float = Field(..., ge=0, description="Rate of penetration in ft/hr") 
    torque: float = Field(..., ge=0, description="Bit torque in klb-ft")
    standpipe_pressure: float = Field(..., ge=0, description="Standpipe pressure in psi")
    hook_load: float = Field(..., ge=0, description="Hook load in klbs")
    
    @validator('well_depth')
    def validate_depth(cls, v):
        if v > 50000:  # Practical limit
            raise ValueError('Well depth exceeds practical limits')
        return v
    
    @validator('rop')
    def validate_rop(cls, v):
        if v > 500:  # Very high ROP
            raise ValueError('ROP exceeds realistic limits')
        return v

class FormationPressureRequest(DrillingParameters):
    """Formation pressure prediction request"""
    differential_pressure: Optional[float] = Field(180, ge=0, description="Differential pressure in psi")

class FormationPressureResponse(BaseModel):
    """Formation pressure prediction response"""
    predicted_pressure: float = Field(..., description="Predicted formation pressure in psi")
    pressure_gradient: float = Field(..., description="Pressure gradient in psi/ft")
    confidence_score: Optional[float] = Field(None, description="Prediction confidence (0-1)")
    mud_weight_recommendation: float = Field(..., description="Recommended mud weight in ppg")
    pressure_category: str = Field(..., description="Normal/High/Low pressure classification")
    recommendations: List[str] = Field(default_factory=list, description="Operational recommendations")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")

class KickDetectionRequest(DrillingParameters):
    """Kick detection request"""
    flow_in: float = Field(..., ge=0, description="Flow in rate in gpm")
    flow_out: float = Field(..., ge=0, description="Flow out rate in gpm")
    active_pit_volume: float = Field(..., ge=0, description="Active pit volume in bbl")
    mud_return_flow: Optional[float] = Field(None, ge=0, description="Mud return flow in gpm")
    block_speed: Optional[float] = Field(50, ge=0, description="Block speed in ft/min")

class KickDetectionResponse(BaseModel):
    """Kick detection response"""
    kick_detected: bool = Field(..., description="Whether a kick is detected")
    anomaly_score: float = Field(..., description="Anomaly score (0-1, higher = more anomalous)")
    confidence_level: str = Field(..., description="High/Medium/Low confidence")
    risk_level: str = Field(..., description="Critical/High/Medium/Low risk level")
    flow_balance: float = Field(..., description="Flow in - flow out balance")
    emergency_actions: Optional[List[str]] = Field(None, description="Emergency procedures if kick detected")
    monitoring_recommendations: List[str] = Field(default_factory=list, description="Monitoring recommendations")
    timestamp: datetime = Field(default_factory=datetime.now, description="Detection timestamp")

class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str
    model_type: str
    is_loaded: bool
    training_date: Optional[datetime] = None
    performance_metrics: Dict[str, float]
    feature_count: int
    description: str

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    data: List[Dict[str, float]] = Field(..., description="List of drilling parameter dictionaries")
    model_type: str = Field(..., description="formation_pressure or kick_detection")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    loaded_models: Dict[str, List[str]]
    system_info: Dict[str, Any]

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup"""
    logger.info("Starting Drilling Operations ML API...")
    
    # Load default models if available
    await load_default_models()
    
    logger.info("API startup completed")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Drilling Operations ML API...")

# Dependency functions
async def get_formation_model(model_name: str = "default"):
    """Get formation pressure model dependency"""
    if model_name not in MODELS['formation_pressure']:
        raise HTTPException(
            status_code=404,
            detail=f"Formation pressure model '{model_name}' not found. Available models: {list(MODELS['formation_pressure'].keys())}"
        )
    return MODELS['formation_pressure'][model_name]

async def get_kick_model(model_name: str = "default"):
    """Get kick detection model dependency"""
    if model_name not in MODELS['kick_detection']:
        raise HTTPException(
            status_code=404,
            detail=f"Kick detection model '{model_name}' not found. Available models: {list(MODELS['kick_detection'].keys())}"
        )
    return MODELS['kick_detection'][model_name]

# Utility functions
async def load_default_models():
    """Load default models on startup"""
    try:
        # Try to load pre-trained models from disk
        model_dir = Path(config.get_model_path(""))
        
        if model_dir.exists():
            # Look for saved models
            formation_models = list(model_dir.glob("*formation*.pkl"))
            kick_models = list(model_dir.glob("*kick*.pkl"))
            
            # Load most recent models
            if formation_models:
                latest_formation = max(formation_models, key=lambda x: x.stat().st_mtime)
                await load_model_from_file(str(latest_formation), "formation_pressure", "default")
            
            if kick_models:
                latest_kick = max(kick_models, key=lambda x: x.stat().st_mtime)
                await load_model_from_file(str(latest_kick), "kick_detection", "default")
        
        # If no saved models, create and train default models
        if not MODELS['formation_pressure'] and not MODELS['kick_detection']:
            await train_default_models()
            
    except Exception as e:
        logger.warning(f"Could not load default models: {str(e)}")

async def load_model_from_file(filepath: str, model_category: str, model_name: str):
    """Load model from file"""
    try:
        model_data = joblib.load(filepath)
        
        # Reconstruct model object
        if model_category == "formation_pressure":
            if "PCR" in model_data.get('model_name', ''):
                model = PCRFormationPressure()
            elif "XGBoost" in model_data.get('model_name', ''):
                model = XGBoostFormationPressure()
            else:
                model = EnsembleFormationPressure()
        
        elif model_category == "kick_detection":
            if "PCA" in model_data.get('model_name', ''):
                model = PCAKickDetection()
            else:
                model = EnsembleKickDetection()
        
        # Restore model state
        model.model = model_data['model']
        model.feature_columns = model_data['feature_columns']
        model.is_fitted = True
        model.metrics = model_data.get('metrics', {})
        model.training_history = model_data.get('training_history', {})
        
        # Store model
        MODELS[model_category][model_name] = model
        MODEL_METADATA[model_category][model_name] = {
            'loaded_at': datetime.now(),
            'file_path': filepath,
            'metrics': model.metrics
        }
        
        logger.info(f"Loaded {model_category} model '{model_name}' from {filepath}")
        
    except Exception as e:
        logger.error(f"Error loading model from {filepath}: {str(e)}")
        raise

async def train_default_models():
    """Train default models with sample data"""
    logger.info("Training default models with sample data...")
    
    try:
        # Create sample training data
        np.random.seed(42)
        n_samples = 1000
        
        # Formation pressure training data
        formation_data = pd.DataFrame({
            'WellDepth': np.cumsum(np.random.normal(1, 0.1, n_samples)) + 2000,
            'WoBit': np.random.normal(25, 5, n_samples),
            'RoPen': np.random.normal(15, 3, n_samples),
            'BTBR': np.random.normal(120, 15, n_samples),
            'WBoPress': np.random.normal(2000, 200, n_samples),
            'HLoad': np.random.normal(150, 15, n_samples),
            'DPPress': np.random.normal(180, 20, n_samples)
        })
        
        formation_pressure = (
            0.01 * formation_data['WellDepth'] + 
            0.05 * formation_data['WBoPress'] +
            np.random.normal(0, 50, n_samples)
        )
        
        # Train formation pressure model
        formation_model = PCRFormationPressure(n_components=4)
        formation_model.train(formation_data, pd.Series(formation_pressure, name='FPress'))
        
        MODELS['formation_pressure']['default'] = formation_model
        MODEL_METADATA['formation_pressure']['default'] = {
            'loaded_at': datetime.now(),
            'training_data': 'synthetic',
            'metrics': formation_model.metrics
        }
        
        # Kick detection training data  
        kick_data = formation_data.copy()
        kick_data['FIn'] = np.random.normal(300, 30, n_samples)
        kick_data['FOut'] = kick_data['FIn'] + np.random.normal(2, 10, n_samples)
        kick_data['ActiveGL'] = np.random.normal(100, 10, n_samples)
        kick_data['MRFlow'] = kick_data['FOut'] - np.random.normal(5, 2, n_samples)
        kick_data['SMSpeed'] = np.random.normal(50, 10, n_samples)
        
        # Train kick detection model
        kick_model = PCAKickDetection()
        kick_model.train(kick_data)
        
        MODELS['kick_detection']['default'] = kick_model
        MODEL_METADATA['kick_detection']['default'] = {
            'loaded_at': datetime.now(),
            'training_data': 'synthetic', 
            'metrics': kick_model.metrics
        }
        
        logger.info("Default models trained successfully")
        
    except Exception as e:
        logger.error(f"Error training default models: {str(e)}")

def calculate_confidence_score(model, input_data: pd.DataFrame) -> float:
    """Calculate confidence score for predictions"""
    try:
        # Simple confidence based on prediction variance
        predictions = []
        for _ in range(5):  # Multiple predictions with noise
            noisy_data = input_data + np.random.normal(0, 0.01, input_data.shape)
            pred = model.predict(noisy_data)[0]
            predictions.append(pred)
        
        # Lower variance = higher confidence
        variance = np.var(predictions)
        confidence = 1.0 / (1.0 + variance / np.mean(predictions)**2)
        
        return min(confidence, 1.0)
    
    except:
        return 0.5  # Default medium confidence

def generate_formation_recommendations(predicted_pressure: float, 
                                     current_params: Dict[str, float]) -> List[str]:
    """Generate formation pressure recommendations"""
    recommendations = []
    
    # Mud weight recommendations
    required_mud_weight = predicted_pressure * 0.052 + 0.5  # Safety margin
    current_mud_weight = current_params.get('standpipe_pressure', 2000) * 0.052 / 100  # Estimate
    
    if required_mud_weight > current_mud_weight + 0.5:
        recommendations.append(f"Increase mud weight to {required_mud_weight:.1f} ppg for adequate overbalance")
    
    # Pressure gradient analysis
    depth = current_params.get('well_depth', 5000)
    gradient = predicted_pressure / depth
    
    if gradient > 0.6:
        recommendations.append("High pressure zone detected - consider reducing ROP and increasing circulation rate")
    elif gradient < 0.3:
        recommendations.append("Low pressure zone - monitor for lost circulation")
    else:
        recommendations.append("Normal pressure gradient - continue current operations")
    
    # ROP optimization
    current_rop = current_params.get('rop', 15)
    if gradient > 0.5 and current_rop > 20:
        recommendations.append("Consider reducing ROP in high-pressure formation")
    
    return recommendations

def generate_kick_emergency_actions() -> List[str]:
    """Generate emergency actions for kick detection"""
    return [
        "1. STOP DRILLING - Cease all drilling operations immediately",
        "2. PICK UP OFF BOTTOM - Lift drill string off bottom",
        "3. CHECK FLOW - Verify mud returns and flow rates",
        "4. CLOSE BOP if necessary - Activate blowout preventers",
        "5. CIRCULATE KICK OUT - Follow proper kick circulation procedures",
        "6. MONITOR PRESSURES - Watch standpipe and annular pressures",
        "7. WEIGH UP MUD - Calculate and mix heavier mud weight",
        "8. NOTIFY SUPERVISION - Alert drilling supervisor and company representative"
    ]

# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Drilling Operations ML API",
        "version": "1.0.0",
        "description": "Advanced ML API for formation pressure prediction and kick detection",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    import psutil
    
    loaded_models = {
        "formation_pressure": list(MODELS['formation_pressure'].keys()),
        "kick_detection": list(MODELS['kick_detection'].keys())
    }
    
    system_info = {
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "uptime": datetime.now().isoformat()
    }
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        loaded_models=loaded_models,
        system_info=system_info
    )

@app.get("/models", response_model=Dict[str, List[ModelInfo]])
async def list_models():
    """List all available models"""
    models_info = {
        "formation_pressure": [],
        "kick_detection": []
    }
    
    # Formation pressure models
    for name, model in MODELS['formation_pressure'].items():
        metadata = MODEL_METADATA['formation_pressure'].get(name, {})
        
        models_info['formation_pressure'].append(ModelInfo(
            model_name=name,
            model_type=model.model_name,
            is_loaded=model.is_fitted,
            training_date=metadata.get('loaded_at'),
            performance_metrics=model.metrics,
            feature_count=len(model.feature_columns),
            description=f"{model.model_name} for formation pressure prediction"
        ))
    
    # Kick detection models
    for name, model in MODELS['kick_detection'].items():
        metadata = MODEL_METADATA['kick_detection'].get(name, {})
        
        models_info['kick_detection'].append(ModelInfo(
            model_name=name,
            model_type=model.model_name,
            is_loaded=model.is_fitted,
            training_date=metadata.get('loaded_at'),
            performance_metrics=model.metrics,
            feature_count=len(model.feature_columns),
            description=f"{model.model_name} for kick detection"
        ))
    
    return models_info

@app.post("/predict/formation-pressure", response_model=FormationPressureResponse)
async def predict_formation_pressure(
    request: FormationPressureRequest,
    model_name: str = "default",
    model = Depends(get_formation_model)
):
    """Predict formation pressure"""
    try:
        # Prepare input data
        input_data = pd.DataFrame({
            'WellDepth': [request.well_depth],
            'WoBit': [request.wob],
            'RoPen': [request.rop],
            'BTBR': [request.torque],
            'WBoPress': [request.standpipe_pressure],
            'HLoad': [request.hook_load],
            'DPPress': [request.differential_pressure]
        })
        
        # Make prediction
        predicted_pressure = model.predict(input_data)[0]
        
        # Calculate additional metrics
        pressure_gradient = predicted_pressure / request.well_depth
        confidence_score = calculate_confidence_score(model, input_data)
        mud_weight_recommendation = predicted_pressure * 0.052 + 0.5  # ppg with safety margin
        
        # Classify pressure
        normal_gradient = 0.433
        gradient_ratio = pressure_gradient / normal_gradient
        
        if gradient_ratio > 1.2:
            pressure_category = "High"
        elif gradient_ratio < 0.8:
            pressure_category = "Low"
        else:
            pressure_category = "Normal"
        
        # Generate recommendations
        current_params = {
            'well_depth': request.well_depth,
            'rop': request.rop,
            'standpipe_pressure': request.standpipe_pressure
        }
        recommendations = generate_formation_recommendations(predicted_pressure, current_params)
        
        return FormationPressureResponse(
            predicted_pressure=predicted_pressure,
            pressure_gradient=pressure_gradient,
            confidence_score=confidence_score,
            mud_weight_recommendation=mud_weight_recommendation,
            pressure_category=pressure_category,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Formation pressure prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/kick-detection", response_model=KickDetectionResponse)
async def predict_kick_detection(
    request: KickDetectionRequest,
    model_name: str = "default",
    model = Depends(get_kick_model)
):
    """Detect drilling kicks"""
    try:
        # Prepare input data
        input_data = pd.DataFrame({
            'WellDepth': [request.well_depth],
            'WoBit': [request.wob],
            'RoPen': [request.rop],
            'BTBR': [request.torque],
            'WBoPress': [request.standpipe_pressure],
            'HLoad': [request.hook_load],
            'FIn': [request.flow_in],
            'FOut': [request.flow_out],
            'ActiveGL': [request.active_pit_volume],
            'MRFlow': [request.mud_return_flow or request.flow_out - 5],
            'SMSpeed': [request.block_speed]
        })
        
        # Add missing features with default values
        for col in model.feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Make prediction
        kick_prediction = model.predict(input_data)[0]
        
        # Get anomaly score if available
        if hasattr(model, 'get_anomaly_scores'):
            anomaly_score = model.get_anomaly_scores(input_data)[0]
            # Normalize to 0-1 range
            anomaly_score = min(max(anomaly_score / 10.0, 0.0), 1.0)
        else:
            anomaly_score = float(kick_prediction)
        
        # Calculate flow balance
        flow_balance = request.flow_out - request.flow_in
        
        # Determine confidence and risk levels
        if anomaly_score > 0.8:
            confidence_level = "High"
            risk_level = "Critical"
        elif anomaly_score > 0.6:
            confidence_level = "High" if kick_prediction else "Medium"
            risk_level = "High"
        elif anomaly_score > 0.4:
            confidence_level = "Medium"
            risk_level = "Medium"
        else:
            confidence_level = "Low"
            risk_level = "Low"
        
        # Generate emergency actions if kick detected
        emergency_actions = None
        if kick_prediction == 1:
            emergency_actions = generate_kick_emergency_actions()
        
        # Generate monitoring recommendations
        monitoring_recommendations = []
        
        if abs(flow_balance) > 10:
            monitoring_recommendations.append("Monitor flow rates closely - significant imbalance detected")
        
        if anomaly_score > 0.5:
            monitoring_recommendations.append("Increase monitoring frequency - elevated anomaly score")
        
        if risk_level in ["High", "Critical"]:
            monitoring_recommendations.append("Consider stopping drilling operations for assessment")
        
        if not monitoring_recommendations:
            monitoring_recommendations = ["Continue normal drilling operations with standard monitoring"]
        
        return KickDetectionResponse(
            kick_detected=bool(kick_prediction),
            anomaly_score=anomaly_score,
            confidence_level=confidence_level,
            risk_level=risk_level,
            flow_balance=flow_balance,
            emergency_actions=emergency_actions,
            monitoring_recommendations=monitoring_recommendations
        )
        
    except Exception as e:
        logger.error(f"Kick detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/predict/batch", response_model=Dict[str, Any])
async def batch_predictions(request: BatchPredictionRequest):
    """Batch predictions for multiple data points"""
    try:
        results = []
        
        if request.model_type == "formation_pressure":
            if "default" not in MODELS['formation_pressure']:
                raise HTTPException(status_code=404, detail="Formation pressure model not available")
            
            model = MODELS['formation_pressure']['default']
            
            for data_point in request.data:
                try:
                    # Convert to DataFrame
                    input_df = pd.DataFrame([data_point])
                    
                    # Ensure required columns exist
                    required_cols = ['WellDepth', 'WoBit', 'RoPen', 'BTBR', 'WBoPress', 'HLoad']
                    for col in required_cols:
                        if col not in input_df.columns:
                            input_df[col] = 0
                    
                    # Make prediction
                    prediction = model.predict(input_df)[0]
                    
                    results.append({
                        'input': data_point,
                        'prediction': prediction,
                        'status': 'success'
                    })
                    
                except Exception as e:
                    results.append({
                        'input': data_point,
                        'error': str(e),
                        'status': 'error'
                    })
        
        elif request.model_type == "kick_detection":
            if "default" not in MODELS['kick_detection']:
                raise HTTPException(status_code=404, detail="Kick detection model not available")
            
            model = MODELS['kick_detection']['default']
            
            for data_point in request.data:
                try:
                    # Convert to DataFrame
                    input_df = pd.DataFrame([data_point])
                    
                    # Add missing features with defaults
                    for col in model.feature_columns:
                        if col not in input_df.columns:
                            input_df[col] = 0
                    
                    # Make prediction
                    kick_pred = model.predict(input_df)[0]
                    
                    # Get anomaly score if available
                    anomaly_score = 0.5
                    if hasattr(model, 'get_anomaly_scores'):
                        anomaly_score = model.get_anomaly_scores(input_df)[0]
                        anomaly_score = min(max(anomaly_score / 10.0, 0.0), 1.0)
                    
                    results.append({
                        'input': data_point,
                        'kick_detected': bool(kick_pred),
                        'anomaly_score': anomaly_score,
                        'status': 'success'
                    })
                    
                except Exception as e:
                    results.append({
                        'input': data_point,
                        'error': str(e),
                        'status': 'error'
                    })
        else:
            raise HTTPException(status_code=400, detail="Invalid model_type. Use 'formation_pressure' or 'kick_detection'")
        
        # Summary statistics
        successful_predictions = sum(1 for r in results if r['status'] == 'success')
        error_count = len(results) - successful_predictions
        
        return {
            'total_requests': len(request.data),
            'successful_predictions': successful_predictions,
            'errors': error_count,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/models/train/{model_category}")
async def train_model(
    model_category: str,
    background_tasks: BackgroundTasks,
    model_name: str = "custom",
    model_type: str = "default"
):
    """Train new model (background task)"""
    
    if model_category not in ["formation_pressure", "kick_detection"]:
        raise HTTPException(status_code=400, detail="Invalid model category")
    
    # Add training task to background
    background_tasks.add_task(
        train_model_background, 
        model_category, 
        model_name, 
        model_type
    )
    
    return {
        "message": f"Training {model_category} model '{model_name}' started in background",
        "status": "training_started",
        "timestamp": datetime.now().isoformat()
    }

async def train_model_background(model_category: str, model_name: str, model_type: str):
    """Background task for model training"""
    try:
        logger.info(f"Starting background training for {model_category} model '{model_name}'")
        
        # Load training data
        data_loader = DataLoader()
        preprocessor = DataPreprocessor()
        
        if model_category == "formation_pressure":
            # Load and prepare formation data
            raw_data = data_loader.load_formation_data()
            processed_data = preprocessor.prepare_formation_pressure_data(raw_data)
            
            # Split features and target
            X = processed_data.drop('FPress', axis=1)
            y = processed_data['FPress']
            
            # Initialize model based on type
            if model_type == "pcr":
                model = PCRFormationPressure()
            elif model_type == "xgboost":
                model = XGBoostFormationPressure()
            else:
                model = EnsembleFormationPressure()
            
        elif model_category == "kick_detection":
            # Load and prepare kick data
            raw_data = data_loader.load_kick_data()
            processed_data = preprocessor.prepare_kick_detection_data(raw_data)
            
            # Prepare features (exclude target if exists)
            X = processed_data.drop(['ActiveGL'], axis=1, errors='ignore')
            y = None  # Unsupervised learning
            
            # Initialize model based on type
            if model_type == "pca":
                model = PCAKickDetection()
            else:
                model = EnsembleKickDetection()
        
        # Train model
        model.train(X, y)
        
        # Store trained model
        MODELS[model_category][model_name] = model
        MODEL_METADATA[model_category][model_name] = {
            'trained_at': datetime.now(),
            'training_method': 'api_background',
            'metrics': model.metrics
        }
        
        # Save model to disk
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_category}_{model_name}_{timestamp}.pkl"
        model.save_model(filename)
        
        logger.info(f"Background training completed for {model_category} model '{model_name}'")
        
    except Exception as e:
        logger.error(f"Background training failed for {model_category} model '{model_name}': {str(e)}")

@app.delete("/models/{model_category}/{model_name}")
async def delete_model(model_category: str, model_name: str):
    """Delete a loaded model"""
    
    if model_category not in ["formation_pressure", "kick_detection"]:
        raise HTTPException(status_code=400, detail="Invalid model category")
    
    if model_name not in MODELS[model_category]:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    # Remove model from memory
    del MODELS[model_category][model_name]
    if model_name in MODEL_METADATA[model_category]:
        del MODEL_METADATA[model_category][model_name]
    
    logger.info(f"Deleted {model_category} model '{model_name}'")
    
    return {
        "message": f"Model '{model_name}' deleted successfully",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/analytics/formation-pressure")
async def formation_pressure_analytics(
    model_name: str = "default",
    days_back: int = 7
):
    """Get formation pressure prediction analytics"""
    
    if model_name not in MODELS['formation_pressure']:
        raise HTTPException(status_code=404, detail="Formation pressure model not found")
    
    # This would typically query a prediction log database
    # For now, return mock analytics data
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Mock analytics data
    analytics = {
        "model_name": model_name,
        "analysis_period": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "days": days_back
        },
        "prediction_summary": {
            "total_predictions": np.random.randint(100, 1000),
            "average_pressure": round(np.random.normal(2500, 500), 2),
            "pressure_range": {
                "min": round(np.random.normal(1800, 200), 2),
                "max": round(np.random.normal(3500, 300), 2)
            }
        },
        "pressure_categories": {
            "normal": np.random.randint(60, 80),
            "high": np.random.randint(15, 25),
            "low": np.random.randint(5, 15)
        },
        "model_performance": MODELS['formation_pressure'][model_name].metrics,
        "recommendations_given": np.random.randint(20, 50)
    }
    
    return analytics

@app.get("/analytics/kick-detection")
async def kick_detection_analytics(
    model_name: str = "default",
    days_back: int = 7
):
    """Get kick detection analytics"""
    
    if model_name not in MODELS['kick_detection']:
        raise HTTPException(status_code=404, detail="Kick detection model not found")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Mock analytics data
    analytics = {
        "model_name": model_name,
        "analysis_period": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "days": days_back
        },
        "detection_summary": {
            "total_analyses": np.random.randint(500, 2000),
            "kicks_detected": np.random.randint(2, 10),
            "false_positive_rate": round(np.random.uniform(0.02, 0.08), 3),
            "average_anomaly_score": round(np.random.uniform(0.1, 0.3), 3)
        },
        "risk_distribution": {
            "low": np.random.randint(70, 85),
            "medium": np.random.randint(10, 20),
            "high": np.random.randint(3, 8),
            "critical": np.random.randint(0, 2)
        },
        "model_performance": MODELS['kick_detection'][model_name].metrics,
        "emergency_alerts_sent": np.random.randint(1, 5)
    }
    
    return analytics

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

# Development server runner
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
        workers=config.api.workers,
        log_level="info"
    )