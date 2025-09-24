#!/usr/bin/env python3
"""
ML Drilling Operations - Main Pipeline Runner
Orchestrates data loading, preprocessing, training, and deployment
"""

import argparse
import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import yaml
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.data.feature_engineering import DrillingFeatureEngineer
from src.models.formation_pressure import (
    PCRFormationPressure, XGBoostFormationPressure, EnsembleFormationPressure,
    FormationPressureAnalyzer, create_formation_pressure_pipeline
)
from src.models.kick_detection import (
    PCAKickDetection, EnsembleKickDetection, KickDetectionAnalyzer,
    create_kick_detection_pipeline
)
from src.utils.config import config, setup_logging

# Setup logging
logger = setup_logging()

class MLDrillingPipeline:
    """Main pipeline orchestrator for ML drilling operations"""
    
    def __init__(self, config_file: str = None):
        """Initialize pipeline with configuration"""
        self.config_file = config_file
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = DrillingFeatureEngineer()
        
        # Results storage
        self.results = {
            'formation_pressure': {},
            'kick_detection': {},
            'pipeline_info': {
                'start_time': datetime.now().isoformat(),
                'config_file': config_file
            }
        }
        
        # Create output directories
        config.create_directories()
    
    def load_and_prepare_data(self, data_type: str = 'both') -> dict:
        """Load and prepare data for training"""
        logger.info(f"Loading and preparing data: {data_type}")
        
        data = {}
        
        try:
            if data_type in ['both', 'formation']:
                # Load formation data
                formation_data = self.data_loader.load_formation_data()
                logger.info(f"Loaded formation data: {formation_data.shape}")
                
                # Preprocess
                processed_formation = self.preprocessor.prepare_formation_pressure_data(formation_data)
                logger.info(f"Processed formation data: {processed_formation.shape}")
                
                # Feature engineering
                enhanced_formation = self.feature_engineer.create_drilling_features_pipeline(
                    processed_formation, target_type='formation'
                )
                logger.info(f"Enhanced formation data: {enhanced_formation.shape}")
                
                data['formation'] = {
                    'raw': formation_data,
                    'processed': processed_formation,
                    'enhanced': enhanced_formation
                }
            
            if data_type in ['both', 'kick']:
                # Load kick data
                kick_data = self.data_loader.load_kick_data()
                logger.info(f"Loaded kick data: {kick_data.shape}")
                
                # Preprocess
                processed_kick = self.preprocessor.prepare_kick_detection_data(kick_data)
                logger.info(f"Processed kick data: {processed_kick.shape}")
                
                # Feature engineering
                enhanced_kick = self.feature_engineer.create_drilling_features_pipeline(
                    processed_kick, target_type='kick'
                )
                logger.info(f"Enhanced kick data: {enhanced_kick.shape}")
                
                data['kick'] = {
                    'raw': kick_data,
                    'processed': processed_kick,
                    'enhanced': enhanced_kick
                }
            
            return data
            
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise
    
    def train_formation_pressure_models(self, data: dict) -> dict:
        """Train formation pressure prediction models"""
        logger.info("Training formation pressure models...")
        
        results = {}
        
        try:
            # Get enhanced data
            enhanced_data = data['formation']['enhanced']
            
            # Split features and target
            X = enhanced_data.drop('FPress', axis=1, errors='ignore')
            y = enhanced_data['FPress'] if 'FPress' in enhanced_data.columns else None
            
            if y is None:
                raise ValueError("Formation pressure target (FPress) not found in data")
            
            logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
            
            # Models to train
            models_to_train = {
                'PCR': PCRFormationPressure(n_components=4),
                'XGBoost': XGBoostFormationPressure(),
                'Ensemble': EnsembleFormationPressure(['pcr', 'xgboost'])
            }
            
            # Train each model
            for name, model in models_to_train.items():
                logger.info(f"Training {name} model...")
                
                try:
                    # Train model
                    metrics = model.train(X, y, validation_split=0.2)
                    
                    # Save model
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_path = config.get_model_path(f"formation_pressure_{name.lower()}_{timestamp}.pkl")
                    saved_path = model.save_model(str(model_path))
                    
                    # Store results
                    results[name] = {
                        'model': model,
                        'metrics': metrics,
                        'saved_path': saved_path,
                        'training_completed': datetime.now().isoformat()
                    }
                    
                    logger.info(f"{name} model training completed. Val R²: {metrics['val_r2']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training {name} model: {str(e)}")
                    results[name] = {'error': str(e)}
            
            # Model comparison
            if len(results) > 1:
                try:
                    # Prepare test data
                    X_test = X.tail(100)
                    y_test = y.tail(100)
                    
                    # Compare models
                    trained_models = {name: res['model'] for name, res in results.items() if 'model' in res}
                    comparison = FormationPressureAnalyzer.compare_models(trained_models, X_test, y_test)
                    
                    results['comparison'] = comparison.to_dict('records')
                    logger.info("Model comparison completed")
                    
                except Exception as e:
                    logger.warning(f"Model comparison failed: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in formation pressure model training: {str(e)}")
            return {'error': str(e)}
    
    def train_kick_detection_models(self, data: dict) -> dict:
        """Train kick detection models"""
        logger.info("Training kick detection models...")
        
        results = {}
        
        try:
            # Get enhanced data
            enhanced_data = data['kick']['enhanced']
            
            # Prepare features (exclude target columns)
            target_cols = ['ActiveGL']  # Main target for kick detection
            feature_cols = [col for col in enhanced_data.columns if col not in target_cols]
            X = enhanced_data[feature_cols]
            
            # Create synthetic labels based on pit volume anomalies for training
            if 'ActiveGL' in enhanced_data.columns:
                pit_volume_changes = enhanced_data['ActiveGL'].diff().abs()
                threshold_99 = pit_volume_changes.quantile(0.99)
                y = (pit_volume_changes > threshold_99).astype(int)
            else:
                y = None
            
            logger.info(f"Training data shape: X={X.shape}")
            if y is not None:
                logger.info(f"Synthetic kick labels created: {y.sum()} kicks out of {len(y)} samples")
            
            # Models to train
            models_to_train = {
                'PCA': PCAKickDetection(variance_threshold=0.9),
                'Ensemble': EnsembleKickDetection(['pca', 'isolation_forest'])
            }
            
            # Train each model
            for name, model in models_to_train.items():
                logger.info(f"Training {name} kick detection model...")
                
                try:
                    # Train model
                    metrics = model.train(X, y, validation_split=0.2)
                    
                    # Save model
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_path = config.get_model_path(f"kick_detection_{name.lower()}_{timestamp}.pkl")
                    saved_path = model.save_model(str(model_path))
                    
                    # Store results
                    results[name] = {
                        'model': model,
                        'metrics': metrics,
                        'saved_path': saved_path,
                        'training_completed': datetime.now().isoformat()
                    }
                    
                    anomaly_rate = metrics.get('val_anomaly_rate', 0)
                    logger.info(f"{name} model training completed. Anomaly rate: {anomaly_rate:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training {name} model: {str(e)}")
                    results[name] = {'error': str(e)}
            
            # Model comparison
            if len(results) > 1:
                try:
                    # Prepare test data
                    X_test = X.tail(200)
                    y_test = y.tail(200) if y is not None else None
                    
                    # Compare models
                    trained_models = {name: res['model'] for name, res in results.items() if 'model' in res}
                    comparison = KickDetectionAnalyzer.compare_detection_models(trained_models, X_test, y_test)
                    
                    results['comparison'] = comparison.to_dict('records')
                    logger.info("Kick detection model comparison completed")
                    
                except Exception as e:
                    logger.warning(f"Kick detection model comparison failed: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in kick detection model training: {str(e)}")
            return {'error': str(e)}
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive pipeline report"""
        logger.info("Generating comprehensive pipeline report...")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ML DRILLING OPERATIONS - COMPREHENSIVE PIPELINE REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Pipeline started: {self.results['pipeline_info']['start_time']}")
        report_lines.append("")
        
        # Formation Pressure Results
        if 'formation_pressure' in self.results and self.results['formation_pressure']:
            report_lines.append("📊 FORMATION PRESSURE PREDICTION RESULTS")
            report_lines.append("-" * 60)
            
            fp_results = self.results['formation_pressure']
            
            # Model performance summary
            for model_name, model_results in fp_results.items():
                if model_name == 'comparison':
                    continue
                    
                if 'error' in model_results:
                    report_lines.append(f"\n❌ {model_name} Model: FAILED")
                    report_lines.append(f"   Error: {model_results['error']}")
                else:
                    metrics = model_results['metrics']
                    report_lines.append(f"\n✅ {model_name} Model: SUCCESS")
                    report_lines.append(f"   R² Score: {metrics.get('val_r2', 'N/A'):.4f}")
                    report_lines.append(f"   RMSE: {metrics.get('val_rmse', 'N/A'):.2f} psi")
                    report_lines.append(f"   MAE: {metrics.get('val_mae', 'N/A'):.2f} psi")
                    
                    if 'explained_variance_ratio' in metrics:
                        report_lines.append(f"   Explained Variance: {metrics['explained_variance_ratio']:.3f}")
                    
                    report_lines.append(f"   Model saved: {model_results['saved_path']}")
            
            # Best model recommendation
            if 'comparison' in fp_results:
                comparison_data = fp_results['comparison']
                if comparison_data:
                    best_model = max(comparison_data, key=lambda x: x.get('R²', 0))
                    report_lines.append(f"\n🏆 RECOMMENDED MODEL: {best_model['Model']}")
                    report_lines.append(f"   Best R² Score: {best_model.get('R²', 0):.4f}")
                    report_lines.append(f"   RMSE: {best_model.get('RMSE', 0):.2f} psi")
        
        # Kick Detection Results
        if 'kick_detection' in self.results and self.results['kick_detection']:
            report_lines.append("\n\n🚨 KICK DETECTION RESULTS")
            report_lines.append("-" * 60)
            
            kd_results = self.results['kick_detection']
            
            # Model performance summary
            for model_name, model_results in kd_results.items():
                if model_name == 'comparison':
                    continue
                    
                if 'error' in model_results:
                    report_lines.append(f"\n❌ {model_name} Model: FAILED")
                    report_lines.append(f"   Error: {model_results['error']}")
                else:
                    metrics = model_results['metrics']
                    report_lines.append(f"\n✅ {model_name} Model: SUCCESS")
                    report_lines.append(f"   Anomaly Detection Rate: {metrics.get('val_anomaly_rate', 'N/A'):.4f}")
                    
                    if 'val_accuracy' in metrics:
                        report_lines.append(f"   Accuracy: {metrics['val_accuracy']:.4f}")
                        report_lines.append(f"   Precision: {metrics.get('val_precision', 'N/A'):.4f}")
                        report_lines.append(f"   Recall: {metrics.get('val_recall', 'N/A'):.4f}")
                    
                    if 'spe_threshold' in metrics:
                        report_lines.append(f"   SPE Threshold: {metrics['spe_threshold']:.3f}")
                    
                    report_lines.append(f"   Model saved: {model_results['saved_path']}")
            
            # Best kick detection model
            if 'comparison' in kd_results:
                comparison_data = kd_results['comparison']
                if comparison_data:
                    # Prioritize recall for safety
                    best_model = max(comparison_data, key=lambda x: x.get('Recall', x.get('Safety_Score', 0)))
                    report_lines.append(f"\n🏆 RECOMMENDED KICK DETECTION MODEL: {best_model['Model']}")
                    report_lines.append(f"   Safety Score (Recall): {best_model.get('Recall', best_model.get('Safety_Score', 0)):.4f}")
                    if 'Precision' in best_model:
                        report_lines.append(f"   Precision: {best_model['Precision']:.4f}")
        
        # Deployment Recommendations
        report_lines.append("\n\n🚀 DEPLOYMENT RECOMMENDATIONS")
        report_lines.append("-" * 60)
        
        report_lines.append("\n1. FORMATION PRESSURE PREDICTION:")
        if 'formation_pressure' in self.results:
            fp_results = self.results['formation_pressure']
            successful_models = [name for name, res in fp_results.items() if 'model' in res]
            
            if successful_models:
                report_lines.append(f"   ✅ Deploy best performing model for real-time predictions")
                report_lines.append(f"   ✅ Use for mud weight optimization and well planning")
                report_lines.append(f"   ✅ Update predictions every 30-60 seconds during drilling")
                report_lines.append(f"   ⚠️  Retrain monthly with new drilling data")
            else:
                report_lines.append(f"   ❌ No successful models - review data quality and features")
        
        report_lines.append("\n2. KICK DETECTION:")
        if 'kick_detection' in self.results:
            kd_results = self.results['kick_detection']
            successful_models = [name for name, res in kd_results.items() if 'model' in res]
            
            if successful_models:
                report_lines.append(f"   ✅ Deploy for continuous safety monitoring")
                report_lines.append(f"   ✅ Set up real-time alerts for drilling crew")
                report_lines.append(f"   ✅ Prioritize high recall (catch all kicks) over low false positives")
                report_lines.append(f"   ⚠️  Validate with drilling engineers before full deployment")
            else:
                report_lines.append(f"   ❌ No successful models - review anomaly detection approach")
        
        # Technical Specifications
        report_lines.append("\n\n⚙️  TECHNICAL SPECIFICATIONS")
        report_lines.append("-" * 60)
        
        report_lines.append("\nAPI Deployment:")
        report_lines.append(f"   • FastAPI server on port {config.api.port}")
        report_lines.append(f"   • Health check endpoint: /health")
        report_lines.append(f"   • Formation pressure: POST /predict/formation-pressure")
        report_lines.append(f"   • Kick detection: POST /predict/kick-detection")
        
        report_lines.append("\nDashboard:")
        report_lines.append(f"   • Streamlit dashboard on port {config.dashboard.port}")
        report_lines.append(f"   • Real-time monitoring and visualization")
        report_lines.append(f"   • Model performance comparison")
        
        report_lines.append("\nData Requirements:")
        report_lines.append(f"   • Sampling rate: 1-10 Hz recommended")
        report_lines.append(f"   • Critical parameters: WOB, ROP, Torque, Pressures, Flow rates")
        report_lines.append(f"   • Data quality: <5% missing values preferred")
        
        # Usage Instructions
        report_lines.append("\n\n📋 USAGE INSTRUCTIONS")
        report_lines.append("-" * 60)
        
        report_lines.append("\n1. Start API Server:")
        report_lines.append("   python -m src.api.app")
        report_lines.append("   # or")
        report_lines.append("   uvicorn src.api.app:app --host 0.0.0.0 --port 8000")
        
        report_lines.append("\n2. Start Dashboard:")
        report_lines.append("   streamlit run src/visualization/dashboard.py")
        
        report_lines.append("\n3. Make Predictions:")
        report_lines.append("   curl -X POST http://localhost:8000/predict/formation-pressure \\")
        report_lines.append("   -H 'Content-Type: application/json' \\")
        report_lines.append("   -d '{\"well_depth\": 5000, \"wob\": 25, \"rop\": 15, ...}'")
        
        report_lines.append("\n4. Retrain Models:")
        report_lines.append("   python run_pipeline.py --mode train --data-type both")
        
        # Footer
        report_lines.append("\n" + "=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def save_results(self) -> str:
        """Save pipeline results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        results_file = config.get_output_path("reports") / f"pipeline_results_{timestamp}.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare serializable results
        serializable_results = {}
        for category, results in self.results.items():
            if category == 'pipeline_info':
                serializable_results[category] = results
                continue
                
            serializable_results[category] = {}
            for model_name, model_results in results.items():
                if model_name in ['comparison']:
                    serializable_results[category][model_name] = model_results
                elif 'error' in model_results:
                    serializable_results[category][model_name] = {'error': model_results['error']}
                else:
                    serializable_results[category][model_name] = {
                        'metrics': model_results.get('metrics', {}),
                        'saved_path': model_results.get('saved_path', ''),
                        'training_completed': model_results.get('training_completed', '')
                    }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Save comprehensive report
        report = self.generate_comprehensive_report()
        report_file = config.get_output_path("reports") / f"pipeline_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Report saved to: {report_file}")
        
        return str(report_file)
    
    def run_full_pipeline(self, data_type: str = 'both') -> dict:
        """Run the complete ML pipeline"""
        logger.info("Starting full ML drilling pipeline...")
        
        try:
            # 1. Load and prepare data
            data = self.load_and_prepare_data(data_type)
            
            # 2. Train formation pressure models
            if data_type in ['both', 'formation'] and 'formation' in data:
                self.results['formation_pressure'] = self.train_formation_pressure_models(data)
            
            # 3. Train kick detection models
            if data_type in ['both', 'kick'] and 'kick' in data:
                self.results['kick_detection'] = self.train_kick_detection_models(data)
            
            # 4. Save results and generate report
            self.results['pipeline_info']['end_time'] = datetime.now().isoformat()
            report_file = self.save_results()
            
            logger.info("Full pipeline completed successfully!")
            logger.info(f"Comprehensive report available at: {report_file}")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            self.results['pipeline_info']['error'] = str(e)
            self.results['pipeline_info']['end_time'] = datetime.now().isoformat()
            return self.results

def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(
        description="ML Drilling Operations Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python run_pipeline.py --mode full --data-type both
  
  # Train only formation pressure models
  python run_pipeline.py --mode train --data-type formation
  
  # Train only kick detection models  
  python run_pipeline.py --mode train --data-type kick
  
  # Start API server
  python run_pipeline.py --mode api
  
  # Start dashboard
  python run_pipeline.py --mode dashboard
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['full', 'train', 'api', 'dashboard', 'report'],
        default='full',
        help='Pipeline mode to run'
    )
    
    parser.add_argument(
        '--data-type',
        choices=['both', 'formation', 'kick'],
        default='both', 
        help='Type of models to train'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for results and models'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Update output directory if specified
    if args.output_dir:
        config.data.processed_data_path = str(Path(args.output_dir) / "processed")
        # Update other paths as needed
    
    try:
        # Initialize pipeline
        pipeline = MLDrillingPipeline(config_file=args.config)
        
        if args.mode == 'full':
            # Run complete pipeline
            results = pipeline.run_full_pipeline(data_type=args.data_type)
            
            # Print summary
            print("\n" + "="*60)
            print("ML DRILLING PIPELINE SUMMARY")
            print("="*60)
            
            if 'formation_pressure' in results:
                fp_success = sum(1 for r in results['formation_pressure'].values() if 'model' in r)
                print(f"Formation Pressure Models: {fp_success} trained successfully")
            
            if 'kick_detection' in results:
                kd_success = sum(1 for r in results['kick_detection'].values() if 'model' in r)
                print(f"Kick Detection Models: {kd_success} trained successfully")
            
            print(f"Results saved to: {config.get_output_path('reports')}")
            
        elif args.mode == 'train':
            # Train models only
            data = pipeline.load_and_prepare_data(args.data_type)
            
            if args.data_type in ['both', 'formation'] and 'formation' in data:
                results = pipeline.train_formation_pressure_models(data)
                print(f"Formation pressure models trained: {len([r for r in results.values() if 'model' in r])}")
            
            if args.data_type in ['both', 'kick'] and 'kick' in data:
                results = pipeline.train_kick_detection_models(data)
                print(f"Kick detection models trained: {len([r for r in results.values() if 'model' in r])}")
        
        elif args.mode == 'api':
            # Start API server
            print("Starting ML Drilling API server...")
            print(f"API will be available at: http://localhost:{config.api.port}")
            print("Press Ctrl+C to stop")
            
            import uvicorn
            from src.api.app import app
            
            uvicorn.run(
                app,
                host=config.api.host,
                port=config.api.port,
                reload=config.api.reload
            )
        
        elif args.mode == 'dashboard':
            # Start Streamlit dashboard
            print("Starting ML Drilling Dashboard...")
            print(f"Dashboard will be available at: http://localhost:{config.dashboard.port}")
            print("Press Ctrl+C to stop")
            
            import subprocess
            import sys
            
            dashboard_path = Path(__file__).parent / "src" / "visualization" / "dashboard.py"
            
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", 
                str(dashboard_path),
                "--server.port", str(config.dashboard.port),
                "--server.address", "0.0.0.0"
            ])
        
        elif args.mode == 'report':
            # Generate report only
            pipeline.results = {
                'pipeline_info': {'start_time': datetime.now().isoformat()},
                'formation_pressure': {},
                'kick_detection': {}
            }
            
            report = pipeline.generate_comprehensive_report()
            print(report)
        
        print("\n✅ Pipeline completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()