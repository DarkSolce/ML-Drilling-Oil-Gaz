"""
Data Preprocessing Module for ML Drilling Project
Handles cleaning, transformation, and preparation of drilling data for ML models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from scipy.signal import savgol_filter
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import config
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Main data preprocessing class"""
    
    def __init__(self):
        """Initialize preprocessor"""
        self.scalers = {}
        self.imputers = {}
        self.feature_columns = []
        self.preprocessing_steps = []
        
    def clean_data(self, df: pd.DataFrame, 
                   remove_outliers: bool = True,
                   outlier_method: str = 'iqr',
                   outlier_factor: float = 1.5) -> pd.DataFrame:
        """Clean data by removing outliers and handling anomalies"""
        
        df_clean = df.copy()
        
        # Remove duplicate rows
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed_duplicates = initial_rows - len(df_clean)
        
        if removed_duplicates > 0:
            logger.info(f"Removed {removed_duplicates} duplicate rows")
        
        # Handle outliers
        if remove_outliers:
            df_clean = self._remove_outliers(df_clean, method=outlier_method, 
                                           factor=outlier_factor)
        
        # Remove constant columns
        constant_columns = []
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            if df_clean[col].nunique() <= 1:
                constant_columns.append(col)
        
        if constant_columns:
            df_clean = df_clean.drop(columns=constant_columns)
            logger.info(f"Removed constant columns: {constant_columns}")
        
        self.preprocessing_steps.append("Data cleaning completed")
        return df_clean
    
    def _remove_outliers(self, df: pd.DataFrame, method: str = 'iqr', 
                        factor: float = 1.5) -> pd.DataFrame:
        """Remove outliers using specified method"""
        df_no_outliers = df.copy()
        outliers_removed = 0
        
        for col in df.select_dtypes(include=[np.number]).columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                outlier_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
                outliers_in_col = len(df) - outlier_mask.sum()
                outliers_removed += outliers_in_col
                
                df_no_outliers = df_no_outliers[outlier_mask]
                
            elif method == 'zscore':
                z_scores = np.abs(zscore(df[col].dropna()))
                outlier_mask = z_scores <= factor
                outliers_in_col = len(df) - outlier_mask.sum()
                outliers_removed += outliers_in_col
                
                df_no_outliers = df_no_outliers.iloc[df[col].dropna().index[outlier_mask]]
        
        if outliers_removed > 0:
            logger.info(f"Removed {outliers_removed} outlier data points using {method} method")
        
        return df_no_outliers
    
    def handle_missing_values(self, df: pd.DataFrame, 
                            strategy: str = 'median',
                            advanced_imputation: bool = False) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        
        df_imputed = df.copy()
        
        # Check missing values
        missing_info = df.isnull().sum()
        missing_cols = missing_info[missing_info > 0]
        
        if missing_cols.empty:
            logger.info("No missing values found")
            return df_imputed
        
        logger.info(f"Found missing values in {len(missing_cols)} columns")
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Handle numeric columns
        if len(numeric_cols) > 0:
            numeric_missing = [col for col in numeric_cols if col in missing_cols.index]
            if numeric_missing:
                if advanced_imputation:
                    # Use KNN imputation for better results
                    imputer = KNNImputer(n_neighbors=5)
                    self.imputers['numeric_knn'] = imputer
                else:
                    # Use simple imputation
                    imputer = SimpleImputer(strategy=strategy)
                    self.imputers['numeric_simple'] = imputer
                
                df_imputed[numeric_missing] = imputer.fit_transform(df[numeric_missing])
        
        # Handle categorical columns
        if len(categorical_cols) > 0:
            categorical_missing = [col for col in categorical_cols if col in missing_cols.index]
            if categorical_missing:
                imputer = SimpleImputer(strategy='most_frequent')
                self.imputers['categorical'] = imputer
                df_imputed[categorical_missing] = imputer.fit_transform(df[categorical_missing])
        
        self.preprocessing_steps.append(f"Missing values handled using {strategy} strategy")
        return df_imputed
    
    def normalize_data(self, df: pd.DataFrame, 
                      method: str = 'standard',
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Normalize/Scale numerical features"""
        
        df_normalized = df.copy()
        
        # Select columns to normalize
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Choose scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Fit and transform
        self.scalers[method] = scaler
        df_normalized[columns] = scaler.fit_transform(df[columns])
        
        self.preprocessing_steps.append(f"Data normalized using {method} scaling")
        logger.info(f"Normalized {len(columns)} columns using {method} scaling")
        
        return df_normalized
    
    def smooth_data(self, df: pd.DataFrame, 
                   window_size: int = 7, 
                   poly_order: int = 2,
                   columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Apply Savitzky-Golay smoothing filter"""
        
        df_smooth = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            try:
                if len(df[col].dropna()) > window_size:
                    df_smooth[col] = savgol_filter(df[col].fillna(method='ffill'), 
                                                  window_size, poly_order)
            except Exception as e:
                logger.warning(f"Could not smooth column {col}: {str(e)}")
        
        self.preprocessing_steps.append(f"Data smoothed with window_size={window_size}")
        return df_smooth
    
    def create_time_based_split(self, df: pd.DataFrame,
                               train_size: float = 0.7,
                               val_size: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create time-based train/validation/test split"""
        
        n_samples = len(df)
        
        # Calculate split indices
        train_end = int(n_samples * train_size)
        val_end = int(n_samples * (train_size + val_size))
        
        # Split data
        train_data = df.iloc[:train_end].copy()
        val_data = df.iloc[train_end:val_end].copy()
        test_data = df.iloc[val_end:].copy()
        
        logger.info(f"Time-based split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        return train_data, val_data, test_data
    
    def add_lag_features(self, df: pd.DataFrame, 
                        lag_columns: List[str],
                        max_lag: int = 5) -> pd.DataFrame:
        """Add lag features for time series analysis"""
        
        df_lagged = df.copy()
        
        for col in lag_columns:
            if col in df.columns:
                for lag in range(1, max_lag + 1):
                    df_lagged[f"{col}_lag_{lag}"] = df[col].shift(lag)
        
        # Remove rows with NaN values created by lagging
        df_lagged = df_lagged.dropna()
        
        self.preprocessing_steps.append(f"Added lag features up to lag={max_lag}")
        logger.info(f"Added lag features for {len(lag_columns)} columns")
        
        return df_lagged
    
    def add_rolling_features(self, df: pd.DataFrame,
                           columns: List[str],
                           windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Add rolling statistics features"""
        
        df_rolling = df.copy()
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    # Rolling mean
                    df_rolling[f"{col}_rolling_mean_{window}"] = df[col].rolling(window=window).mean()
                    
                    # Rolling std
                    df_rolling[f"{col}_rolling_std_{window}"] = df[col].rolling(window=window).std()
                    
                    # Rolling min/max
                    df_rolling[f"{col}_rolling_min_{window}"] = df[col].rolling(window=window).min()
                    df_rolling[f"{col}_rolling_max_{window}"] = df[col].rolling(window=window).max()
        
        # Remove rows with NaN values
        df_rolling = df_rolling.dropna()
        
        self.preprocessing_steps.append(f"Added rolling features with windows {windows}")
        logger.info(f"Added rolling features for {len(columns)} columns")
        
        return df_rolling
    
    def add_derivative_features(self, df: pd.DataFrame,
                              columns: List[str]) -> pd.DataFrame:
        """Add derivative (rate of change) features"""
        
        df_deriv = df.copy()
        
        for col in columns:
            if col in df.columns:
                # First derivative (rate of change)
                df_deriv[f"{col}_derivative"] = df[col].diff()
                
                # Second derivative (acceleration)
                df_deriv[f"{col}_second_derivative"] = df[col].diff().diff()
        
        # Remove NaN values
        df_deriv = df_deriv.dropna()
        
        self.preprocessing_steps.append(f"Added derivative features")
        logger.info(f"Added derivative features for {len(columns)} columns")
        
        return df_deriv
    
    def create_interaction_features(self, df: pd.DataFrame,
                                  feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """Create interaction features between specified pairs"""
        
        df_interact = df.copy()
        
        for col1, col2 in feature_pairs:
            if col1 in df.columns and col2 in df.columns:
                # Multiplication
                df_interact[f"{col1}_x_{col2}"] = df[col1] * df[col2]
                
                # Ratio (avoid division by zero)
                df_interact[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + 1e-8)
                
                # Difference
                df_interact[f"{col1}_minus_{col2}"] = df[col1] - df[col2]
        
        self.preprocessing_steps.append(f"Created {len(feature_pairs)} interaction features")
        logger.info(f"Created interaction features for {len(feature_pairs)} pairs")
        
        return df_interact
    
    def prepare_formation_pressure_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Specialized preprocessing for formation pressure prediction"""
        
        logger.info("Starting formation pressure data preprocessing...")
        
        # Select relevant columns
        formation_columns = config.model.formation_pressure_features + ['FPress']
        df_formation = df[formation_columns].copy()
        
        # Clean data
        df_formation = self.clean_data(df_formation, remove_outliers=True)
        
        # Handle missing values
        df_formation = self.handle_missing_values(df_formation, strategy='median')
        
        # Add engineering features
        key_features = ['WellDepth', 'BTBR', 'WBoPress', 'RoPen']
        df_formation = self.add_derivative_features(df_formation, key_features)
        df_formation = self.add_rolling_features(df_formation, key_features, windows=[5, 10])
        
        # Add interaction features
        interaction_pairs = [
            ('WellDepth', 'BTBR'),
            ('WBoPress', 'HLoad'),
            ('WoBit', 'RoPen')
        ]
        df_formation = self.create_interaction_features(df_formation, interaction_pairs)
        
        # Smooth data
        numeric_cols = df_formation.select_dtypes(include=[np.number]).columns.tolist()
        if 'FPress' in numeric_cols:
            numeric_cols.remove('FPress')  # Don't smooth target
        df_formation = self.smooth_data(df_formation, columns=numeric_cols)
        
        logger.info(f"Formation pressure preprocessing completed. Shape: {df_formation.shape}")
        return df_formation
    
    def prepare_kick_detection_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Specialized preprocessing for kick detection"""
        
        logger.info("Starting kick detection data preprocessing...")
        
        # Select relevant columns  
        kick_columns = config.model.kick_detection_features.copy()
        if 'ActiveGL' in df.columns:
            kick_columns.append('ActiveGL')
        
        df_kick = df[kick_columns].copy()
        
        # Clean data
        df_kick = self.clean_data(df_kick, remove_outliers=True, outlier_factor=2.0)
        
        # Handle missing values
        df_kick = self.handle_missing_values(df_kick, strategy='median', advanced_imputation=True)
        
        # Add lag features (important for anomaly detection)
        key_features = ['FRate', 'ActiveGL', 'SMSpeed', 'FIn', 'FOut']
        available_features = [f for f in key_features if f in df_kick.columns]
        df_kick = self.add_lag_features(df_kick, available_features, max_lag=3)
        
        # Add rolling statistics
        df_kick = self.add_rolling_features(df_kick, available_features, windows=[5, 10, 20])
        
        # Normalize data (important for PCA-based kick detection)
        numeric_cols = df_kick.select_dtypes(include=[np.number]).columns.tolist()
        df_kick = self.normalize_data(df_kick, method='standard', columns=numeric_cols)
        
        logger.info(f"Kick detection preprocessing completed. Shape: {df_kick.shape}")
        return df_kick
    
    def transform_new_data(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Transform new data using fitted preprocessors"""
        
        if not self.scalers and not self.imputers:
            raise ValueError("Preprocessor not fitted. Please run prepare_*_data first.")
        
        df_transformed = df.copy()
        
        # Apply same preprocessing steps
        if data_type == 'formation':
            # Use formation-specific preprocessing
            df_transformed = self.clean_data(df_transformed, remove_outliers=False)  # Don't remove outliers from new data
            
            # Apply fitted imputers
            if 'numeric_simple' in self.imputers:
                numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns
                df_transformed[numeric_cols] = self.imputers['numeric_simple'].transform(df_transformed[numeric_cols])
            
            # Apply fitted scalers
            if 'standard' in self.scalers:
                numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns
                df_transformed[numeric_cols] = self.scalers['standard'].transform(df_transformed[numeric_cols])
        
        elif data_type == 'kick':
            # Use kick-specific preprocessing
            df_transformed = self.clean_data(df_transformed, remove_outliers=False)
            
            # Apply fitted transformations
            if 'numeric_knn' in self.imputers:
                numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns
                df_transformed[numeric_cols] = self.imputers['numeric_knn'].transform(df_transformed[numeric_cols])
            
            if 'standard' in self.scalers:
                numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns
                df_transformed[numeric_cols] = self.scalers['standard'].transform(df_transformed[numeric_cols])
        
        return df_transformed
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing steps applied"""
        return {
            'steps': self.preprocessing_steps,
            'scalers_fitted': list(self.scalers.keys()),
            'imputers_fitted': list(self.imputers.keys()),
            'feature_columns': self.feature_columns
        }

class FeatureSelector:
    """Feature selection utilities for drilling data"""
    
    @staticmethod
    def correlation_filter(df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
        """Remove highly correlated features"""
        
        # Calculate correlation matrix
        corr_matrix = df.select_dtypes(include=[np.number]).corr().abs()
        
        # Find features to drop
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        features_to_drop = [
            column for column in upper_triangle.columns 
            if any(upper_triangle[column] > threshold)
        ]
        
        return features_to_drop
    
    @staticmethod
    def variance_filter(df: pd.DataFrame, threshold: float = 0.01) -> List[str]:
        """Remove low variance features"""
        
        numeric_df = df.select_dtypes(include=[np.number])
        variances = numeric_df.var()
        
        low_variance_features = variances[variances < threshold].index.tolist()
        
        return low_variance_features
    
    @staticmethod
    def mutual_information_filter(X: pd.DataFrame, y: pd.Series, 
                                k: int = 10) -> List[str]:
        """Select top k features based on mutual information"""
        
        from sklearn.feature_selection import SelectKBest, mutual_info_regression
        
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
        selector.fit(X, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        
        return selected_features

def preprocess_pipeline(df: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, DataPreprocessor]:
    """Complete preprocessing pipeline"""
    
    preprocessor = DataPreprocessor()
    
    if data_type == 'formation':
        processed_df = preprocessor.prepare_formation_pressure_data(df)
    elif data_type == 'kick':
        processed_df = preprocessor.prepare_kick_detection_data(df)
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    return processed_df, preprocessor

if __name__ == "__main__":
    # Test preprocessing
    from data_loader import DataLoader
    
    try:
        # Load data
        loader = DataLoader()
        formation_data = loader.load_formation_data()
        
        # Test preprocessing
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.prepare_formation_pressure_data(formation_data)
        
        print(f"Original data shape: {formation_data.shape}")
        print(f"Processed data shape: {processed_data.shape}")
        print(f"Processing steps: {len(preprocessor.preprocessing_steps)}")
        
        # Show preprocessing summary
        summary = preprocessor.get_preprocessing_summary()
        print("\nPreprocessing Summary:")
        for step in summary['steps']:
            print(f"- {step}")
            
    except Exception as e:
        print(f"Error testing preprocessor: {str(e)}")