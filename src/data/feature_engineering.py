"""
Feature Engineering Module for Drilling Operations ML
Specialized feature engineering for oil & gas drilling data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.signal import savgol_filter
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import config
import logging

logger = logging.getLogger(__name__)

class DrillingFeatureEngineer:
    """Advanced feature engineering for drilling operations"""
    
    def __init__(self):
        self.feature_history = []
        self.drilling_physics_enabled = True
        
    def create_drilling_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create drilling efficiency and performance features"""
        
        df_enhanced = df.copy()
        
        # Mechanical Specific Energy (MSE) - Key drilling efficiency metric
        if all(col in df.columns for col in ['WoBit', 'RoPen', 'BTBR']):
            # MSE = (WOB/bit_area) + (2*π*RPM*Torque)/(ROP*bit_area)
            # Simplified version without bit area
            df_enhanced['MSE'] = (df['WoBit'] + 2 * np.pi * df.get('BTBR', 0) * df.get('Torque', 0)) / (df['RoPen'] + 1e-8)
            self.feature_history.append('MSE: Mechanical Specific Energy')
        
        # WOB Efficiency (Rate of Penetration per unit WOB)
        if 'WoBit' in df.columns and 'RoPen' in df.columns:
            df_enhanced['WOB_Efficiency'] = df['RoPen'] / (df['WoBit'] + 1e-8)
            self.feature_history.append('WOB_Efficiency: ROP per unit WOB')
        
        # Torque to WOB Ratio
        if 'BTBR' in df.columns and 'WoBit' in df.columns:
            df_enhanced['Torque_WOB_Ratio'] = df['BTBR'] / (df['WoBit'] + 1e-8)
            self.feature_history.append('Torque_WOB_Ratio: Torque efficiency')
        
        # Hydraulic Horsepower
        if 'FRate' in df.columns and 'WBoPress' in df.columns:
            df_enhanced['Hydraulic_HP'] = (df['FRate'] * df['WBoPress']) / 1714  # Convert to HP
            self.feature_history.append('Hydraulic_HP: Hydraulic power')
        
        # Flow Rate per Hole Volume (circulation efficiency)
        if 'FRate' in df.columns and 'WellDepth' in df.columns:
            # Approximate hole volume (simplified)
            hole_volume = df['WellDepth'] * 0.1  # Simplified calculation
            df_enhanced['Flow_Efficiency'] = df['FRate'] / (hole_volume + 1e-8)
            self.feature_history.append('Flow_Efficiency: Circulation efficiency')
        
        logger.info(f"Created {5} drilling efficiency features")
        return df_enhanced
    
    def create_formation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create formation-related features"""
        
        df_enhanced = df.copy()
        
        # Formation Pressure Gradient
        if 'FPress' in df.columns and 'WellDepth' in df.columns:
            df_enhanced['FPress_Gradient'] = df['FPress'] / (df['WellDepth'] + 1e-8)
            self.feature_history.append('FPress_Gradient: Formation pressure per depth')
        
        # Overbalance (difference between mud pressure and formation pressure)
        if 'WBoPress' in df.columns and 'FPress' in df.columns:
            df_enhanced['Overbalance'] = df['WBoPress'] - df['FPress']
            df_enhanced['Overbalance_Ratio'] = df['WBoPress'] / (df['FPress'] + 1e-8)
            self.feature_history.append('Overbalance: Mud vs Formation pressure difference')
        
        # Formation Strength Indicator
        if 'WoBit' in df.columns and 'RoPen' in df.columns:
            df_enhanced['Formation_Strength'] = df['WoBit'] / (df['RoPen'] + 1e-8)
            self.feature_history.append('Formation_Strength: Resistance to drilling')
        
        # Differential Pressure indicators
        if 'DPPress' in df.columns and 'FPress' in df.columns:
            df_enhanced['DP_FP_Ratio'] = df['DPPress'] / (df['FPress'] + 1e-8)
            self.feature_history.append('DP_FP_Ratio: Differential to Formation pressure ratio')
        
        logger.info(f"Created formation-related features")
        return df_enhanced
    
    def create_kick_detection_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features specific to kick detection"""
        
        df_enhanced = df.copy()
        
        # Flow Rate Balance (key kick indicator)
        if 'FIn' in df.columns and 'FOut' in df.columns:
            df_enhanced['Flow_Balance'] = df['FOut'] - df['FIn']
            df_enhanced['Flow_Balance_Ratio'] = df['FOut'] / (df['FIn'] + 1e-8)
            self.feature_history.append('Flow_Balance: Flow in vs out difference')
        
        # Active Pit Volume Rate of Change
        if 'ActiveGL' in df.columns:
            df_enhanced['ActiveGL_Rate'] = df['ActiveGL'].diff()
            df_enhanced['ActiveGL_Acceleration'] = df_enhanced['ActiveGL_Rate'].diff()
            self.feature_history.append('ActiveGL_Rate: Pit volume change rate')
        
        # Standpipe Pressure Variations
        if 'WBoPress' in df.columns:
            df_enhanced['SPP_Variation'] = df['WBoPress'].rolling(window=5).std()
            df_enhanced['SPP_Trend'] = df['WBoPress'].rolling(window=10).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0]
            )
            self.feature_history.append('SPP_Variation: Standpipe pressure variations')
        
        # Mud Return Flow Anomalies
        if 'MRFlow' in df.columns:
            df_enhanced['MRFlow_MA'] = df['MRFlow'].rolling(window=10).mean()
            df_enhanced['MRFlow_Deviation'] = df['MRFlow'] - df_enhanced['MRFlow_MA']
            self.feature_history.append('MRFlow_Deviation: Mud return flow anomalies')
        
        # Hook Load Variations (can indicate kick)
        if 'HLoad' in df.columns:
            df_enhanced['HLoad_Variation'] = df['HLoad'].rolling(window=5).std()
            df_enhanced['HLoad_Trend'] = df['HLoad'].rolling(window=10).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0
            )
            self.feature_history.append('HLoad_Variation: Hook load variations')
        
        logger.info(f"Created kick detection features")
        return df_enhanced
    
    def create_temporal_features(self, df: pd.DataFrame, 
                               key_columns: List[str]) -> pd.DataFrame:
        """Create temporal features for time series analysis"""
        
        df_enhanced = df.copy()
        
        for col in key_columns:
            if col in df.columns:
                # Rate of change features
                df_enhanced[f'{col}_Rate'] = df[col].diff()
                df_enhanced[f'{col}_Acceleration'] = df_enhanced[f'{col}_Rate'].diff()
                
                # Rolling statistics
                for window in [5, 10, 20]:
                    df_enhanced[f'{col}_MA_{window}'] = df[col].rolling(window=window).mean()
                    df_enhanced[f'{col}_STD_{window}'] = df[col].rolling(window=window).std()
                    df_enhanced[f'{col}_MIN_{window}'] = df[col].rolling(window=window).min()
                    df_enhanced[f'{col}_MAX_{window}'] = df[col].rolling(window=window).max()
                
                # Trend features
                df_enhanced[f'{col}_Trend_5'] = df[col].rolling(window=5).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0
                )
                
                # Volatility features
                df_enhanced[f'{col}_Volatility'] = df[col].rolling(window=10).std() / (
                    df[col].rolling(window=10).mean() + 1e-8
                )
        
        self.feature_history.append(f'Temporal features for {len(key_columns)} columns')
        logger.info(f"Created temporal features for {len(key_columns)} columns")
        return df_enhanced
    
    def create_depth_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on well depth"""
        
        if 'WellDepth' not in df.columns:
            logger.warning("WellDepth not available for depth-based features")
            return df
        
        df_enhanced = df.copy()
        
        # Depth bins (shallow, medium, deep)
        depth_percentiles = df['WellDepth'].quantile([0.33, 0.66])
        df_enhanced['Depth_Category'] = pd.cut(
            df['WellDepth'], 
            bins=[-np.inf, depth_percentiles.iloc[0], depth_percentiles.iloc[1], np.inf],
            labels=['Shallow', 'Medium', 'Deep']
        )
        
        # Depth-normalized features
        depth_related_cols = ['FPress', 'WBoPress', 'HLoad']
        for col in depth_related_cols:
            if col in df.columns:
                df_enhanced[f'{col}_per_Depth'] = df[col] / (df['WellDepth'] + 1e-8)
        
        # Cumulative depth change
        df_enhanced['Depth_Change'] = df['WellDepth'].diff()
        df_enhanced['Cumulative_Depth_Change'] = df_enhanced['Depth_Change'].cumsum()
        
        self.feature_history.append('Depth-based features')
        logger.info("Created depth-based features")
        return df_enhanced
    
    def create_physics_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on drilling physics principles"""
        
        if not self.drilling_physics_enabled:
            return df
        
        df_enhanced = df.copy()
        
        # Bit Hydraulics Features
        if all(col in df.columns for col in ['FRate', 'WBoPress']):
            # Hydraulic Impact (jet impact force)
            df_enhanced['Hydraulic_Impact'] = df['FRate'] * np.sqrt(df['WBoPress'])
            
            # Flow velocity (simplified)
            df_enhanced['Flow_Velocity'] = df['FRate'] / 100  # Simplified nozzle area
            
            self.feature_history.append('Hydraulic physics features')
        
        # Weight Transfer Efficiency
        if all(col in df.columns for col in ['WoBit', 'HLoad']):
            # Effective WOB (considering hook load)
            df_enhanced['Effective_WOB'] = df['WoBit'] - (df['HLoad'] * 0.1)  # Simplified
            df_enhanced['Weight_Transfer_Efficiency'] = df['WoBit'] / (df['HLoad'] + 1e-8)
            
            self.feature_history.append('Weight transfer features')
        
        # Torque and Power Features
        if all(col in df.columns for col in ['BTBR', 'RoPen']):
            # Specific Torque (torque per ROP)
            df_enhanced['Specific_Torque'] = df['BTBR'] / (df['RoPen'] + 1e-8)
            
            # Rotary Power (simplified)
            if 'RPM' in df.columns:
                df_enhanced['Rotary_Power'] = df['BTBR'] * df['RPM'] / 5252  # Convert to HP
            
            self.feature_history.append('Torque and power features')
        
        # Pressure Balance Features
        if all(col in df.columns for col in ['WBoPress', 'FPress', 'DPPress']):
            # Total system pressure
            df_enhanced['Total_System_Pressure'] = df['WBoPress'] + df['DPPress']
            
            # Pressure efficiency
            df_enhanced['Pressure_Efficiency'] = df['FPress'] / (df['WBoPress'] + 1e-8)
            
            self.feature_history.append('Pressure balance features')
        
        logger.info("Created physics-based features")
        return df_enhanced
    
    def create_anomaly_detection_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features specifically for anomaly detection"""
        
        df_enhanced = df.copy()
        
        # Statistical anomaly indicators
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in df.columns:
                # Z-score based anomaly indicator
                mean_val = df[col].rolling(window=50).mean()
                std_val = df[col].rolling(window=50).std()
                df_enhanced[f'{col}_ZScore'] = (df[col] - mean_val) / (std_val + 1e-8)
                
                # Distance from moving median
                median_val = df[col].rolling(window=20).median()
                df_enhanced[f'{col}_Median_Distance'] = np.abs(df[col] - median_val)
                
                # Rate of change anomaly
                rate_change = df[col].diff()
                rate_std = rate_change.rolling(window=20).std()
                df_enhanced[f'{col}_Rate_Anomaly'] = np.abs(rate_change) / (rate_std + 1e-8)
        
        self.feature_history.append('Anomaly detection features')
        logger.info("Created anomaly detection features")
        return df_enhanced
    
    def create_interaction_features(self, df: pd.DataFrame, 
                                  feature_groups: Dict[str, List[str]]) -> pd.DataFrame:
        """Create interaction features between related groups"""
        
        df_enhanced = df.copy()
        
        for group_name, features in feature_groups.items():
            available_features = [f for f in features if f in df.columns]
            
            if len(available_features) >= 2:
                # Pairwise interactions within group
                for i, feat1 in enumerate(available_features):
                    for feat2 in available_features[i+1:]:
                        # Multiplicative interaction
                        df_enhanced[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                        
                        # Ratio interaction
                        df_enhanced[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-8)
                        
                        # Difference interaction
                        df_enhanced[f'{feat1}_minus_{feat2}'] = df[feat1] - df[feat2]
        
        self.feature_history.append(f'Interaction features for {len(feature_groups)} groups')
        logger.info(f"Created interaction features for {len(feature_groups)} feature groups")
        return df_enhanced
    
    def create_lag_features(self, df: pd.DataFrame, 
                           target_columns: List[str],
                           lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Create lagged features for temporal dependencies"""
        
        df_enhanced = df.copy()
        
        for col in target_columns:
            if col in df.columns:
                for lag in lags:
                    df_enhanced[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Remove NaN rows created by lagging
        df_enhanced = df_enhanced.dropna()
        
        self.feature_history.append(f'Lag features for {len(target_columns)} columns')
        logger.info(f"Created lag features for {len(target_columns)} columns")
        return df_enhanced
    
    def create_polynomial_features(self, df: pd.DataFrame,
                                 feature_columns: List[str],
                                 degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for non-linear relationships"""
        
        available_features = [col for col in feature_columns if col in df.columns]
        
        if not available_features:
            logger.warning("No features available for polynomial transformation")
            return df
        
        df_enhanced = df.copy()
        
        # Select subset of features to avoid explosion
        poly = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=False)
        
        # Apply to subset of features
        subset_data = df[available_features[:5]]  # Limit to first 5 to avoid too many features
        poly_features = poly.fit_transform(subset_data)
        
        # Create feature names
        poly_feature_names = poly.get_feature_names_out(available_features[:5])
        
        # Add new polynomial features
        for i, name in enumerate(poly_feature_names):
            if name not in available_features:  # Don't duplicate existing features
                df_enhanced[f'poly_{name}'] = poly_features[:, i]
        
        self.feature_history.append(f'Polynomial features degree {degree}')
        logger.info(f"Created polynomial features of degree {degree}")
        return df_enhanced
    
    def apply_feature_selection(self, df: pd.DataFrame, 
                              target_column: str,
                              method: str = 'correlation',
                              top_k: int = 50) -> Tuple[pd.DataFrame, List[str]]:
        """Apply feature selection to reduce dimensionality"""
        
        if target_column not in df.columns:
            raise ValueError(f"Target column {target_column} not found")
        
        feature_columns = [col for col in df.columns if col != target_column]
        X = df[feature_columns]
        y = df[target_column]
        
        if method == 'correlation':
            # Select features based on correlation with target
            correlations = X.corrwith(y).abs().sort_values(ascending=False)
            selected_features = correlations.head(top_k).index.tolist()
        
        elif method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression, SelectKBest
            
            # Handle any remaining NaN values
            X_clean = X.fillna(X.mean())
            y_clean = y.fillna(y.mean())
            
            selector = SelectKBest(score_func=mutual_info_regression, k=min(top_k, len(feature_columns)))
            selector.fit(X_clean, y_clean)
            
            selected_features = X.columns[selector.get_support()].tolist()
        
        elif method == 'variance':
            from sklearn.feature_selection import VarianceThreshold
            
            # Remove low variance features first
            selector = VarianceThreshold(threshold=0.01)
            selector.fit(X.fillna(X.mean()))
            
            high_var_features = X.columns[selector.get_support()].tolist()
            
            # Then select by correlation
            if len(high_var_features) > top_k:
                correlations = X[high_var_features].corrwith(y).abs().sort_values(ascending=False)
                selected_features = correlations.head(top_k).index.tolist()
            else:
                selected_features = high_var_features
        
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        # Create reduced dataframe
        selected_features.append(target_column)  # Add target back
        df_selected = df[selected_features].copy()
        
        self.feature_history.append(f'Feature selection: {method}, kept {len(selected_features)-1} features')
        logger.info(f"Selected {len(selected_features)-1} features using {method} method")
        
        return df_selected, selected_features[:-1]  # Return without target column in feature list
    
    def get_feature_importance_drilling(self, df: pd.DataFrame, 
                                      target_column: str) -> Dict[str, float]:
        """Get feature importance specific to drilling domain"""
        
        feature_columns = [col for col in df.columns if col != target_column]
        
        # Domain-specific feature importance weights
        drilling_importance = {
            # Core drilling parameters (high importance)
            'WoBit': 1.0, 'RoPen': 1.0, 'BTBR': 1.0,
            'WBoPress': 0.9, 'FPress': 0.9,
            
            # Flow and circulation (medium-high)
            'FRate': 0.8, 'FIn': 0.8, 'FOut': 0.8,
            'MRFlow': 0.7, 'ActiveGL': 0.9,
            
            # Mechanical parameters (medium)
            'HLoad': 0.7, 'SMSpeed': 0.6,
            
            # Depth and formation (high for formation pressure)
            'WellDepth': 0.9, 'DPPress': 0.8,
            
            # Derived features (variable)
            'MSE': 0.9, 'WOB_Efficiency': 0.8,
            'Flow_Balance': 0.9, 'Overbalance': 0.8
        }
        
        # Calculate statistical correlation
        correlations = {}
        for col in feature_columns:
            if col in df.columns and df[col].dtype in [np.float64, np.int64]:
                corr = df[col].corr(df[target_column])
                correlations[col] = abs(corr) if not np.isnan(corr) else 0
        
        # Combine domain knowledge with statistical correlation
        combined_importance = {}
        for col in feature_columns:
            domain_weight = drilling_importance.get(col, 0.5)  # Default weight
            stat_weight = correlations.get(col, 0)
            
            # Weighted combination
            combined_importance[col] = 0.6 * domain_weight + 0.4 * stat_weight
        
        return combined_importance
    
    def create_drilling_features_pipeline(self, df: pd.DataFrame, 
                                        target_type: str = 'formation') -> pd.DataFrame:
        """Complete feature engineering pipeline for drilling data"""
        
        logger.info(f"Starting {target_type} feature engineering pipeline...")
        
        df_enhanced = df.copy()
        
        # Core drilling features
        df_enhanced = self.create_drilling_efficiency_features(df_enhanced)
        
        if target_type == 'formation':
            df_enhanced = self.create_formation_features(df_enhanced)
            
            # Key temporal features for formation prediction
            key_cols = ['WellDepth', 'WBoPress', 'RoPen', 'BTBR']
            df_enhanced = self.create_temporal_features(df_enhanced, key_cols)
            
            # Depth-based features
            df_enhanced = self.create_depth_based_features(df_enhanced)
            
            # Physics-based features
            df_enhanced = self.create_physics_based_features(df_enhanced)
            
            # Feature interactions for formation pressure
            interaction_groups = {
                'pressure': ['WBoPress', 'FPress', 'DPPress'],
                'mechanical': ['WoBit', 'RoPen', 'BTBR'],
                'hydraulic': ['FRate', 'WBoPress']
            }
            df_enhanced = self.create_interaction_features(df_enhanced, interaction_groups)
        
        elif target_type == 'kick':
            df_enhanced = self.create_kick_detection_features(df_enhanced)
            
            # Temporal features for kick detection
            key_cols = ['ActiveGL', 'FIn', 'FOut', 'WBoPress', 'MRFlow']
            df_enhanced = self.create_temporal_features(df_enhanced, key_cols)
            
            # Anomaly detection features
            df_enhanced = self.create_anomaly_detection_features(df_enhanced)
            
            # Lag features (important for kick detection)
            lag_cols = ['ActiveGL', 'FIn', 'FOut', 'MRFlow']
            df_enhanced = self.create_lag_features(df_enhanced, lag_cols, lags=[1, 2, 3, 5])
        
        # Remove any infinite or very large values
        df_enhanced = df_enhanced.replace([np.inf, -np.inf], np.nan)
        df_enhanced = df_enhanced.fillna(df_enhanced.mean(numeric_only=True))
        
        logger.info(f"Feature engineering completed. Shape: {df.shape} -> {df_enhanced.shape}")
        logger.info(f"Created {len(self.feature_history)} feature groups")
        
        return df_enhanced
    
    def get_feature_engineering_summary(self) -> Dict[str, Any]:
        """Get summary of feature engineering process"""
        return {
            'feature_groups_created': len(self.feature_history),
            'feature_history': self.feature_history,
            'physics_features_enabled': self.drilling_physics_enabled
        }

# Utility functions for specific drilling calculations
def calculate_mse(wob: np.ndarray, rop: np.ndarray, torque: np.ndarray, 
                 rpm: np.ndarray, bit_diameter: float = 8.5) -> np.ndarray:
    """
    Calculate Mechanical Specific Energy (MSE)
    MSE = (WOB/A) + (2*π*N*T)/(ROP*A)
    where A = bit area, N = RPM, T = Torque
    """
    bit_area = np.pi * (bit_diameter / 2) ** 2  # square inches
    
    mse = (wob / bit_area) + (2 * np.pi * rpm * torque) / (rop * bit_area + 1e-8)
    
    return mse

def calculate_hydraulic_hp(flow_rate: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    """Calculate Hydraulic Horsepower"""
    return (flow_rate * pressure) / 1714

def calculate_bit_hydraulics(flow_rate: np.ndarray, nozzle_area: float = 0.5) -> Dict[str, np.ndarray]:
    """Calculate bit hydraulic parameters"""
    
    # Flow velocity through nozzles
    velocity = flow_rate / nozzle_area
    
    # Jet impact force (simplified)
    jet_impact = flow_rate * velocity * 0.052  # Conversion factor
    
    return {
        'velocity': velocity,
        'jet_impact': jet_impact
    }

if __name__ == "__main__":
    # Test feature engineering
    from data_loader import DataLoader
    
    try:
        # Load data
        loader = DataLoader()
        formation_data = loader.load_formation_data()
        
        # Test feature engineering
        engineer = DrillingFeatureEngineer()
        enhanced_data = engineer.create_drilling_features_pipeline(
            formation_data, target_type='formation'
        )
        
        print(f"Original features: {len(formation_data.columns)}")
        print(f"Enhanced features: {len(enhanced_data.columns)}")
        print(f"Feature engineering steps: {len(engineer.feature_history)}")
        
        # Show feature engineering summary
        summary = engineer.get_feature_engineering_summary()
        print("\nFeature Engineering Summary:")
        for step in summary['feature_history']:
            print(f"- {step}")
            
    except Exception as e:
        print(f"Error testing feature engineering: {str(e)}")
        
        # Create sample data for testing
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'WellDepth': np.cumsum(np.random.normal(1, 0.1, 100)) + 1000,
            'WoBit': np.random.normal(25, 5, 100),
            'RoPen': np.random.normal(15, 3, 100),
            'BTBR': np.random.normal(120, 10, 100),
            'WBoPress': np.random.normal(200, 20, 100),
            'FPress': np.random.normal(180, 15, 100),
            'FRate': np.random.normal(300, 30, 100)
        })
        
        engineer = DrillingFeatureEngineer()
        enhanced_data = engineer.create_drilling_efficiency_features(sample_data)
        
        print("Sample feature engineering test completed successfully!")
        print(f"Created {len(enhanced_data.columns) - len(sample_data.columns)} new features")