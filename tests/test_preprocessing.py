"""
Tests for Data Preprocessing Module
===================================

Tests unitaires pour les modules de prétraitement des données de forage
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import warnings
warnings.filterwarnings('ignore')

# Imports du projet (avec gestion d'erreur pour CI/CD)
try:
    from src.data.data_loader import DrillingDataLoader
    from src.data.data_preprocessor import DrillingDataPreprocessor
    from src.data.feature_engineering import FeatureEngineer
except ImportError:
    # Fallback pour développement local
    import sys
    sys.path.append('.')

class TestDrillingDataLoader(unittest.TestCase):
    """
    Tests pour la classe DrillingDataLoader
    """
    
    def setUp(self):
        """Configuration initiale des tests"""
        self.config = {'data_path': 'test_data'}
        self.loader = DrillingDataLoader(self.config)
        
        # Données de test
        self.sample_formation_data = pd.DataFrame({
            'Depth': [1000, 1100, 1200],
            'FormationPressure': [2000, 2200, 2400],
            'MudWeight': [1.2, 1.3, 1.4],
            'Temperature': [25, 30, 35]
        })
        
        self.sample_kick_data = pd.DataFrame({
            'FlowRateIn': [350, 360, 340],
            'FlowRateOut': [350, 355, 342],
            'StandpipePressure': [200, 205, 198],
            'Kick': [0, 1, 0]
        })
    
    def test_init(self):
        """Test de l'initialisation"""
        self.assertIsNotNone(self.loader)
        self.assertEqual(self.loader.data_path.name, 'test_data')
    
    def test_load_synthetic_data(self):
        """Test de génération de données synthétiques"""
        df = self.loader.load_synthetic_drilling_data(100)
        
        self.assertEqual(len(df), 100)
        self.assertIn('Depth', df.columns)
        self.assertIn('WOB', df.columns)
        self.assertIn('RPM', df.columns)
        
        # Vérifier que les valeurs sont positives
        self.assertTrue((df['WOB'] >= 0).all())
        self.assertTrue((df['RPM'] >= 0).all())
    
    @patch('pandas.read_csv')
    def test_load_formation_data_success(self, mock_read_csv):
        """Test de chargement réussi des données de formation"""
        mock_read_csv.return_value = self.sample_formation_data
        
        df = self.loader.load_formation_data('dummy_path.csv')
        
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 3)
        self.assertIn('FormationPressure', df.columns)
    
    @patch('pandas.read_csv')
    def test_load_kick_data_success(self, mock_read_csv):
        """Test de chargement réussi des données de kick"""
        mock_read_csv.return_value = self.sample_kick_data
        
        df = self.loader.load_kick_detection_data('dummy_path.csv')
        
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 3)
        self.assertIn('Kick', df.columns)

class TestDrillingDataPreprocessor(unittest.TestCase):
    """
    Tests pour la classe DrillingDataPreprocessor
    """
    
    def setUp(self):
        """Configuration initiale des tests"""
        self.preprocessor = DrillingDataPreprocessor()
        
        # Données de test avec quelques outliers et valeurs manquantes
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'WOB': [10, 20, 30, 100, 25, np.nan, 22],  # Outlier à 100
            'RPM': [100, 120, 110, 115, np.nan, 125, 130],
            'FlowRate': [300, 350, 320, 340, 330, 360, 310],
            'Target': [5, 10, 8, 12, 9, 11, 7]
        })
    
    def test_init(self):
        """Test de l'initialisation"""
        self.assertIsNotNone(self.preprocessor)
        self.assertIsInstance(self.preprocessor.config, dict)
    
    def test_handle_missing_values(self):
        """Test de gestion des valeurs manquantes"""
        df_cleaned = self.preprocessor.handle_missing_values(self.sample_data.copy())
        
        # Vérifier qu'il n'y a plus de valeurs manquantes
        self.assertFalse(df_cleaned.isnull().any().any())
        
        # Vérifier que les dimensions sont conservées
        self.assertEqual(df_cleaned.shape, self.sample_data.shape)
    
    def test_detect_outliers(self):
        """Test de détection des outliers"""
        outliers_idx = self.preprocessor.detect_outliers(self.sample_data, ['WOB'])
        
        # L'outlier à 100 devrait être détecté
        self.assertTrue(len(outliers_idx) > 0)
        self.assertIn(3, outliers_idx)  # Index de la valeur 100
    
    def test_remove_outliers(self):
        """Test de suppression des outliers"""
        df_no_outliers = self.preprocessor.remove_outliers(self.sample_data.copy(), ['WOB'])
        
        # Vérifier que l'outlier a été supprimé
        self.assertTrue(df_no_outliers['WOB'].max() < 100)
        self.assertTrue(len(df_no_outliers) < len(self.sample_data))
    
    def test_cap_outliers(self):
        """Test de capping des outliers"""
        df_capped = self.preprocessor.cap_outliers(self.sample_data.copy(), ['WOB'])
        
        # Vérifier que les valeurs extrêmes ont été plafonnées
        self.assertTrue(df_capped['WOB'].max() <= df_capped['WOB'].quantile(0.95))
        self.assertEqual(len(df_capped), len(self.sample_data))
    
    def test_normalize_data(self):
        """Test de normalisation des données"""
        df_clean = self.preprocessor.handle_missing_values(self.sample_data.copy())
        df_normalized, scaler = self.preprocessor.normalize_data(df_clean[['WOB', 'RPM']])
        
        # Vérifier que les données sont normalisées (moyenne ≈ 0, std ≈ 1)
        self.assertAlmostEqual(df_normalized.mean().mean(), 0, places=1)
        self.assertAlmostEqual(df_normalized.std().mean(), 1, places=1)
        self.assertIsNotNone(scaler)
    
    def test_split_data(self):
        """Test de division des données"""
        df_clean = self.preprocessor.handle_missing_values(self.sample_data.copy())
        X = df_clean[['WOB', 'RPM', 'FlowRate']]
        y = df_clean['Target']
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.split_data(X, y)
        
        # Vérifier les dimensions
        total_size = len(X)
        self.assertEqual(len(X_train) + len(X_val) + len(X_test), total_size)
        self.assertEqual(len(y_train) + len(y_val) + len(y_test), total_size)
        
        # Vérifier que train est le plus grand
        self.assertTrue(len(X_train) >= len(X_val))
        self.assertTrue(len(X_train) >= len(X_test))

class TestFeatureEngineer(unittest.TestCase):
    """
    Tests pour la classe FeatureEngineer
    """
    
    def setUp(self):
        """Configuration initiale des tests"""
        self.feature_engineer = FeatureEngineer()
        
        # Données de test avec timestamp
        dates = pd.date_range('2024-01-01', periods=10, freq='1H')
        self.sample_data = pd.DataFrame({
            'Timestamp': dates,
            'WOB': [20, 22, 25, 23, 21, 24, 26, 22, 20, 25],
            'RPM': [100, 110, 120, 115, 105, 125, 130, 118, 108, 122],
            'FlowRate': [300, 320, 340, 330, 310, 350, 360, 335, 315, 345],
            'ROP': [5, 7, 10, 8, 6, 12, 15, 9, 5, 11]
        })
    
    def test_init(self):
        """Test de l'initialisation"""
        self.assertIsNotNone(self.feature_engineer)
    
    def test_create_temporal_features(self):
        """Test de création des features temporelles"""
        df_with_temporal = self.feature_engineer.create_temporal_features(
            self.sample_data.copy(), 'Timestamp'
        )
        
        # Vérifier que les nouvelles colonnes ont été ajoutées
        expected_cols = ['hour', 'day_of_week', 'month', 'quarter']
        for col in expected_cols:
            self.assertIn(col, df_with_temporal.columns)
        
        # Vérifier les valeurs
        self.assertTrue((df_with_temporal['hour'] >= 0).all())
        self.assertTrue((df_with_temporal['hour'] <= 23).all())
    
    def test_create_rolling_features(self):
        """Test de création des features de moyennes mobiles"""
        df_with_rolling = self.feature_engineer.create_rolling_features(
            self.sample_data.copy(), ['WOB', 'RPM'], [3, 5]
        )
        
        # Vérifier que les nouvelles colonnes ont été créées
        expected_cols = ['WOB_rolling_mean_3', 'WOB_rolling_std_3', 
                        'RPM_rolling_mean_5', 'RPM_rolling_std_5']
        for col in expected_cols:
            self.assertIn(col, df_with_rolling.columns)
        
        # Vérifier qu'il n'y a pas de NaN dans les moyennes mobiles
        self.assertFalse(df_with_rolling['WOB_rolling_mean_3'].isnull().any())
    
    def test_create_lag_features(self):
        """Test de création des features de lag"""
        df_with_lags = self.feature_engineer.create_lag_features(
            self.sample_data.copy(), ['WOB'], [1, 2]
        )
        
        # Vérifier que les colonnes de lag ont été créées
        expected_cols = ['WOB_lag_1', 'WOB_lag_2']
        for col in expected_cols:
            self.assertIn(col, df_with_lags.columns)
        
        # Vérifier les valeurs de lag
        self.assertEqual(df_with_lags['WOB_lag_1'].iloc[1], self.sample_data['WOB'].iloc[0])
        self.assertEqual(df_with_lags['WOB_lag_2'].iloc[2], self.sample_data['WOB'].iloc[0])
    
    def test_create_ratio_features(self):
        """Test de création des features de ratios"""
        df_with_ratios = self.feature_engineer.create_ratio_features(
            self.sample_data.copy(), [('WOB', 'RPM'), ('FlowRate', 'ROP')]
        )
        
        # Vérifier que les colonnes de ratios ont été créées
        expected_cols = ['WOB_RPM_ratio', 'FlowRate_ROP_ratio']
        for col in expected_cols:
            self.assertIn(col, df_with_ratios.columns)
        
        # Vérifier le calcul du ratio
        expected_ratio = self.sample_data['WOB'].iloc[0] / self.sample_data['RPM'].iloc[0]
        self.assertAlmostEqual(df_with_ratios['WOB_RPM_ratio'].iloc[0], expected_ratio, places=5)
    
    def test_create_drilling_efficiency_features(self):
        """Test de création des features d'efficacité de forage"""
        df_with_efficiency = self.feature_engineer.create_drilling_efficiency_features(
            self.sample_data.copy()
        )
        
        # Vérifier que les features d'efficacité ont été créées
        expected_cols = ['drilling_efficiency', 'mechanical_specific_energy', 'rop_per_rpm']
        for col in expected_cols:
            self.assertIn(col, df_with_efficiency.columns)
        
        # Vérifier que les valeurs sont positives (ou nulles)
        for col in expected_cols:
            self.assertTrue((df_with_efficiency[col] >= 0).all())
    
    def test_create_anomaly_features(self):
        """Test de création des features d'anomalies"""
        df_with_anomalies = self.feature_engineer.create_anomaly_features(
            self.sample_data.copy(), ['WOB', 'RPM']
        )
        
        # Vérifier que les features d'anomalies ont été créées
        expected_cols = ['WOB_zscore', 'RPM_zscore', 'WOB_is_outlier', 'RPM_is_outlier']
        for col in expected_cols:
            self.assertIn(col, df_with_anomalies.columns)
        
        # Vérifier le type des features booléennes
        self.assertTrue(df_with_anomalies['WOB_is_outlier'].dtype == bool)
        self.assertTrue(df_with_anomalies['RPM_is_outlier'].dtype == bool)
    
    def test_select_features_correlation(self):
        """Test de sélection de features par corrélation"""
        # Créer des données avec corrélation
        df_corr = self.sample_data.copy()
        df_corr['WOB_copy'] = df_corr['WOB']  # Feature parfaitement corrélée
        
        selected_features = self.feature_engineer.select_features_correlation(
            df_corr, 'ROP', threshold=0.95
        )
        
        # Une des features corrélées devrait être supprimée
        self.assertNotIn('WOB_copy', selected_features)
        self.assertIn('WOB', selected_features)
    
    def test_select_features_importance(self):
        """Test de sélection de features par importance"""
        from sklearn.ensemble import RandomForestRegressor
        
        X = self.sample_data[['WOB', 'RPM', 'FlowRate']]
        y = self.sample_data['ROP']
        
        selected_features = self.feature_engineer.select_features_importance(
            X, y, RandomForestRegressor(n_estimators=10, random_state=42), n_features=2
        )
        
        # Devrait retourner 2 features
        self.assertEqual(len(selected_features), 2)
        self.assertTrue(all(feat in X.columns for feat in selected_features))

class TestIntegration(unittest.TestCase):
    """
    Tests d'intégration pour le pipeline complet de preprocessing
    """
    
    def setUp(self):
        """Configuration initiale des tests d'intégration"""
        self.loader = DrillingDataLoader()
        self.preprocessor = DrillingDataPreprocessor()
        self.feature_engineer = FeatureEngineer()
    
    def test_full_preprocessing_pipeline(self):
        """Test du pipeline complet de preprocessing"""
        # 1. Générer des données synthétiques
        df = self.loader.load_synthetic_drilling_data(100)
        
        # 2. Preprocessing de base
        df_clean = self.preprocessor.handle_missing_values(df)
        df_no_outliers = self.preprocessor.remove_outliers(df_clean, ['WOB', 'RPM'])
        
        # 3. Feature engineering
        if 'Timestamp' in df_no_outliers.columns:
            df_engineered = self.feature_engineer.create_temporal_features(
                df_no_outliers, 'Timestamp'
            )
        else:
            df_engineered = df_no_outliers.copy()
        
        df_final = self.feature_engineer.create_rolling_features(
            df_engineered, ['WOB', 'RPM'], [3, 5]
        )
        
        # 4. Normalisation
        numeric_cols = df_final.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            df_normalized, scaler = self.preprocessor.normalize_data(df_final[numeric_cols])
            
            # Vérifications finales
            self.assertIsNotNone(df_normalized)
            self.assertIsNotNone(scaler)
            self.assertTrue(len(df_final) > 0)
            self.assertFalse(df_final.select_dtypes(include=[np.number]).isnull().any().any())
    
    def test_data_consistency(self):
        """Test de cohérence des données après preprocessing"""
        df = self.loader.load_synthetic_drilling_data(50)
        
        # Preprocessing
        df_processed = self.preprocessor.preprocess_data(df, target_column='ROP')
        
        # Vérifications de cohérence
        self.assertEqual(len(df_processed['X_train']) + len(df_processed['X_val']) + len(df_processed['X_test']), len(df))
        self.assertEqual(len(df_processed['y_train']) + len(df_processed['y_val']) + len(df_processed['y_test']), len(df))
        
        # Vérifier que les shapes correspondent
        self.assertEqual(df_processed['X_train'].shape[0], len(df_processed['y_train']))
        self.assertEqual(df_processed['X_val'].shape[0], len(df_processed['y_val']))
        self.assertEqual(df_processed['X_test'].shape[0], len(df_processed['y_test']))

class TestErrorHandling(unittest.TestCase):
    """
    Tests de gestion d'erreurs
    """
    
    def setUp(self):
        """Configuration pour les tests d'erreurs"""
        self.preprocessor = DrillingDataPreprocessor()
        self.feature_engineer = FeatureEngineer()
    
    def test_empty_dataframe(self):
        """Test avec DataFrame vide"""
        empty_df = pd.DataFrame()
        
        # Ne devrait pas lever d'exception
        result = self.preprocessor.handle_missing_values(empty_df)
        self.assertTrue(result.empty)
    
    def test_single_column_dataframe(self):
        """Test avec DataFrame à une seule colonne"""
        single_col_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        
        # Normalisation avec une seule colonne
        normalized_df, scaler = self.preprocessor.normalize_data(single_col_df)
        
        self.assertIsNotNone(normalized_df)
        self.assertIsNotNone(scaler)
        self.assertEqual(normalized_df.shape, single_col_df.shape)
    
    def test_all_missing_values(self):
        """Test avec toutes les valeurs manquantes dans une colonne"""
        df_all_nan = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [np.nan, np.nan, np.nan]
        })
        
        result = self.preprocessor.handle_missing_values(df_all_nan)
        
        # La colonne B devrait être remplie avec une valeur par défaut
        self.assertFalse(result['B'].isnull().any())
    
    def test_invalid_timestamp_column(self):
        """Test avec colonne timestamp invalide"""
        df = pd.DataFrame({
            'invalid_timestamp': ['not', 'a', 'timestamp'],
            'value': [1, 2, 3]
        })
        
        # Devrait gérer gracieusement l'erreur
        try:
            result = self.feature_engineer.create_temporal_features(df, 'invalid_timestamp')
            # Si ça marche, vérifier qu'on a le DataFrame original
            self.assertEqual(len(result), len(df))
        except Exception:
            # L'exception est acceptable pour des données invalides
            pass
    
    def test_zero_division_features(self):
        """Test avec des valeurs qui pourraient causer une division par zéro"""
        df_with_zeros = pd.DataFrame({
            'WOB': [0, 10, 20],
            'RPM': [100, 0, 120],
            'ROP': [5, 8, 0]
        })
        
        # Ne devrait pas lever d'exception
        result = self.feature_engineer.create_drilling_efficiency_features(df_with_zeros)
        
        self.assertIsNotNone(result)
        # Vérifier que les features ont été créées sans NaN ou Inf
        efficiency_cols = [col for col in result.columns if 'efficiency' in col.lower()]
        for col in efficiency_cols:
            self.assertFalse(np.isinf(result[col]).any())

# Suite de tests pour les métriques personnalisées
class TestCustomMetrics(unittest.TestCase):
    """
    Tests pour les métriques personnalisées du domaine forage
    """
    
    def setUp(self):
        """Configuration des données de test"""
        np.random.seed(42)
        self.y_true_regression = np.array([10, 15, 20, 25, 30])
        self.y_pred_regression = np.array([12, 14, 22, 24, 32])
        
        self.y_true_classification = np.array([0, 1, 0, 1, 0])
        self.y_pred_classification = np.array([0, 1, 1, 1, 0])
        self.y_proba_classification = np.array([0.1, 0.9, 0.6, 0.8, 0.2])
    
    def test_drilling_efficiency_score(self):
        """Test du score d'efficacité de forage"""
        from src.utils.metrics import RegressionMetrics
        
        score = RegressionMetrics.drilling_efficiency_score(
            self.y_true_regression, self.y_pred_regression, target_rop=20.0
        )
        
        self.assertIsInstance(score, float)
        self.assertTrue(0 <= score <= 1)
    
    def test_rop_prediction_accuracy(self):
        """Test de précision de prédiction ROP"""
        from src.utils.metrics import DrillingSpecificMetrics
        
        accuracy = DrillingSpecificMetrics.rop_prediction_accuracy(
            self.y_true_regression, self.y_pred_regression, tolerance_percentage=15.0
        )
        
        self.assertIsInstance(accuracy, float)
        self.assertTrue(0 <= accuracy <= 100)
    
    def test_kick_detection_metrics(self):
        """Test des métriques de détection de kick"""
        from src.utils.metrics import ClassificationMetrics
        
        metrics = ClassificationMetrics.kick_detection_metrics(
            self.y_true_classification, self.y_pred_classification, self.y_proba_classification
        )
        
        # Vérifier que toutes les métriques attendues sont présentes
        expected_metrics = ['detection_rate', 'false_alarm_rate', 'specificity', 
                          'kick_precision', 'kick_f1', 'weighted_cost']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))

# Configuration des tests à exécuter
def create_test_suite():
    """
    Crée une suite de tests complète
    
    Returns:
        Test suite
    """
    test_suite = unittest.TestSuite()
    
    # Tests de base
    test_suite.addTest(unittest.makeSuite(TestDrillingDataLoader))
    test_suite.addTest(unittest.makeSuite(TestDrillingDataPreprocessor))
    test_suite.addTest(unittest.makeSuite(TestFeatureEngineer))
    
    # Tests d'intégration
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Tests de gestion d'erreurs
    test_suite.addTest(unittest.makeSuite(TestErrorHandling))
    
    # Tests des métriques
    test_suite.addTest(unittest.makeSuite(TestCustomMetrics))
    
    return test_suite

if __name__ == '__main__':
    # Configuration du logging pour les tests
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Lancer les tests
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    result = runner.run(suite)
    
    # Afficher un résumé
    print(f"\n{'='*50}")
    print(f"RÉSUMÉ DES TESTS")
    print(f"{'='*50}")
    print(f"Tests exécutés: {result.testsRun}")
    print(f"Succès: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Échecs: {len(result.failures)}")
    print(f"Erreurs: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ ÉCHECS:")
        for test, traceback in result.failures:
            print(f"  • {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n🚨 ERREURS:")
        for test, traceback in result.errors:
            print(f"  • {test}: {traceback.split('Exception:')[-1].strip()}")
    
    if result.wasSuccessful():
        print(f"\n✅ TOUS LES TESTS ONT RÉUSSI!")
    else:
        print(f"\n❌ CERTAINS TESTS ONT ÉCHOUÉ")
    
    print(f"{'='*50}")
    
    # Code de sortie approprié pour CI/CD
    exit(0 if result.wasSuccessful() else 1)