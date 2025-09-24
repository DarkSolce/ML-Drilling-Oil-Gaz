"""
Data Loader Module for ML Drilling Project
==========================================

Ce module gère le chargement et la validation des données de forage
depuis différentes sources (CSV, bases de données, APIs temps réel).
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import sqlite3
from datetime import datetime, timedelta
import warnings
import json
import requests
from urllib.parse import urlparse
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DrillingDataLoader:
    """
    Classe principale pour le chargement des données de forage
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise le data loader avec la configuration
        
        Args:
            config: Dictionnaire de configuration
        """
        self.config = config or {}
        self.data_path = Path(self.config.get('data_path', 'data/'))
        self.raw_path = self.data_path / 'raw'
        self.processed_path = self.data_path / 'processed'
        self.external_path = self.data_path / 'external'
        
        # Créer les dossiers s'ils n'existent pas
        for path in [self.raw_path, self.processed_path, self.external_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Configuration des colonnes attendues
        self.formation_columns = [
            'Depth', 'FormationPressure', 'MudWeight', 'Temperature',
            'Porosity', 'Permeability', 'RockType', 'WellboreAngle'
        ]
        
        self.kick_columns = [
            'Timestamp', 'FlowRateIn', 'FlowRateOut', 'StandpipePressure',
            'CasingPressure', 'MudWeight', 'HookLoad', 'RPM', 'Torque',
            'ROP', 'Kick'
        ]
        
        logger.info(f"DataLoader initialisé avec data_path: {self.data_path}")
    
    def load_formation_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Charge les données de formation/pression
        
        Args:
            file_path: Chemin vers le fichier (optionnel)
            
        Returns:
            DataFrame avec les données de formation
        """
        if file_path is None:
            file_path = self.raw_path / 'FormationChangeData.csv'
        
        try:
            # Détecter l'encodage du fichier
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    logger.info(f"Chargement depuis API: {api_url}")
            
            response = requests.get(api_url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Convertir en DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                else:
                    df = pd.DataFrame([data])
            else:
                raise ValueError("Format de données API non supporté")
            
            logger.info(f"Données chargées depuis API: {len(df)} lignes")
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement depuis l'API: {e}")
            raise
    
    def load_from_excel(self, file_path: str, sheet_name: str = None) -> pd.DataFrame:
        """
        Charge des données depuis un fichier Excel
        
        Args:
            file_path: Chemin vers le fichier Excel
            sheet_name: Nom de la feuille (optionnel)
            
        Returns:
            DataFrame avec les données
        """
        try:
            logger.info(f"Chargement depuis Excel: {file_path}")
            
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(file_path)
            
            logger.info(f"Données Excel chargées: {len(df)} lignes")
            return self._basic_cleaning(df)
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement Excel: {e}")
            raise
    
    def load_las_file(self, file_path: str) -> pd.DataFrame:
        """
        Charge un fichier LAS (Log ASCII Standard) - format standard pour logs de puits
        
        Args:
            file_path: Chemin vers le fichier LAS
            
        Returns:
            DataFrame avec les données de log
        """
        try:
            import lasio
            
            logger.info(f"Chargement fichier LAS: {file_path}")
            
            las = lasio.read(file_path)
            df = las.df().reset_index()
            
            logger.info(f"Fichier LAS chargé: {len(df)} lignes, {len(df.columns)} colonnes")
            
            # Ajouter des métadonnées LAS
            df.attrs['well_name'] = las.well.WELL.value if hasattr(las.well, 'WELL') else 'Unknown'
            df.attrs['location'] = las.well.LOC.value if hasattr(las.well, 'LOC') else 'Unknown'
            
            return self._basic_cleaning(df)
            
        except ImportError:
            logger.error("Module 'lasio' non disponible. Installez avec: pip install lasio")
            raise
        except Exception as e:
            logger.error(f"Erreur lors du chargement LAS: {e}")
            raise
    
    def load_real_time_data(self, source_config: Dict) -> pd.DataFrame:
        """
        Charge des données temps réel depuis une source configurée
        
        Args:
            source_config: Configuration de la source de données
            
        Returns:
            DataFrame avec données temps réel
        """
        try:
            source_type = source_config.get('type', 'api')
            
            if source_type == 'api':
                return self.load_from_api(
                    source_config['url'],
                    source_config.get('params', {}),
                    source_config.get('headers', {})
                )
            elif source_type == 'database':
                return self.load_from_database(
                    source_config['connection_string'],
                    source_config['query']
                )
            else:
                raise ValueError(f"Type de source non supporté: {source_type}")
                
        except Exception as e:
            logger.error(f"Erreur chargement temps réel: {e}")
            raise
    
    def stream_data(self, source_config: Dict, callback_func, batch_size: int = 100):
        """
        Stream des données en temps réel avec callback
        
        Args:
            source_config: Configuration de la source
            callback_func: Fonction appelée pour chaque batch
            batch_size: Taille des batches
        """
        try:
            logger.info("Démarrage du streaming de données...")
            
            while True:
                # Charger un batch de données
                df_batch = self.load_real_time_data(source_config)
                
                if len(df_batch) > 0:
                    # Traiter par batches
                    for i in range(0, len(df_batch), batch_size):
                        batch = df_batch.iloc[i:i+batch_size]
                        callback_func(batch)
                
                #
    
    def load_multiple_files(self, file_pattern: str, 
                          data_type: str = 'auto') -> pd.DataFrame:
        """
        Charge et combine plusieurs fichiers
        
        Args:
            file_pattern: Pattern des fichiers (ex: "data_*.csv")
            data_type: Type de données ('formation', 'kick', 'auto')
            
        Returns:
            DataFrame combiné
        """
        try:
            files = list(self.raw_path.glob(file_pattern))
            if not files:
                raise FileNotFoundError(f"Aucun fichier trouvé pour le pattern: {file_pattern}")
            
            logger.info(f"Chargement de {len(files)} fichiers...")
            
            dfs = []
            for file_path in files:
                logger.info(f"Chargement: {file_path.name}")
                
                if data_type == 'formation' or (data_type == 'auto' and 'formation' in file_path.name.lower()):
                    df = self.load_formation_data(file_path)
                elif data_type == 'kick' or (data_type == 'auto' and 'kick' in file_path.name.lower()):
                    df = self.load_kick_detection_data(file_path)
                else:
                    # Chargement générique
                    df = pd.read_csv(file_path)
                    df = self._basic_cleaning(df)
                
                # Ajouter une colonne source
                df['source_file'] = file_path.name
                dfs.append(df)
            
            # Combiner tous les DataFrames
            combined_df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Fichiers combinés: {len(combined_df)} lignes totales")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement multiple: {e}")
            raise
    
    def _basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoyage basique des données
        
        Args:
            df: DataFrame à nettoyer
            
        Returns:
            DataFrame nettoyé
        """
        df_clean = df.copy()
        
        # Supprimer les lignes complètement vides
        df_clean = df_clean.dropna(how='all')
        
        # Nettoyer les noms de colonnes
        df_clean.columns = df_clean.columns.str.strip()
        df_clean.columns = df_clean.columns.str.replace(' ', '_')
        
        # Convertir les colonnes numériques
        numeric_columns = df_clean.select_dtypes(include=['object']).columns
        for col in numeric_columns:
            # Essayer de convertir en numérique
            try:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
            except:
                pass
        
        # Supprimer les duplicatas complets
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed_duplicates = initial_rows - len(df_clean)
        
        if removed_duplicates > 0:
            logger.info(f"Supprimé {removed_duplicates} duplicatas")
        
        return df_clean
    
    def _process_timestamp_column(self, df: pd.DataFrame, 
                                col_name: str) -> pd.DataFrame:
        """
        Traite la colonne timestamp
        
        Args:
            df: DataFrame
            col_name: Nom de la colonne timestamp
            
        Returns:
            DataFrame avec timestamp traité
        """
        df_processed = df.copy()
        
        try:
            # Essayer plusieurs formats de date
            date_formats = [
                '%Y-%m-%d %H:%M:%S',
                '%d/%m/%Y %H:%M:%S', 
                '%m/%d/%Y %H:%M:%S',
                '%Y-%m-%d',
                '%d/%m/%Y',
                '%m/%d/%Y'
            ]
            
            for fmt in date_formats:
                try:
                    df_processed[col_name] = pd.to_datetime(df_processed[col_name], format=fmt)
                    logger.info(f"Timestamp parsé avec format: {fmt}")
                    break
                except:
                    continue
            else:
                # Si aucun format ne marche, essayer la détection automatique
                df_processed[col_name] = pd.to_datetime(df_processed[col_name], errors='coerce')
                
        except Exception as e:
            logger.warning(f"Impossible de traiter la colonne timestamp: {e}")
        
        return df_processed
    
    def save_processed_data(self, df: pd.DataFrame, filename: str, 
                          format: str = 'csv') -> str:
        """
        Sauvegarde les données traitées
        
        Args:
            df: DataFrame à sauvegarder
            filename: Nom du fichier
            format: Format ('csv', 'parquet', 'json')
            
        Returns:
            Chemin du fichier sauvegardé
        """
        if format == 'csv':
            file_path = self.processed_path / f"{filename}.csv"
            df.to_csv(file_path, index=False)
        elif format == 'parquet':
            file_path = self.processed_path / f"{filename}.parquet"
            df.to_parquet(file_path, index=False)
        elif format == 'json':
            file_path = self.processed_path / f"{filename}.json"
            df.to_json(file_path, orient='records', indent=2)
        else:
            raise ValueError(f"Format non supporté: {format}")
        
        logger.info(f"Données sauvegardées: {file_path}")
        return str(file_path)
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Génère un résumé des données
        
        Args:
            df: DataFrame à analyser
            
        Returns:
            Dictionnaire avec le résumé
        """
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns),
            'datetime_columns': list(df.select_dtypes(include=['datetime64']).columns)
        }
        
        # Statistiques pour colonnes numériques
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            summary['numeric_stats'] = {
                'mean': numeric_df.mean().to_dict(),
                'std': numeric_df.std().to_dict(),
                'min': numeric_df.min().to_dict(),
                'max': numeric_df.max().to_dict(),
                'median': numeric_df.median().to_dict()
            }
        
        return summary
    
    def validate_data_quality(self, df: pd.DataFrame, 
                            data_type: str = 'general') -> Dict:
        """
        Valide la qualité des données
        
        Args:
            df: DataFrame à valider
            data_type: Type de données ('formation', 'kick', 'general')
            
        Returns:
            Rapport de validation
        """
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'data_type': data_type,
            'total_rows': len(df),
            'issues': [],
            'warnings': [],
            'score': 0
        }
        
        # Vérifications générales
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_percentage > 10:
            validation_report['issues'].append(f"Trop de valeurs manquantes: {missing_percentage:.1f}%")
        elif missing_percentage > 5:
            validation_report['warnings'].append(f"Valeurs manquantes modérées: {missing_percentage:.1f}%")
        
        # Duplicatas
        duplicates = len(df) - len(df.drop_duplicates())
        if duplicates > 0:
            duplicate_percentage = (duplicates / len(df)) * 100
            if duplicate_percentage > 5:
                validation_report['issues'].append(f"Duplicatas: {duplicates} lignes ({duplicate_percentage:.1f}%)")
            else:
                validation_report['warnings'].append(f"Quelques duplicatas: {duplicates} lignes")
        
        # Vérifications spécifiques par type
        if data_type == 'formation':
            required_cols = ['Depth', 'FormationPressure', 'MudWeight']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                validation_report['issues'].append(f"Colonnes requises manquantes: {missing_cols}")
            
            # Vérifier la cohérence des valeurs
            if 'Depth' in df.columns:
                if (df['Depth'] < 0).any():
                    validation_report['issues'].append("Profondeurs négatives détectées")
                if (df['Depth'] > 15000).any():
                    validation_report['warnings'].append("Profondeurs très élevées (>15km)")
        
        elif data_type == 'kick':
            required_cols = ['FlowRateIn', 'FlowRateOut', 'Kick']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                validation_report['issues'].append(f"Colonnes requises manquantes: {missing_cols}")
            
            # Vérifier l'équilibre des classes
            if 'Kick' in df.columns:
                kick_rate = df['Kick'].mean()
                if kick_rate < 0.01:
                    validation_report['warnings'].append(f"Très peu de kicks: {kick_rate*100:.2f}%")
                elif kick_rate > 0.20:
                    validation_report['warnings'].append(f"Taux de kicks élevé: {kick_rate*100:.1f}%")
        
        # Calcul du score de qualité
        base_score = 100
        base_score -= len(validation_report['issues']) * 15
        base_score -= len(validation_report['warnings']) * 5
        base_score = max(0, min(100, base_score))
        validation_report['score'] = base_score
        
        return validation_report
    
    def create_data_profile(self, df: pd.DataFrame, 
                          output_path: Optional[str] = None) -> str:
        """
        Crée un profil détaillé des données
        
        Args:
            df: DataFrame à profiler
            output_path: Chemin de sortie du rapport
            
        Returns:
            Chemin du rapport généré
        """
        try:
           import ProfileReport
            
            # Générer le profil
            profile = ydata-profiling.ProfileReport(
                df, 
                title="Drilling Data Profile Report",
                explorative=True,
                dark_mode=False
            )
            
            # Définir le chemin de sortie
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = self.processed_path / f"data_profile_{timestamp}.html"
            
            # Sauvegarder
            profile.to_file(output_path)
            logger.info(f"Profil des données généré: {output_path}")
            
            return str(output_path)
            
        except ImportError:
            logger.warning("pandas_profiling non disponible. Utilisez: pip install pandas-profiling")
            
            # Générer un rapport basique en HTML
            summary = self.get_data_summary(df)
            validation = self.validate_data_quality(df)
            
            html_content = self._generate_basic_html_report(df, summary, validation)
            
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = self.processed_path / f"data_summary_{timestamp}.html"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Rapport basique généré: {output_path}")
            return str(output_path)
    
    def _generate_basic_html_report(self, df: pd.DataFrame, 
                                  summary: Dict, validation: Dict) -> str:
        """Génère un rapport HTML basique"""
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Drilling Data Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; 
                          background-color: #f8f9fa; border-radius: 5px; }}
                .issue {{ color: red; }}
                .warning {{ color: orange; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🛢️ Drilling Data Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📊 Data Overview</h2>
                <div class="metric">
                    <strong>Rows:</strong> {summary['shape'][0]:,}
                </div>
                <div class="metric">
                    <strong>Columns:</strong> {summary['shape'][1]}
                </div>
                <div class="metric">
                    <strong>Memory:</strong> {summary['memory_usage_mb']:.1f} MB
                </div>
                <div class="metric">
                    <strong>Quality Score:</strong> {validation['score']}/100
                </div>
            </div>
            
            <div class="section">
                <h2>🔍 Data Quality</h2>
                {'<h3 class="issue">Issues:</h3><ul>' + ''.join([f'<li class="issue">{issue}</li>' for issue in validation['issues']]) + '</ul>' if validation['issues'] else ''}
                {'<h3 class="warning">Warnings:</h3><ul>' + ''.join([f'<li class="warning">{warning}</li>' for warning in validation['warnings']]) + '</ul>' if validation['warnings'] else ''}
            </div>
            
            <div class="section">
                <h2>📈 Column Statistics</h2>
                <table>
                    <tr><th>Column</th><th>Type</th><th>Missing</th><th>Unique</th></tr>
                    {''.join([f'<tr><td>{col}</td><td>{str(dtype)}</td><td>{summary["missing_values"].get(col, 0)}</td><td>{df[col].nunique() if col in df.columns else "N/A"}</td></tr>' for col, dtype in summary['dtypes'].items()])}
                </table>
            </div>
            
        </body>
        </html>
        """
        
        return html_template

# Fonctions utilitaires
def quick_load_drilling_data(data_type: str = 'synthetic', **kwargs) -> pd.DataFrame:
    """
    Fonction de chargement rapide pour les tests
    
    Args:
        data_type: 'formation', 'kick', 'synthetic'
        **kwargs: Arguments pour le loader
        
    Returns:
        DataFrame chargé
    """
    loader = DrillingDataLoader()
    
    if data_type == 'formation':
        return loader.load_formation_data(**kwargs)
    elif data_type == 'kick':
        return loader.load_kick_detection_data(**kwargs)
    elif data_type == 'synthetic':
        n_samples = kwargs.get('n_samples', 1000)
        return loader.load_synthetic_drilling_data(n_samples)
    else:
        raise ValueError(f"Type de données non supporté: {data_type}")

def validate_drilling_dataset(df: pd.DataFrame, 
                            expected_type: str = 'auto') -> bool:
    """
    Validation rapide d'un dataset de forage
    
    Args:
        df: DataFrame à valider
        expected_type: Type attendu ('formation', 'kick', 'auto')
        
    Returns:
        True si valide
    """
    loader = DrillingDataLoader()
    
    if expected_type == 'auto':
        # Détection automatique du type
        if 'FormationPressure' in df.columns:
            expected_type = 'formation'
        elif 'Kick' in df.columns:
            expected_type = 'kick'
        else:
            expected_type = 'general'
    
    validation = loader.validate_data_quality(df, expected_type)
    
    # Critères de validation
    is_valid = (
        validation['score'] >= 70 and
        len(validation['issues']) == 0 and
        validation['total_rows'] > 10
    )
    
    if not is_valid:
        logger.warning(f"Dataset invalide - Score: {validation['score']}, Issues: {len(validation['issues'])}")
    
    return is_valid

# Point d'entrée pour tests
if __name__ == "__main__":
    # Tests basiques
    loader = DrillingDataLoader()
    
    print("🧪 Test de génération de données synthétiques...")
    synthetic_data = loader.load_synthetic_drilling_data(n_samples=1000)
    print(f"✅ Généré: {len(synthetic_data)} lignes")
    
    print("\n📊 Résumé des données:")
    summary = loader.get_data_summary(synthetic_data)
    print(f"  • Shape: {summary['shape']}")
    print(f"  • Colonnes numériques: {len(summary['numeric_columns'])}")
    print(f"  • Valeurs manquantes: {sum(summary['missing_values'].values())}")
    
    print("\n🔍 Validation des données:")
    validation = loader.validate_data_quality(synthetic_data, 'kick')
    print(f"  • Score de qualité: {validation['score']}/100")
    print(f"  • Issues: {len(validation['issues'])}")
    print(f"  • Warnings: {len(validation['warnings'])}")
    
    print("\n💾 Sauvegarde des données de test...")
    saved_path = loader.save_processed_data(synthetic_data, "test_synthetic_data")
    print(f"✅ Sauvegardé: {saved_path}")
    
    print("\n🎉 Tests terminés avec succès!")Fichier chargé avec encodage: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Impossible de décoder le fichier avec les encodages testés")
            
            logger.info(f"Données de formation chargées: {len(df)} lignes, {len(df.columns)} colonnes")
            
            # Validation des colonnes
            missing_cols = [col for col in ['Depth', 'FormationPressure', 'MudWeight'] 
                          if col not in df.columns]
            if missing_cols:
                logger.warning(f"Colonnes manquantes: {missing_cols}")
            
            # Nettoyage basique
            df = self._basic_cleaning(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données de formation: {e}")
            raise
    
    def load_kick_detection_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Charge les données de détection de kicks
        
        Args:
            file_path: Chemin vers le fichier (optionnel)
            
        Returns:
            DataFrame avec les données de kick
        """
        if file_path is None:
            file_path = self.raw_path / 'Kick_Detection_Data2.csv'
        
        try:
            # Charger avec gestion d'encodage
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    logger.info(f"Fichier chargé avec encodage: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Impossible de décoder le fichier")
            
            logger.info(f"Données de kick chargées: {len(df)} lignes, {len(df.columns)} colonnes")
            
            # Validation de la colonne cible
            if 'Kick' not in df.columns:
                logger.warning("Colonne 'Kick' manquante dans les données")
                # Essayer des noms alternatifs
                alt_names = ['kick', 'KICK', 'Kick_Flag', 'IsKick']
                for alt_name in alt_names:
                    if alt_name in df.columns:
                        df = df.rename(columns={alt_name: 'Kick'})
                        logger.info(f"Colonne '{alt_name}' renommée en 'Kick'")
                        break
            
            # Traitement de la colonne timestamp si présente
            if 'Timestamp' in df.columns:
                df = self._process_timestamp_column(df, 'Timestamp')
            
            # Nettoyage basique
            df = self._basic_cleaning(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données de kick: {e}")
            raise
    
    def load_synthetic_drilling_data(self, n_samples: int = 10000, 
                                   random_seed: int = 42) -> pd.DataFrame:
        """
        Génère des données de forage synthétiques pour les tests
        
        Args:
            n_samples: Nombre d'échantillons à générer
            random_seed: Graine pour la reproductibilité
            
        Returns:
            DataFrame avec données synthétiques
        """
        logger.info(f"Génération de {n_samples} échantillons synthétiques")
        
        np.random.seed(random_seed)
        
        # Génération des variables de base avec corrélations réalistes
        data = {}
        
        # Timestamp
        start_date = datetime(2024, 1, 1)
        data['Timestamp'] = pd.date_range(start_date, periods=n_samples, freq='1min')
        
        # Profondeur progressive
        data['Depth'] = np.cumsum(np.random.exponential(0.5, n_samples))
        
        # Paramètres de forage avec corrélations
        base_wob = 25
        data['WOB'] = np.maximum(0, np.random.normal(base_wob, 8, n_samples))
        
        base_rpm = 120
        data['RPM'] = np.maximum(0, np.random.normal(base_rpm, 30, n_samples))
        
        # FlowRate corrélé avec WOB
        data['FlowRate'] = np.maximum(0, 300 + data['WOB'] * 2 + np.random.normal(0, 50, n_samples))
        
        # MudWeight qui augmente avec la profondeur
        data['MudWeight'] = 1.0 + data['Depth'] * 0.0002 + np.random.normal(0, 0.1, n_samples)
        data['MudWeight'] = np.clip(data['MudWeight'], 0.8, 2.5)
        
        # HookLoad corrélé avec WOB et profondeur
        data['HookLoad'] = 150 + data['WOB'] * 2 + data['Depth'] * 0.01 + np.random.normal(0, 20, n_samples)
        
        # Pression standpipe corrélée avec flow rate et mud weight
        data['StandpipePressure'] = (data['FlowRate'] * 0.3 + 
                                   data['MudWeight'] * 50 + 
                                   np.random.normal(0, 30, n_samples))
        data['StandpipePressure'] = np.maximum(0, data['StandpipePressure'])
        
        # Torque corrélé avec WOB et RPM
        data['Torque'] = (data['WOB'] * 0.8 + 
                         data['RPM'] * 0.1 + 
                         np.random.normal(0, 5, n_samples))
        data['Torque'] = np.maximum(0, data['Torque'])
        
        # ROP (Rate of Penetration) - variable cible principale
        rop_base = (data['WOB'] * 0.3 + 
                   data['RPM'] * 0.05 + 
                   data['FlowRate'] * 0.01 - 
                   data['MudWeight'] * 5 +
                   np.random.normal(0, 3, n_samples))
        data['ROP'] = np.maximum(0, rop_base)
        
        # Formation Pressure corrélée avec profondeur
        data['FormationPressure'] = (data['Depth'] * 0.52 +  # Gradient hydrost. normal
                                    np.random.normal(0, 200, n_samples))
        data['FormationPressure'] = np.maximum(0, data['FormationPressure'])
        
        # FlowRateOut légèrement différent de FlowRateIn
        data['FlowRateOut'] = data['FlowRate'] + np.random.normal(0, 5, n_samples)
        data['FlowRateOut'] = np.maximum(0, data['FlowRateOut'])
        
        # CasingPressure
        data['CasingPressure'] = np.maximum(0, np.random.normal(50, 15, n_samples))
        
        # Variables géologiques
        data['Temperature'] = 20 + data['Depth'] * 0.025 + np.random.normal(0, 5, n_samples)
        data['Porosity'] = np.clip(np.random.beta(2, 5, n_samples), 0.05, 0.4)
        data['Permeability'] = np.random.lognormal(2, 2, n_samples)
        
        # Génération des événements de kick (5% de probabilité)
        kick_probability = 0.05
        kick_events = np.random.binomial(1, kick_probability, n_samples)
        
        # Les kicks affectent le flow rate out
        kick_indices = np.where(kick_events == 1)[0]
        for idx in kick_indices:
            # Augmenter le flow rate out lors d'un kick
            data['FlowRateOut'][idx] += np.random.normal(20, 5)
            # Légère augmentation de la pression casing
            data['CasingPressure'][idx] += np.random.normal(10, 3)
        
        data['Kick'] = kick_events
        
        # Créer le DataFrame
        df = pd.DataFrame(data)
        
        # Ajouter quelques valeurs manquantes de façon réaliste
        missing_rate = 0.02  # 2% de valeurs manquantes
        for col in ['Temperature', 'Porosity', 'Permeability']:
            missing_idx = np.random.choice(n_samples, 
                                         size=int(n_samples * missing_rate), 
                                         replace=False)
            df.loc[missing_idx, col] = np.nan
        
        # Renommage pour cohérence
        df = df.rename(columns={
            'FlowRate': 'FlowRateIn'
        })
        
        logger.info(f"✅ Données synthétiques générées: {len(df)} échantillons")
        logger.info(f"📊 Kicks générés: {df['Kick'].sum()} ({df['Kick'].mean()*100:.1f}%)")
        
        return df
    
    def load_from_database(self, connection_string: str, query: str) -> pd.DataFrame:
        """
        Charge des données depuis une base de données
        
        Args:
            connection_string: Chaîne de connexion à la DB
            query: Requête SQL
            
        Returns:
            DataFrame avec les données
        """
        try:
            logger.info(f"Connexion à la base de données...")
            
            if connection_string.startswith('sqlite'):
                # SQLite
                db_path = connection_string.replace('sqlite:///', '')
                conn = sqlite3.connect(db_path)
            else:
                # PostgreSQL, MySQL, etc.
                import sqlalchemy
                engine = sqlalchemy.create_engine(connection_string)
                conn = engine
            
            df = pd.read_sql_query(query, conn)
            
            if hasattr(conn, 'close'):
                conn.close()
            
            logger.info(f"Données chargées depuis DB: {len(df)} lignes")
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement depuis la DB: {e}")
            raise
    
    def load_from_api(self, api_url: str, params: Dict = None, 
                     headers: Dict = None) -> pd.DataFrame:
        """
        Charge des données depuis une API REST
        
        Args:
            api_url: URL de l'API
            params: Paramètres de requête
            headers: En-têtes HTTP
            
        Returns:
            DataFrame avec les données
        """
        try:
            logger.info(f", na=False).any():
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
                except:
                    pass
        
        # Supprimer les duplicatas complets
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed_duplicates = initial_rows - len(df_clean)
        
        if removed_duplicates > 0:
            logger.info(f"Supprimé {removed_duplicates} duplicatas")
        
        return df_clean
    
    def _process_timestamp_column(self, df: pd.DataFrame, 
                                col_name: str) -> pd.DataFrame:
        """
        Traite la colonne timestamp
        
        Args:
            df: DataFrame
            col_name: Nom de la colonne timestamp
            
        Returns:
            DataFrame avec timestamp traité
        """
        df_processed = df.copy()
        
        try:
            # Essayer plusieurs formats de date
            date_formats = [
                '%Y-%m-%d %H:%M:%S',
                '%d/%m/%Y %H:%M:%S', 
                '%m/%d/%Y %H:%M:%S',
                '%Y-%m-%d',
                '%d/%m/%Y',
                '%m/%d/%Y',
                '%Y%m%d',
                '%d%m%Y',
                'ISO8601'
            ]
            
            for fmt in date_formats:
                try:
                    if fmt == 'ISO8601':
                        df_processed[col_name] = pd.to_datetime(df_processed[col_name], 
                                                              format='mixed', errors='coerce')
                    else:
                        df_processed[col_name] = pd.to_datetime(df_processed[col_name], 
                                                              format=fmt, errors='raise')
                    logger.info(f"Timestamp parsé avec format: {fmt}")
                    break
                except:
                    continue
            else:
                # Si aucun format ne marche, essayer la détection automatique
                df_processed[col_name] = pd.to_datetime(df_processed[col_name], 
                                                      errors='coerce', infer_datetime_format=True)
                logger.info("Timestamp parsé avec détection automatique")
                
        except Exception as e:
            logger.warning(f"Impossible de traiter la colonne timestamp: {e}")
        
        # Vérifier le résultat
        if df_processed[col_name].isnull().all():
            logger.warning(f"Aucune date valide trouvée dans la colonne {col_name}")
        else:
            valid_dates = df_processed[col_name].count()
            total_dates = len(df_processed)
            logger.info(f"Dates valides: {valid_dates}/{total_dates} ({valid_dates/total_dates*100:.1f}%)")
        
        return df_processed
    
    def save_processed_data(self, df: pd.DataFrame, filename: str, 
                          format: str = 'csv', compression: str = None) -> str:
        """
        Sauvegarde les données traitées
        
        Args:
            df: DataFrame à sauvegarder
            filename: Nom du fichier
            format: Format ('csv', 'parquet', 'json', 'excel')
            compression: Type de compression ('gzip', 'bz2', 'xz')
            
        Returns:
            Chemin du fichier sauvegardé
        """
        # S'assurer que le dossier existe
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        # Déterminer l'extension
        extensions = {
            'csv': '.csv',
            'parquet': '.parquet',
            'json': '.json',
            'excel': '.xlsx'
        }
        
        if format not in extensions:
            raise ValueError(f"Format non supporté: {format}")
        
        ext = extensions[format]
        if compression:
            ext += f'.{compression}'
        
        file_path = self.processed_path / f"{filename}{ext}"
        
        try:
            if format == 'csv':
                df.to_csv(file_path, index=False, compression=compression)
            elif format == 'parquet':
                df.to_parquet(file_path, index=False, compression=compression)
            elif format == 'json':
                df.to_json(file_path, orient='records', indent=2)
            elif format == 'excel':
                df.to_excel(file_path, index=False)
            
            logger.info(f"Données sauvegardées: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")
            raise
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Génère un résumé complet des données
        
        Args:
            df: DataFrame à analyser
            
        Returns:
            Dictionnaire avec le résumé détaillé
        """
        summary = {
            'basic_info': {
                'shape': df.shape,
                'columns': list(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'creation_date': datetime.now().isoformat()
            },
            'data_types': {
                'dtypes': df.dtypes.to_dict(),
                'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
                'datetime_columns': list(df.select_dtypes(include=['datetime64']).columns),
                'boolean_columns': list(df.select_dtypes(include=['bool']).columns)
            },
            'missing_data': {
                'total_missing': df.isnull().sum().sum(),
                'missing_by_column': df.isnull().sum().to_dict(),
                'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
                'columns_with_missing': df.columns[df.isnull().any()].tolist()
            },
            'duplicates': {
                'duplicate_rows': df.duplicated().sum(),
                'unique_rows': len(df.drop_duplicates()),
                'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
            }
        }
        
        # Statistiques pour colonnes numériques
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            summary['numeric_stats'] = {
                'count': numeric_df.count().to_dict(),
                'mean': numeric_df.mean().to_dict(),
                'std': numeric_df.std().to_dict(),
                'min': numeric_df.min().to_dict(),
                'max': numeric_df.max().to_dict(),
                'median': numeric_df.median().to_dict(),
                'q25': numeric_df.quantile(0.25).to_dict(),
                'q75': numeric_df.quantile(0.75).to_dict(),
                'skewness': numeric_df.skew().to_dict(),
                'kurtosis': numeric_df.kurtosis().to_dict()
            }
        
        # Statistiques pour colonnes catégorielles
        categorical_df = df.select_dtypes(include=['object', 'category'])
        if not categorical_df.empty:
            summary['categorical_stats'] = {}
            for col in categorical_df.columns:
                value_counts = df[col].value_counts()
                summary['categorical_stats'][col] = {
                    'unique_values': df[col].nunique(),
                    'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                    'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                    'top_5_values': value_counts.head(5).to_dict()
                }
        
        # Analyse des corrélations pour colonnes numériques
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            # Trouver les corrélations les plus élevées
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    corr_val = corr_matrix.iloc[i, j]
                    if not np.isnan(corr_val):
                        corr_pairs.append({
                            'column1': col1,
                            'column2': col2, 
                            'correlation': corr_val
                        })
            
            # Trier par corrélation absolue
            corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
            summary['correlations'] = corr_pairs[:10]  # Top 10 corrélations
        
        return summary
    
    def validate_data_quality(self, df: pd.DataFrame, 
                            data_type: str = 'general',
                            custom_rules: List[Dict] = None) -> Dict:
        """
        Valide la qualité des données avec règles personnalisées
        
        Args:
            df: DataFrame à valider
            data_type: Type de données ('formation', 'kick', 'general')
            custom_rules: Règles de validation personnalisées
            
        Returns:
            Rapport de validation détaillé
        """
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'data_type': data_type,
            'dataset_info': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            },
            'issues': [],
            'warnings': [],
            'passed_checks': [],
            'failed_checks': [],
            'overall_score': 0
        }
        
        # Vérifications générales
        checks_performed = 0
        checks_passed = 0
        
        # 1. Vérification des valeurs manquantes
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        checks_performed += 1
        if missing_percentage < 1:
            validation_report['passed_checks'].append("Très peu de valeurs manquantes")
            checks_passed += 1
        elif missing_percentage < 5:
            validation_report['warnings'].append(f"Valeurs manquantes modérées: {missing_percentage:.1f}%")
            checks_passed += 0.5
        elif missing_percentage < 15:
            validation_report['warnings'].append(f"Beaucoup de valeurs manquantes: {missing_percentage:.1f}%")
        else:
            validation_report['issues'].append(f"Trop de valeurs manquantes: {missing_percentage:.1f}%")
            validation_report['failed_checks'].append("Test valeurs manquantes")
        
        # 2. Vérification des duplicatas
        checks_performed += 1
        duplicates = len(df) - len(df.drop_duplicates())
        duplicate_percentage = (duplicates / len(df)) * 100
        if duplicates == 0:
            validation_report['passed_checks'].append("Aucun duplicata")
            checks_passed += 1
        elif duplicate_percentage < 1:
            validation_report['warnings'].append(f"Quelques duplicatas: {duplicates} lignes")
            checks_passed += 0.5
        else:
            validation_report['issues'].append(f"Beaucoup de duplicatas: {duplicates} lignes ({duplicate_percentage:.1f}%)")
            validation_report['failed_checks'].append("Test duplicatas")
        
        # 3. Vérification de la cohérence des types de données
        checks_performed += 1
        type_issues = []
        for col in df.columns:
            if col.lower() in ['depth', 'pressure', 'weight', 'rate', 'temperature']:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    type_issues.append(f"Colonne '{col}' devrait être numérique")
            elif 'time' in col.lower() or 'date' in col.lower():
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    type_issues.append(f"Colonne '{col}' devrait être datetime")
        
        if not type_issues:
            validation_report['passed_checks'].append("Types de données cohérents")
            checks_passed += 1
        else:
            validation_report['warnings'].extend(type_issues)
        
        # Vérifications spécifiques par type de données
        if data_type == 'formation':
            required_cols = ['Depth', 'FormationPressure', 'MudWeight']
            missing_cols = [col for col in required_cols if col not in df.columns]
            checks_performed += 1
            
            if not missing_cols:
                validation_report['passed_checks'].append("Toutes les colonnes requises présentes")
                checks_passed += 1
            else:
                validation_report['issues'].append(f"Colonnes requises manquantes: {missing_cols}")
                validation_report['failed_checks'].append("Test colonnes requises")
            
            # Vérifier la cohérence des valeurs
            checks_performed += 1
            physical_issues = []
            if 'Depth' in df.columns:
                if (df['Depth'] < 0).any():
                    physical_issues.append("Profondeurs négatives détectées")
                if (df['Depth'] > 15000).any():
                    physical_issues.append("Profondeurs très élevées (>15km) détectées")
            
            if 'FormationPressure' in df.columns:
                if (df['FormationPressure'] < 0).any():
                    physical_issues.append("Pressions négatives détectées")
                if (df['FormationPressure'] > 20000).any():
                    physical_issues.append("Pressions très élevées (>20000 psi) détectées")
            
            if not physical_issues:
                validation_report['passed_checks'].append("Valeurs physiquement cohérentes")
                checks_passed += 1
            else:
                validation_report['warnings'].extend(physical_issues)
        
        elif data_type == 'kick':
            required_cols = ['FlowRateIn', 'FlowRateOut', 'Kick']
            missing_cols = [col for col in required_cols if col not in df.columns]
            checks_performed += 1
            
            if not missing_cols:
                validation_report['passed_checks'].append("Toutes les colonnes requises présentes")
                checks_passed += 1
            else:
                validation_report['issues'].append(f"Colonnes requises manquantes: {missing_cols}")
                validation_report['failed_checks'].append("Test colonnes requises")
            
            # Vérifier l'équilibre des classes
            if 'Kick' in df.columns:
                checks_performed += 1
                kick_rate = df['Kick'].mean()
                if 0.01 <= kick_rate <= 0.15:
                    validation_report['passed_checks'].append(f"Taux de kicks réaliste: {kick_rate*100:.2f}%")
                    checks_passed += 1
                elif kick_rate < 0.005:
                    validation_report['warnings'].append(f"Très peu de kicks: {kick_rate*100:.3f}%")
                elif kick_rate > 0.20:
                    validation_report['warnings'].append(f"Taux de kicks élevé: {kick_rate*100:.1f}%")
        
        # Appliquer les règles personnalisées
        if custom_rules:
            for rule in custom_rules:
                checks_performed += 1
                try:
                    rule_name = rule.get('name', 'Règle personnalisée')
                    condition = rule.get('condition')
                    message = rule.get('message', '')
                    severity = rule.get('severity', 'warning')  # warning, error
                    
                    # Évaluer la condition (attention: eval() est dangereux en production)
                    # En production, utiliser une approche plus sécurisée
                    if eval(condition, {"df": df, "np": np, "pd": pd}):
                        validation_report['passed_checks'].append(f"{rule_name}: {message}")
                        checks_passed += 1
                    else:
                        if severity == 'error':
                            validation_report['issues'].append(f"{rule_name}: {message}")
                            validation_report['failed_checks'].append(rule_name)
                        else:
                            validation_report['warnings'].append(f"{rule_name}: {message}")
                        
                except Exception as e:
                    validation_report['warnings'].append(f"Erreur dans règle '{rule_name}': {e}")
        
        # Calculer le score global
        if checks_performed > 0:
            base_score = (checks_passed / checks_performed) * 100
            # Pénalités
            base_score -= len(validation_report['issues']) * 10
            base_score -= len(validation_report['warnings']) * 3
            validation_report['overall_score'] = max(0, min(100, base_score))
        
        # Ajouter des statistiques de validation
        validation_report['validation_stats'] = {
            'total_checks': checks_performed,
            'passed_checks': int(checks_passed),
            'failed_checks': len(validation_report['failed_checks']),
            'warnings': len(validation_report['warnings']),
            'issues': len(validation_report['issues'])
        }
        
        return validation_report
    
    def create_data_profile(self, df: pd.DataFrame, 
                          output_path: Optional[str] = None,
                          include_correlations: bool = True,
                          include_sample_data: bool = True) -> str:
        """
        Crée un profil détaillé des données au format HTML
        
        Args:
            df: DataFrame à profiler
            output_path: Chemin de sortie du rapport
            include_correlations: Inclure l'analyse de corrélation
            include_sample_data: Inclure des échantillons de données
            
        Returns:
            Chemin du rapport généré
        """
        try:
            # Essayer pandas-profiling d'abord
            try:
                import ydata_profiling
                
                profile = ydata_profiling.ProfileReport(
                    df,
                    title="Drilling Data Profile Report",
                    explorative=True,
                    correlations={
                        "auto": {"calculate": include_correlations},
                        "pearson": {"calculate": include_correlations},
                        "spearman": {"calculate": include_correlations}
                    }
                )
                
                if output_path is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = self.processed_path / f"data_profile_{timestamp}.html"
                
                profile.to_file(output_path)
                logger.info(f"Profil avancé généré: {output_path}")
                return str(output_path)
                
            except ImportError:
                logger.info("ydata-profiling non disponible, génération d'un rapport personnalisé...")
        
        except Exception as e:
            logger.warning(f"Erreur avec pandas-profiling: {e}")
        
        # Générer un rapport HTML personnalisé
        summary = self.get_data_summary(df)
        validation = self.validate_data_quality(df)
        
        html_content = self._generate_comprehensive_html_report(
            df, summary, validation, include_correlations, include_sample_data
        )
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.processed_path / f"data_profile_{timestamp}.html"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Profil personnalisé généré: {output_path}")
        return str(output_path)
    
    def _generate_comprehensive_html_report(self, df: pd.DataFrame, 
                                          summary: Dict, validation: Dict,
                                          include_correlations: bool,
                                          include_sample_data: bool) -> str:
        """Génère un rapport HTML complet et détaillé"""
        
        # Créer des graphiques pour l'inclusion dans le HTML
        import base64
        from io import BytesIO
        import matplotlib
        matplotlib.use('Agg')  # Backend non-interactif
        import matplotlib.pyplot as plt
        
        def fig_to_base64(fig):
            """Convertit une figure matplotlib en base64 pour inclusion HTML"""
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()
            plt.close(fig)
            return image_base64
        
        # Générer graphiques de distribution
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]  # Limiter à 6
        distribution_plots = ""
        
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten() if len(numeric_cols) > 1 else [axes]
            
            for i, col in enumerate(numeric_cols):
                if i < len(axes):
                    axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'Distribution de {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Fréquence')
                    axes[i].grid(True, alpha=0.3)
            
            # Supprimer axes vides
            for i in range(len(numeric_cols), len(axes)):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            distribution_img = fig_to_base64(fig)
            distribution_plots = f'<img src="data:image/png;base64,{distribution_img}" style="max-width: 100%; height: auto;">'
        
        # Générer matrice de corrélation si demandé
        correlation_plot = ""
        if include_correlations and len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = df[numeric_cols].corr()
            
            im = ax.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto')
            ax.set_xticks(range(len(numeric_cols)))
            ax.set_yticks(range(len(numeric_cols)))
            ax.set_xticklabels(numeric_cols, rotation=45)
            ax.set_yticklabels(numeric_cols)
            
            # Ajouter les valeurs de corrélation
            for i in range(len(numeric_cols)):
                for j in range(len(numeric_cols)):
                    text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                  ha="center", va="center", color="black")
            
            ax.set_title('Matrice de Corrélation')
            plt.colorbar(im)
            plt.tight_layout()
            
            correlation_img = fig_to_base64(fig)
            correlation_plot = f'<img src="data:image/png;base64,{correlation_img}" style="max-width: 100%; height: auto;">'
        
        # Échantillon de données
        sample_data_html = ""
        if include_sample_data:
            sample_df = df.head(10)
            sample_data_html = f"""
            <h3>🔍 Échantillon de Données (10 premières lignes)</h3>
            <div style="overflow-x: auto;">
                {sample_df.to_html(classes='table table-striped', table_id='sample-data')}
            </div>
            """
        
        # Construire le HTML complet
        html_template = f"""
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Profil des Données de Forage</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    color: #333;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 15px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                    overflow: hidden;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 40px;
                    text-align: center;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 2.5em;
                    font-weight: 300;
                }}
                .header p {{
                    margin: 10px 0 0 0;
                    font-size: 1.1em;
                    opacity: 0.9;
                }}
                .section {{
                    padding: 30px 40px;
                    border-bottom: 1px solid #eee;
                }}
                .section:last-child {{
                    border-bottom: none;
                }}
                .section h2 {{
                    color: #667eea;
                    margin-bottom: 20px;
                    font-size: 1.8em;
                    border-bottom: 2px solid #667eea;
                    padding-bottom: 10px;
                }}
                .metric-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;