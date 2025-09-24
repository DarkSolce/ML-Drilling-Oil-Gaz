"""
Visualization Module for ML Drilling Project
==========================================

Module de visualisation pour les données et résultats de forage
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Configuration des styles
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class DrillingDataVisualizer:
    """
    Classe principale pour la visualisation des données de forage
    """
    
    def __init__(self, style: str = 'seaborn'):
        """
        Initialise le visualiseur
        
        Args:
            style: Style matplotlib à utiliser
        """
        plt.style.use(style)
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def plot_drilling_parameters_evolution(self, df: pd.DataFrame, 
                                         time_col: str = 'Timestamp',
                                         save_path: Optional[str] = None) -> None:
        """
        Visualise l'évolution des paramètres de forage dans le temps
        
        Args:
            df: DataFrame avec les données
            time_col: Nom de la colonne temporelle
            save_path: Chemin de sauvegarde (optionnel)
        """
        parameters = ['WOB', 'RPM', 'FlowRate', 'MudWeight', 'HookLoad']
        available_params = [p for p in parameters if p in df.columns]
        
        n_params = len(available_params)
        fig, axes = plt.subplots(n_params, 1, figsize=(15, 3*n_params))
        
        if n_params == 1:
            axes = [axes]
        
        fig.suptitle('🔧 Évolution des Paramètres de Forage', fontsize=16, fontweight='bold')
        
        for i, param in enumerate(available_params):
            axes[i].plot(df[time_col], df[param], color=self.colors[i % len(self.colors)], alpha=0.7)
            axes[i].set_title(f'{param}', fontweight='bold')
            axes[i].set_ylabel(param)
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_matrix(self, df: pd.DataFrame, 
                               save_path: Optional[str] = None) -> None:
        """
        Crée une matrice de corrélation
        
        Args:
            df: DataFrame avec les données numériques
            save_path: Chemin de sauvegarde (optionnel)
        """
        # Sélectionner seulement les colonnes numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r',
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('🔗 Matrice de Corrélation des Variables', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_distributions(self, df: pd.DataFrame, columns: List[str],
                          save_path: Optional[str] = None) -> None:
        """
        Visualise les distributions des variables
        
        Args:
            df: DataFrame avec les données
            columns: Liste des colonnes à visualiser
            save_path: Chemin de sauvegarde (optionnel)
        """
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('📊 Distributions des Variables', fontsize=16, fontweight='bold')
        
        for i, col in enumerate(columns):
            row, col_idx = divmod(i, n_cols)
            
            if col in df.columns:
                axes[row, col_idx].hist(df[col], bins=50, alpha=0.7,
                                       color=self.colors[i % len(self.colors)],
                                       edgecolor='black')
                axes[row, col_idx].set_title(f'{col}', fontweight='bold')
                axes[row, col_idx].set_xlabel(col)
                axes[row, col_idx].set_ylabel('Fréquence')
                axes[row, col_idx].grid(True, alpha=0.3)
        
        # Supprimer les subplots vides
        for i in range(len(columns), n_rows * n_cols):
            row, col_idx = divmod(i, n_cols)
            fig.delaxes(axes[row, col_idx])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_scatter_matrix(self, df: pd.DataFrame, columns: List[str],
                           target_col: Optional[str] = None,
                           save_path: Optional[str] = None) -> None:
        """
        Crée une matrice de scatter plots
        
        Args:
            df: DataFrame avec les données
            columns: Colonnes à inclure
            target_col: Colonne cible pour la coloration
            save_path: Chemin de sauvegarde (optionnel)
        """
        if target_col and target_col in df.columns:
            scatter_df = df[columns + [target_col]]
            pd.plotting.scatter_matrix(scatter_df, alpha=0.6, figsize=(15, 15),
                                     c=df[target_col], colormap='viridis')
        else:
            scatter_df = df[columns]
            pd.plotting.scatter_matrix(scatter_df, alpha=0.6, figsize=(15, 15))
        
        plt.suptitle('🔍 Matrice de Scatter Plots', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class ModelResultsVisualizer:
    """
    Classe pour visualiser les résultats des modèles
    """
    
    def __init__(self):
        self.colors = px.colors.qualitative.Set1
    
    def plot_model_comparison(self, results_df: pd.DataFrame,
                             metric_col: str = 'Mean_Score',
                             save_path: Optional[str] = None) -> None:
        """
        Compare les performances des modèles
        
        Args:
            results_df: DataFrame avec les résultats
            metric_col: Colonne de métrique à comparer
            save_path: Chemin de sauvegarde
        """
        plt.figure(figsize=(12, 8))
        
        bars = plt.bar(results_df['Model'], results_df[metric_col],
                      color=self.colors[:len(results_df)])
        
        plt.title('📊 Comparaison des Performances des Modèles', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Modèles')
        plt.ylabel(metric_col)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_prediction_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 title: str = "Prédictions vs Valeurs Réelles",
                                 save_path: Optional[str] = None) -> None:
        """
        Plot prédictions vs valeurs réelles
        
        Args:
            y_true: Valeurs réelles
            y_pred: Prédictions
            title: Titre du graphique
            save_path: Chemin de sauvegarde
        """
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.6, color='blue')
        
        # Ligne parfaite (y=x)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Prédiction parfaite')
        
        plt.xlabel('Valeurs Réelles')
        plt.ylabel('Prédictions')
        plt.title(title, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Calculer et afficher R²
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      save_path: Optional[str] = None) -> None:
        """
        Plot des résidus
        
        Args:
            y_true: Valeurs réelles
            y_pred: Prédictions
            save_path: Chemin de sauvegarde
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Résidus vs Prédictions
        axes[0].scatter(y_pred, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Prédictions')
        axes[0].set_ylabel('Résidus')
        axes[0].set_title('Résidus vs Prédictions')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Distribution des résidus
        axes[1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Résidus')
        axes[1].set_ylabel('Fréquence')
        axes[1].set_title('Distribution des Résidus')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 20,
                               save_path: Optional[str] = None) -> None:
        """
        Visualise l'importance des features
        
        Args:
            importance_df: DataFrame avec les importances
            top_n: Nombre de features à afficher
            save_path: Chemin de sauvegarde
        """
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'🎯 Top {top_n} Features les plus importantes', fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        
        # Colorer les barres
        for i, bar in enumerate(bars):
            bar.set_color(self.colors[i % len(self.colors)])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class InteractivePlots:
    """
    Classe pour créer des visualisations interactives avec Plotly
    """
    
    @staticmethod
    def create_drilling_dashboard(df: pd.DataFrame, time_col: str = 'Timestamp') -> go.Figure:
        """
        Crée un dashboard interactif des paramètres de forage
        
        Args:
            df: DataFrame avec les données
            time_col: Colonne temporelle
            
        Returns:
            Figure Plotly
        """
        parameters = ['WOB', 'RPM', 'FlowRate', 'MudWeight']
        available_params = [p for p in parameters if p in df.columns]
        
        fig = make_subplots(
            rows=len(available_params), cols=1,
            subplot_titles=available_params,
            vertical_spacing=0.08
        )
        
        for i, param in enumerate(available_params):
            fig.add_trace(
                go.Scatter(x=df[time_col], y=df[param], name=param,
                          mode='lines', line=dict(width=2)),
                row=i+1, col=1
            )
        
        fig.update_layout(
            title="🔧 Dashboard Interactif des Paramètres de Forage",
            height=200*len(available_params),
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_3d_scatter(df: pd.DataFrame, x_col: str, y_col: str, z_col: str,
                         color_col: Optional[str] = None) -> go.Figure:
        """
        Crée un scatter plot 3D interactif
        
        Args:
            df: DataFrame avec les données
            x_col, y_col, z_col: Colonnes pour les axes
            color_col: Colonne pour la couleur (optionnel)
            
        Returns:
            Figure Plotly 3D
        """
        fig = go.Figure(data=[go.Scatter3d(
            x=df[x_col],
            y=df[y_col],
            z=df[z_col],
            mode='markers',
            marker=dict(
                size=5,
                color=df[color_col] if color_col else 'blue',
                colorscale='Viridis' if color_col else None,
                showscale=True if color_col else False
            ),
            text=df.index,
            hovertemplate=f'<b>{x_col}</b>: %{{x}}<br>' +
                         f'<b>{y_col}</b>: %{{y}}<br>' +
                         f'<b>{z_col}</b>: %{{z}}<br>' +
                         '<extra></extra>'
        )])
        
        fig.update_layout(
            title=f"📊 Scatter Plot 3D: {x_col} vs {y_col} vs {z_col}",
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            )
        )
        
        return fig
    
    @staticmethod
    def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
        """
        Crée une heatmap de corrélation interactive
        
        Args:
            df: DataFrame avec les données numériques
            
        Returns:
            Figure Plotly
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdYlBu',
            zmid=0,
            text=np.around(corr_matrix.values, decimals=2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Corrélation: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title="🔗 Matrice de Corrélation Interactive",
            width=800,
            height=800
        )
        
        return fig

class AlertSystem:
    """
    Système d'alerte pour les visualisations
    """
    
    @staticmethod
    def detect_anomalies_in_plot(df: pd.DataFrame, column: str, 
                                threshold: float = 3.0) -> Tuple[np.ndarray, List[int]]:
        """
        Détecte les anomalies dans une série temporelle
        
        Args:
            df: DataFrame avec les données
            column: Colonne à analyser
            threshold: Seuil en écarts-types
            
        Returns:
            Masque d'anomalies et indices
        """
        from scipy import stats
        
        z_scores = np.abs(stats.zscore(df[column]))
        anomalies_mask = z_scores > threshold
        anomalies_indices = np.where(anomalies_mask)[0].tolist()
        
        return anomalies_mask, anomalies_indices
    
    @staticmethod
    def plot_with_anomalies(df: pd.DataFrame, time_col: str, value_col: str,
                           threshold: float = 3.0, save_path: Optional[str] = None) -> None:
        """
        Plot une série temporelle avec mise en évidence des anomalies
        
        Args:
            df: DataFrame avec les données
            time_col: Colonne temporelle
            value_col: Colonne des valeurs
            threshold: Seuil de détection d'anomalies
            save_path: Chemin de sauvegarde
        """
        anomalies_mask, anomalies_indices = AlertSystem.detect_anomalies_in_plot(
            df, value_col, threshold
        )
        
        plt.figure(figsize=(15, 8))
        
        # Plot des données normales
        plt.plot(df[time_col], df[value_col], 'b-', alpha=0.7, label='Données normales')
        
        # Mise en évidence des anomalies
        if len(anomalies_indices) > 0:
            plt.scatter(df[time_col].iloc[anomalies_indices], 
                       df[value_col].iloc[anomalies_indices],
                       color='red', s=50, alpha=0.8, label='Anomalies détectées')
        
        plt.title(f'🚨 Détection d\'anomalies: {value_col}', fontweight='bold')
        plt.xlabel(time_col)
        plt.ylabel(value_col)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Statistiques
        n_anomalies = len(anomalies_indices)
        percentage = (n_anomalies / len(df)) * 100
        plt.text(0.02, 0.98, f'Anomalies: {n_anomalies} ({percentage:.2f}%)',
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def create_report_figures(df: pd.DataFrame, model_results: Dict[str, Any],
                         output_dir: str = 'outputs/figures/') -> Dict[str, str]:
    """
    Crée toutes les figures pour un rapport
    
    Args:
        df: DataFrame avec les données
        model_results: Résultats des modèles
        output_dir: Dossier de sortie
        
    Returns:
        Dictionnaire avec les chemins des figures créées
    """
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    figures_paths = {}
    
    # Visualiseur principal
    viz = DrillingDataVisualizer()
    
    # 1. Évolution des paramètres
    if 'Timestamp' in df.columns:
        fig_path = f"{output_dir}/drilling_evolution.png"
        viz.plot_drilling_parameters_evolution(df, save_path=fig_path)
        figures_paths['drilling_evolution'] = fig_path
    
    # 2. Matrice de corrélation
    fig_path = f"{output_dir}/correlation_matrix.png"
    viz.plot_correlation_matrix(df, save_path=fig_path)
    figures_paths['correlation_matrix'] = fig_path
    
    # 3. Distributions
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]  # Top 6
    fig_path = f"{output_dir}/distributions.png"
    viz.plot_distributions(df, numeric_cols.tolist(), save_path=fig_path)
    figures_paths['distributions'] = fig_path
    
    # 4. Résultats des modèles si disponibles
    if model_results and 'comparison_df' in model_results:
        model_viz = ModelResultsVisualizer()
        fig_path = f"{output_dir}/model_comparison.png"
        model_viz.plot_model_comparison(model_results['comparison_df'], save_path=fig_path)
        figures_paths['model_comparison'] = fig_path
    
    return figures_paths

def save_interactive_plots(df: pd.DataFrame, output_dir: str = 'outputs/figures/') -> Dict[str, str]:
    """
    Sauvegarde les plots interactifs en HTML
    
    Args:
        df: DataFrame avec les données
        output_dir: Dossier de sortie
        
    Returns:
        Dictionnaire avec les chemins des fichiers HTML
    """
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    html_files = {}
    
    # Dashboard interactif
    if 'Timestamp' in df.columns:
        fig = InteractivePlots.create_drilling_dashboard(df)
        html_path = f"{output_dir}/interactive_dashboard.html"
        fig.write_html(html_path)
        html_files['dashboard'] = html_path
    
    # Heatmap de corrélation
    fig = InteractivePlots.create_correlation_heatmap(df)
    html_path = f"{output_dir}/correlation_heatmap.html"
    fig.write_html(html_path)
    html_files['correlation_heatmap'] = html_path
    
    return html_files