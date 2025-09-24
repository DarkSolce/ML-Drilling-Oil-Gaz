"""
Visualization Module for ML-Drilling-Oil-Gas
===========================================

Module de visualisation contenant tous les outils pour créer
des graphiques, dashboards et rapports visuels pour les données
et résultats de forage pétrolier.
"""

__version__ = "1.0.0"

# Imports principaux
try:
    from .plots import (
        DrillingDataVisualizer,
        ModelResultsVisualizer, 
        InteractivePlots,
        AlertSystem,
        create_report_figures,
        save_interactive_plots
    )
    
    from .dashboard import (
        create_drilling_dashboard,
        run_dashboard,
        DrillingDashboard
    )
    
    __all__ = [
        # Visualiseurs principaux
        'DrillingDataVisualizer',
        'ModelResultsVisualizer',
        'InteractivePlots', 
        'AlertSystem',
        
        # Fonctions utilitaires
        'create_report_figures',
        'save_interactive_plots',
        
        # Dashboard
        'create_drilling_dashboard',
        'run_dashboard', 
        'DrillingDashboard'
    ]
    
except ImportError as e:
    import warnings
    warnings.warn(f"Erreur d'import dans le module visualization: {e}")
    __all__ = []

# Configuration par défaut
DEFAULT_VIZ_CONFIG = {
    'style': 'seaborn-v0_8',
    'color_palette': 'husl',
    'figure_size': (12, 8),
    'dpi': 300,
    'font_size': 10,
    'save_format': 'png',
    'interactive': True,
    'dark_mode': False
}

# Palettes de couleurs pour le forage
DRILLING_COLOR_PALETTES = {
    'formation_types': {
        'sandstone': '#F4A460',
        'shale': '#696969', 
        'limestone': '#D3D3D3',
        'dolomite': '#DDA0DD',
        'anhydrite': '#FFE4E1',
        'salt': '#F0F8FF'
    },
    'drilling_parameters': {
        'wob': '#1f77b4',      # Bleu
        'rpm': '#ff7f0e',      # Orange  
        'rop': '#2ca02c',      # Vert
        'torque': '#d62728',   # Rouge
        'pressure': '#9467bd', # Violet
        'flow_rate': '#8c564b' # Marron
    },
    'alerts': {
        'normal': '#28a745',   # Vert
        'warning': '#ffc107',  # Jaune
        'critical': '#dc3545', # Rouge
        'info': '#17a2b8'      # Bleu clair
    },
    'well_sections': {
        'surface': '#8B4513',
        'intermediate': '#CD853F', 
        'production': '#DAA520',
        'horizontal': '#B8860B'
    }
}

def get_visualization_info():
    """
    Retourne les informations sur le module de visualisation
    
    Returns:
        dict: Informations sur le module
    """
    return {
        'name': 'ML Drilling Visualization',
        'version': __version__,
        'description': 'Module de visualisation pour les données de forage',
        'supported_formats': ['PNG', 'SVG', 'PDF', 'HTML', 'JSON'],
        'chart_types': [
            'Time Series', 'Scatter Plots', 'Histograms', 
            'Correlation Heatmaps', '3D Plots', 'Box Plots',
            'Violin Plots', 'ROC Curves', 'Confusion Matrix'
        ],
        'interactive_features': [
            'Plotly Integration', 'Zoom/Pan', 'Hover Info',
            'Filtering', 'Real-time Updates', 'Export Options'
        ],
        'dashboard_features': [
            'Streamlit Interface', 'Real-time Monitoring',
            'Multi-page Layout', 'Parameter Controls',
            'Alert System', 'Model Comparison'
        ]
    }

def setup_visualization(config=None):
    """
    Configure le module de visualisation
    
    Args:
        config: Configuration personnalisée
    """
    if config is None:
        config = DEFAULT_VIZ_CONFIG.copy()
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("🎨 Configuration du module Visualization...")
    
    # Configuration matplotlib
    try:
        plt.style.use(config.get('style', 'seaborn-v0_8'))
        plt.rcParams['figure.figsize'] = config.get('figure_size', (12, 8))
        plt.rcParams['font.size'] = config.get('font_size', 10)
        plt.rcParams['savefig.dpi'] = config.get('dpi', 300)
        plt.rcParams['savefig.format'] = config.get('save_format', 'png')
        print("✅ Matplotlib configuré")
    except Exception as e:
        print(f"⚠️ Erreur configuration matplotlib: {e}")
    
    # Configuration seaborn
    try:
        sns.set_palette(config.get('color_palette', 'husl'))
        if config.get('dark_mode', False):
            sns.set_theme(style="darkgrid")
        print("✅ Seaborn configuré")
    except Exception as e:
        print(f"⚠️ Erreur configuration seaborn: {e}")
    
    print("✅ Module Visualization configuré")

def get_drilling_color(parameter_name, palette='drilling_parameters'):
    """
    Retourne la couleur associée à un paramètre de forage
    
    Args:
        parameter_name: Nom du paramètre
        palette: Nom de la palette à utiliser
        
    Returns:
        str: Code couleur hexadécimal
    """
    palettes = DRILLING_COLOR_PALETTES.get(palette, {})
    return palettes.get(parameter_name.lower(), '#1f77b4')  # Bleu par défaut

def create_drilling_theme():
    """
    Crée un thème personnalisé pour les visualisations de forage
    
    Returns:
        dict: Configuration du thème
    """
    return {
        'background_color': '#f8f9fa',
        'grid_color': '#dee2e6',
        'text_color': '#212529',
        'primary_color': '#007bff',
        'success_color': '#28a745',
        'warning_color': '#ffc107', 
        'danger_color': '#dc3545',
        'info_color': '#17a2b8',
        'font_family': 'Arial, sans-serif',
        'font_sizes': {
            'title': 16,
            'subtitle': 14,
            'body': 12,
            'caption': 10
        },
        'line_styles': {
            'solid': '-',
            'dashed': '--',
            'dotted': ':',
            'dashdot': '-.'
        }
    }

def apply_drilling_theme(ax, theme=None):
    """
    Applique le thème de forage à un axe matplotlib
    
    Args:
        ax: Axe matplotlib
        theme: Thème à appliquer (optionnel)
    """
    if theme is None:
        theme = create_drilling_theme()
    
    # Configuration de l'axe
    ax.set_facecolor(theme['background_color'])
    ax.grid(True, color=theme['grid_color'], alpha=0.7)
    ax.tick_params(colors=theme['text_color'])
    
    # Configuration des labels
    ax.xaxis.label.set_color(theme['text_color'])
    ax.yaxis.label.set_color(theme['text_color'])
    
    # Configuration du titre
    if ax.get_title():
        ax.title.set_color(theme['text_color'])
        ax.title.set_fontsize(theme['font_sizes']['title'])

def create_quick_plot(data, plot_type='line', **kwargs):
    """
    Création rapide de graphiques pour les données de forage
    
    Args:
        data: Données à visualiser (DataFrame ou dict)
        plot_type: Type de graphique ('line', 'scatter', 'hist', 'box')
        **kwargs: Arguments supplémentaires
        
    Returns:
        matplotlib.figure.Figure: Figure créée
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Convertir en DataFrame si nécessaire
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if plot_type == 'line':
        for col in data.select_dtypes(include=['number']).columns:
            ax.plot(data.index, data[col], label=col, 
                   color=get_drilling_color(col))
        ax.legend()
        
    elif plot_type == 'scatter':
        x_col = kwargs.get('x', data.columns[0])
        y_col = kwargs.get('y', data.columns[1] if len(data.columns) > 1 else data.columns[0])
        ax.scatter(data[x_col], data[y_col], alpha=0.6)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        
    elif plot_type == 'hist':
        col = kwargs.get('column', data.columns[0])
        ax.hist(data[col], bins=kwargs.get('bins', 30), 
               alpha=0.7, color=get_drilling_color(col))
        ax.set_xlabel(col)
        ax.set_ylabel('Fréquence')
        
    elif plot_type == 'box':
        numeric_cols = data.select_dtypes(include=['number']).columns
        data[numeric_cols].boxplot(ax=ax)
        
    # Appliquer le thème
    apply_drilling_theme(ax)
    
    plt.tight_layout()
    return fig

def save_figure(fig, filename, output_dir='outputs/figures/', **kwargs):
    """
    Sauvegarde une figure avec les bonnes pratiques
    
    Args:
        fig: Figure matplotlib à sauvegarder
        filename: Nom du fichier (sans extension)
        output_dir: Dossier de sortie
        **kwargs: Arguments pour savefig
    """
    from pathlib import Path
    
    # S'assurer que le dossier existe
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Configuration par défaut
    save_kwargs = {
        'dpi': 300,
        'bbox_inches': 'tight',
        'facecolor': 'white',
        'edgecolor': 'none'
    }
    save_kwargs.update(kwargs)
    
    # Sauvegarder en PNG par défaut
    file_path = output_path / f"{filename}.png"
    fig.savefig(file_path, **save_kwargs)
    
    print(f"✅ Figure sauvegardée: {file_path}")
    return str(file_path)

def create_drilling_report_template():
    """
    Crée un template HTML pour les rapports de forage
    
    Returns:
        str: Template HTML
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Drilling Analysis Report</title>
        <meta charset="utf-8">
        <style>
            body {
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f8f9fa;
                color: #212529;
            }
            .header {
                background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
                color: white;
                padding: 30px;
                text-align: center;
                border-radius: 10px;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .section {
                background: white;
                padding: 25px;
                margin-bottom: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .metric-card {
                display: inline-block;
                background: #f8f9fa;
                padding: 15px;
                margin: 10px;
                border-radius: 8px;
                border-left: 4px solid #007bff;
                min-width: 150px;
            }
            .metric-value {
                font-size: 24px;
                font-weight: bold;
                color: #007bff;
            }
            .metric-label {
                font-size: 14px;
                color: #6c757d;
                margin-top: 5px;
            }
            .alert {
                padding: 12px;
                margin: 10px 0;
                border-radius: 5px;
                border-left: 4px solid;
            }
            .alert-success { background-color: #d4edda; border-color: #28a745; }
            .alert-warning { background-color: #fff3cd; border-color: #ffc107; }
            .alert-danger { background-color: #f8d7da; border-color: #dc3545; }
            .chart-container {
                text-align: center;
                margin: 20px 0;
            }
            .chart-container img {
                max-width: 100%;
                height: auto;
                border-radius: 5px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            .footer {
                text-align: center;
                margin-top: 40px;
                padding: 20px;
                background: #e9ecef;
                border-radius: 8px;
                color: #6c757d;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #dee2e6;
            }
            th {
                background-color: #f8f9fa;
                font-weight: bold;
                color: #495057;
            }
            .status-active { color: #28a745; font-weight: bold; }
            .status-warning { color: #ffc107; font-weight: bold; }
            .status-error { color: #dc3545; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🛢️ ML Drilling Analysis Report</h1>
            <p>Generated on: {timestamp}</p>
        </div>
        
        <!-- Le contenu sera inséré ici -->
        {content}
        
        <div class="footer">
            <p>Generated by ML-Drilling-Oil-Gas v{version}</p>
            <p>© 2024 ML Engineering Team</p>
        </div>
    </body>
    </html>
    """

class PlotManager:
    """
    Gestionnaire centralisé pour tous les graphiques
    """
    
    def __init__(self, output_dir='outputs/figures/', theme=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.theme = theme or create_drilling_theme()
        self.figures = {}
    
    def create_figure(self, name, figsize=(12, 8)):
        """Crée une nouvelle figure"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=figsize)
        apply_drilling_theme(ax, self.theme)
        self.figures[name] = fig
        return fig, ax
    
    def save_figure(self, name, **kwargs):
        """Sauvegarde une figure"""
        if name not in self.figures:
            raise ValueError(f"Figure '{name}' non trouvée")
        
        fig = self.figures[name]
        file_path = save_figure(fig, name, str(self.output_dir), **kwargs)
        return file_path
    
    def save_all_figures(self, **kwargs):
        """Sauvegarde toutes les figures"""
        saved_paths = {}
        for name in self.figures:
            saved_paths[name] = self.save_figure(name, **kwargs)
        return saved_paths
    
    def close_all(self):
        """Ferme toutes les figures"""
        import matplotlib.pyplot as plt
        for fig in self.figures.values():
            plt.close(fig)
        self.figures.clear()

# Fonctions de validation
def validate_plot_data(data, plot_type):
    """
    Valide les données pour un type de graphique donné
    
    Args:
        data: Données à valider
        plot_type: Type de graphique
        
    Returns:
        tuple: (is_valid, message)
    """
    import pandas as pd
    
    if data is None or (hasattr(data, '__len__') and len(data) == 0):
        return False, "Données vides ou None"
    
    if not isinstance(data, (pd.DataFrame, dict, list)):
        return False, "Type de données non supporté"
    
    # Convertir en DataFrame pour l'analyse
    if isinstance(data, dict):
        data = pd.DataFrame(data)
    elif isinstance(data, list):
        data = pd.DataFrame(data)
    
    if len(data) == 0:
        return False, "DataFrame vide"
    
    # Validations spécifiques par type de graphique
    if plot_type in ['line', 'scatter']:
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            return False, f"Graphique {plot_type} nécessite au moins une colonne numérique"
    
    elif plot_type == 'hist':
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            return False, "Histogramme nécessite des données numériques"
    
    elif plot_type == 'heatmap':
        if data.shape[1] < 2:
            return False, "Heatmap nécessite au moins 2 colonnes"
    
    return True, "Données valides"

def check_plot_requirements():
    """
    Vérifie que toutes les dépendances pour la visualisation sont installées
    
    Returns:
        dict: Statut des dépendances
    """
    requirements = {
        'matplotlib': False,
        'seaborn': False,
        'plotly': False,
        'streamlit': False,
        'pandas': False,
        'numpy': False
    }
    
    for package in requirements:
        try:
            __import__(package)
            requirements[package] = True
        except ImportError:
            requirements[package] = False
    
    return requirements

# Utilitaires d'export
def export_plot_config(config, filename='plot_config.json'):
    """
    Exporte la configuration des graphiques
    
    Args:
        config: Configuration à exporter
        filename: Nom du fichier
    """
    import json
    from pathlib import Path
    
    config_path = Path('configs') / filename
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration exportée: {config_path}")

def import_plot_config(filename='plot_config.json'):
    """
    Importe la configuration des graphiques
    
    Args:
        filename: Nom du fichier à importer
        
    Returns:
        dict: Configuration importée
    """
    import json
    from pathlib import Path
    
    config_path = Path('configs') / filename
    
    if not config_path.exists():
        print(f"⚠️ Fichier de configuration non trouvé: {config_path}")
        return DEFAULT_VIZ_CONFIG.copy()
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"✅ Configuration importée: {config_path}")
    return config

# Classes d'exception pour la visualisation
class VisualizationError(Exception):
    """Exception de base pour les erreurs de visualisation"""
    pass

class PlotConfigError(VisualizationError):
    """Exception pour les erreurs de configuration de graphique"""
    pass

class DataVisualizationError(VisualizationError):
    """Exception pour les erreurs liées aux données de visualisation"""
    pass

# Fonction de diagnostic
def diagnose_visualization_setup():
    """
    Diagnostique la configuration de visualisation
    
    Returns:
        dict: Rapport de diagnostic
    """
    import sys
    import matplotlib
    
    report = {
        'timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else 'N/A',
        'python_version': sys.version,
        'dependencies': check_plot_requirements(),
        'matplotlib_backend': matplotlib.get_backend(),
        'matplotlib_version': matplotlib.__version__,
        'issues': [],
        'recommendations': []
    }
    
    # Vérifier les dépendances critiques
    critical_deps = ['matplotlib', 'pandas', 'numpy']
    missing_critical = [dep for dep in critical_deps if not report['dependencies'][dep]]
    
    if missing_critical:
        report['issues'].append(f"Dépendances critiques manquantes: {missing_critical}")
        report['recommendations'].append("Installez les dépendances avec: pip install matplotlib pandas numpy")
    
    # Vérifier les dépendances optionnelles
    optional_deps = ['seaborn', 'plotly', 'streamlit']
    missing_optional = [dep for dep in optional_deps if not report['dependencies'][dep]]
    
    if missing_optional:
        report['recommendations'].append(f"Pour toutes les fonctionnalités, installez: pip install {' '.join(missing_optional)}")
    
    # Vérifier le backend matplotlib
    if report['matplotlib_backend'] == 'Agg':
        report['recommendations'].append("Backend matplotlib 'Agg' détecté - les graphiques interactifs ne seront pas disponibles")
    
    return report

# Cache pour les figures fréquemment utilisées
_figure_cache = {}

def get_cached_figure(cache_key):
    """Récupère une figure du cache"""
    return _figure_cache.get(cache_key)

def cache_figure(cache_key, figure):
    """Met en cache une figure"""
    _figure_cache[cache_key] = figure

def clear_figure_cache():
    """Vide le cache des figures"""
    global _figure_cache
    _figure_cache.clear()
    print("✅ Cache des figures vidé")

# Message d'information du module
def _visualization_info():
    """Affiche les informations du module visualization"""
    info = get_visualization_info()
    print(f"""
🎨 {info['name']} v{info['version']}
{info['description']}

📊 Types de graphiques supportés:""")
    for chart_type in info['chart_types'][:4]:  # Afficher les 4 premiers
        print(f"  • {chart_type}")
    print(f"  • ... et {len(info['chart_types'])-4} autres")
    
    print(f"""
✨ Fonctionnalités interactives:""")
    for feature in info['interactive_features'][:3]:  # Afficher les 3 premières
        print(f"  • {feature}")
    print(f"  • ... et {len(info['interactive_features'])-3} autres")
    
    print(f"""
🎛️  Création rapide:
  • create_quick_plot(data, 'line')   - Graphique linéaire
  • create_quick_plot(data, 'hist')   - Histogramme
  • get_drilling_color('wob')         - Couleurs standardisées
  • apply_drilling_theme(ax)          - Thème professionnel
""")
    
    # Diagnostic rapide
    deps = check_plot_requirements()
    missing = [name for name, available in deps.items() if not available]
    if missing:
        print(f"⚠️ Dépendances manquantes: {', '.join(missing)}")
    else:
        print("✅ Toutes les dépendances sont disponibles")

# Afficher l'info lors de l'import (seulement si pas dans __main__)
if __name__ != "__main__":
    try:
        _visualization_info()
    except:
        pass  # Silencieux si erreur d'affichage