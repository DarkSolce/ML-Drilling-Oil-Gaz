"""
Setup script for ML-Drilling-Oil-Gas Project
===========================================

Script d'installation pour le package de machine learning
appliqué au forage pétrolier et gazier.
"""

from setuptools import setup, find_packages
import os
import sys
from pathlib import Path

# Lire le fichier README pour la description longue
def read_readme():
    """Lit le fichier README.md pour la description"""
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "ML project for drilling operations in oil and gas industry"

# Lire le fichier requirements.txt
def read_requirements(filename="requirements.txt"):
    """
    Lit le fichier requirements et retourne la liste des dépendances
    
    Args:
        filename: Nom du fichier requirements
        
    Returns:
        Liste des dépendances
    """
    requirements_path = Path(__file__).parent / filename
    if requirements_path.exists():
        with open(requirements_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
            
        # Filtrer les commentaires, lignes vides et options
        requirements = []
        for line in lines:
            line = line.strip()
            if (line and 
                not line.startswith("#") and 
                not line.startswith("-") and
                "://" not in line):  # Exclure les URLs git
                requirements.append(line)
        
        return requirements
    return []

# Dépendances par catégorie
def get_requirements_by_category():
    """
    Organise les dépendances par catégorie
    
    Returns:
        Dictionnaire des dépendances par catégorie
    """
    # Dépendances de base (minimum requis)
    base_requirements = [
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "scikit-learn>=1.1.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pyyaml>=6.0",
        "joblib>=1.2.0",
        "tqdm>=4.64.0"
    ]
    
    # Dépendances ML avancées
    ml_requirements = [
        "xgboost>=1.6.0",
        "lightgbm>=3.3.0",
        "optuna>=3.0.0",
        "shap>=0.41.0",
        "imbalanced-learn>=0.9.0"
    ]
    
    # Dépendances pour l'API
    api_requirements = [
        "fastapi>=0.85.0",
        "uvicorn[standard]>=0.18.0",
        "pydantic>=1.10.0",
        "python-multipart>=0.0.5"
    ]
    
    # Dépendances pour le dashboard
    dashboard_requirements = [
        "streamlit>=1.12.0",
        "plotly>=5.10.0",
        "dash>=2.6.0"
    ]
    
    # Dépendances de développement
    dev_requirements = [
        "pytest>=7.1.0",
        "pytest-cov>=4.0.0",
        "black>=22.6.0",
        "flake8>=5.0.0",
        "isort>=5.10.0",
        "jupyter>=1.0.0"
    ]
    
    # Dépendances Deep Learning (optionnelles)
    dl_requirements = [
        "tensorflow>=2.10.0,<2.16.0",
        "torch>=1.12.0",
        "torchvision>=0.13.0"
    ]
    
    # Dépendances Cloud (optionnelles)
    cloud_requirements = [
        "boto3>=1.24.0",
        "google-cloud-storage>=2.5.0",
        "azure-storage-blob>=12.13.0"
    ]
    
    return {
        "base": base_requirements,
        "ml": ml_requirements, 
        "api": api_requirements,
        "dashboard": dashboard_requirements,
        "dev": dev_requirements,
        "deep_learning": dl_requirements,
        "cloud": cloud_requirements
    }

# Lire la version depuis un fichier
def get_version():
    """
    Lit la version depuis le fichier __init__.py ou un fichier VERSION
    
    Returns:
        Version du package
    """
    # Essayer de lire depuis src/__init__.py
    init_file = Path(__file__).parent / "src" / "__init__.py"
    if init_file.exists():
        with open(init_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    
    # Essayer de lire depuis VERSION
    version_file = Path(__file__).parent / "VERSION"
    if version_file.exists():
        with open(version_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    
    # Version par défaut
    return "1.0.0"

# Configuration principale
def main():
    """Configuration principale du package"""
    
    requirements_dict = get_requirements_by_category()
    
    # Vérification de la version Python
    if sys.version_info < (3, 8):
        sys.exit("Python 3.8 ou plus récent est requis.")
    
    setup(
        # Informations de base
        name="ml-drilling-oil-gas",
        version=get_version(),
        description="Machine Learning pour opérations de forage pétrolier et gazier",
        long_description=read_readme(),
        long_description_content_type="text/markdown",
        
        # Informations sur l'auteur/organisation
        author="ML Engineering Team",
        author_email="ml-team@drilling-company.com",
        maintainer="ML Engineering Team", 
        maintainer_email="ml-team@drilling-company.com",
        
        # URLs du projet
        url="https://github.com/your-org/ML-Drilling-Oil-Gas",
        project_urls={
            "Bug Reports": "https://github.com/your-org/ML-Drilling-Oil-Gas/issues",
            "Source": "https://github.com/your-org/ML-Drilling-Oil-Gas",
            "Documentation": "https://ml-drilling.readthedocs.io/",
        },
        
        # Classification du package
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Manufacturing",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Information Analysis",
            "Topic :: Scientific/Engineering :: Physics",
        ],
        
        # Mots-clés
        keywords=[
            "machine learning", "drilling", "oil and gas", "petroleum engineering",
            "formation pressure", "kick detection", "predictive maintenance",
            "time series", "anomaly detection", "deep learning"
        ],
        
        # Configuration des packages
        packages=find_packages(include=["src", "src.*"]),
        package_dir={"": "."},
        
        # Inclure les fichiers de données
        include_package_data=True,
        package_data={
            "": [
                "configs/*.yaml",
                "configs/*.yml", 
                "configs/*.json",
                "data/external/*",
                "*.md",
                "*.txt",
                "*.yml",
                "*.yaml"
            ],
        },
        
        # Version Python requise
        python_requires=">=3.8",
        
        # Dépendances de base (toujours installées)
        install_requires=requirements_dict["base"],
        
        # Dépendances optionnelles
        extras_require={
            "full": (requirements_dict["ml"] + 
                    requirements_dict["api"] + 
                    requirements_dict["dashboard"]),
            "ml": requirements_dict["ml"],
            "api": requirements_dict["api"],
            "dashboard": requirements_dict["dashboard"],
            "dev": requirements_dict["dev"],
            "deep-learning": requirements_dict["deep_learning"],
            "cloud": requirements_dict["cloud"],
            "all": (requirements_dict["ml"] +
                   requirements_dict["api"] +
                   requirements_dict["dashboard"] +
                   requirements_dict["dev"] +
                   requirements_dict["deep_learning"] +
                   requirements_dict["cloud"])
        },
        
        # Points d'entrée (commandes CLI)
        entry_points={
            "console_scripts": [
                "ml-drilling=src.cli:main",
                "drilling-train=src.models.train:main",
                "drilling-predict=src.models.predict:main",
                "drilling-api=src.api.app:run_api",
                "drilling-dashboard=src.visualization.dashboard:run_dashboard",
                "drilling-pipeline=run_pipeline:main"
            ],
        },
        
        # Configuration des tests
        test_suite="tests",
        tests_require=requirements_dict["dev"],
        
        # Métadonnées additionnelles
        license="MIT",
        platforms=["any"],
        
        # Configuration pour wheel
        zip_safe=False,
        
        # Options pour develop
        options={
            "develop": {
                "easy_install": None,
            },
        },
    )

# Scripts d'installation personnalisés
class PostInstallCommand:
    """Commandes post-installation"""
    
    @staticmethod
    def create_directories():
        """Crée les dossiers nécessaires"""
        directories = [
            "data/raw",
            "data/processed", 
            "data/external",
            "models/formation_pressure",
            "models/kick_detection",
            "outputs/logs",
            "outputs/figures",
            "outputs/reports",
            "configs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✅ Créé le dossier: {directory}")
    
    @staticmethod
    def download_sample_data():
        """Télécharge des données d'exemple (optionnel)"""
        import urllib.request
        import zipfile
        
        sample_data_url = "https://example.com/sample_drilling_data.zip"
        try:
            print("📥 Téléchargement des données d'exemple...")
            urllib.request.urlretrieve(sample_data_url, "sample_data.zip")
            
            with zipfile.ZipFile("sample_data.zip", 'r') as zip_ref:
                zip_ref.extractall("data/external/")
            
            os.remove("sample_data.zip")
            print("✅ Données d'exemple téléchargées et extraites")
            
        except Exception as e:
            print(f"⚠️ Impossible de télécharger les données d'exemple: {e}")
    
    @staticmethod
    def setup_git_hooks():
        """Configure les hooks Git pour le développement"""
        hooks_dir = Path(".git/hooks")
        if hooks_dir.exists():
            # Pre-commit hook
            pre_commit_hook = hooks_dir / "pre-commit"
            hook_content = """#!/bin/sh
# Pre-commit hook for ML-Drilling project
echo "🔍 Exécution des vérifications pre-commit..."

# Formatage du code avec black
black src/ tests/ --check --quiet
if [ $? -ne 0 ]; then
    echo "❌ Code non formaté. Exécutez: black src/ tests/"
    exit 1
fi

# Vérification des imports avec isort
isort src/ tests/ --check-only --quiet
if [ $? -ne 0 ]; then
    echo "❌ Imports non triés. Exécutez: isort src/ tests/"
    exit 1
fi

# Linting avec flake8
flake8 src/ tests/
if [ $? -ne 0 ]; then
    echo "❌ Erreurs de linting détectées"
    exit 1
fi

# Tests unitaires rapides
pytest tests/ -x --tb=short
if [ $? -ne 0 ]; then
    echo "❌ Tests unitaires échoués"
    exit 1
fi

echo "✅ Toutes les vérifications ont réussi"
"""
            
            with open(pre_commit_hook, "w") as f:
                f.write(hook_content)
            
            # Rendre le hook exécutable
            os.chmod(pre_commit_hook, 0o755)
            print("✅ Hook pre-commit configuré")

def post_install():
    """Fonction appelée après l'installation"""
    print("\n" + "="*50)
    print("🎉 ML-Drilling-Oil-Gas installé avec succès!")
    print("="*50)
    
    post_install_cmd = PostInstallCommand()
    
    # Créer les dossiers
    post_install_cmd.create_directories()
    
    # Configuration Git hooks (seulement en mode développement)
    if os.path.exists(".git"):
        post_install_cmd.setup_git_hooks()
    
    print("\n📋 PROCHAINES ÉTAPES:")
    print("1. Copiez vos données dans data/raw/")
    print("2. Configurez configs/model_config.yaml selon vos besoins")
    print("3. Lancez l'entraînement: python run_pipeline.py")
    print("4. Démarrez l'API: drilling-api")
    print("5. Ouvrez le dashboard: drilling-dashboard")
    
    print("\n📚 DOCUMENTATION:")
    print("- README.md pour les instructions détaillées")
    print("- notebooks/ pour des exemples d'utilisation")
    print("- configs/ pour la configuration")
    
    print("\n🔧 OUTILS DISPONIBLES:")
    print("- ml-drilling: CLI principal")
    print("- drilling-train: Entraîner les modèles")
    print("- drilling-predict: Faire des prédictions")
    print("- drilling-api: Serveur API")
    print("- drilling-dashboard: Interface web")
    
    print("="*50)

# Configuration des commandes personnalisées
from setuptools.command.develop import develop
from setuptools.command.install import install

class PostDevelopCommand(develop):
    """Commande post-develop personnalisée"""
    def run(self):
        develop.run(self)
        post_install()

class PostInstallCommand(install):
    """Commande post-install personnalisée"""
    def run(self):
        install.run(self)
        post_install()

# Fonction de validation de l'environnement
def validate_environment():
    """
    Valide l'environnement avant l'installation
    
    Returns:
        bool: True si l'environnement est valide
    """
    errors = []
    
    # Vérifier Python version
    if sys.version_info < (3, 8):
        errors.append("Python 3.8+ requis")
    
    # Vérifier les dépendances système critiques
    try:
        import numpy
    except ImportError:
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
        except:
            errors.append("Impossible d'installer numpy")
    
    # Vérifier l'espace disque disponible
    import shutil
    free_space_gb = shutil.disk_usage(".")[2] / (1024**3)
    if free_space_gb < 1.0:  # 1GB minimum
        errors.append(f"Espace disque insuffisant: {free_space_gb:.1f}GB disponible, 1GB requis")
    
    # Vérifier la mémoire disponible
    try:
        import psutil
        memory_gb = psutil.virtual_memory().available / (1024**3)
        if memory_gb < 2.0:  # 2GB minimum recommandé
            print(f"⚠️ Mémoire limitée: {memory_gb:.1f}GB disponible, 2GB+ recommandé")
    except ImportError:
        pass
    
    if errors:
        print("❌ ERREURS DE VALIDATION:")
        for error in errors:
            print(f"  • {error}")
        return False
    
    print("✅ Environnement validé")
    return True

# Point d'entrée principal
if __name__ == "__main__":
    # Validation de l'environnement
    if not validate_environment():
        sys.exit("Installation annulée en raison d'erreurs de validation")
    
    # Mise à jour de la configuration setup avec les commandes personnalisées
    setup_config = main.__globals__.copy()
    setup_config.update({
        'cmdclass': {
            'develop': PostDevelopCommand,
            'install': PostInstallCommand,
        }
    })
    
    # Installation
    try:
        main()
        print("\n🎊 Installation terminée avec succès!")
    except Exception as e:
        print(f"\n❌ Erreur lors de l'installation: {e}")
        sys.exit(1)