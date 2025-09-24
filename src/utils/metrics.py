"""
Metrics Module for ML Drilling Project
======================================

Module contenant toutes les métriques personnalisées pour l'évaluation 
des modèles de forage pétrolier
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import logging

logger = logging.getLogger(__name__)

class RegressionMetrics:
    """
    Classe pour calculer les métriques de régression
    """
    
    @staticmethod
    def calculate_skewness(data: np.ndarray) -> float:
        """
        Calcule l'asymétrie (skewness) des données
        
        Args:
            data: Données à analyser
            
        Returns:
            Coefficient d'asymétrie
        """
        from scipy.stats import skew
        return float(skew(data))
    
    @staticmethod
    def drilling_efficiency_score(y_true: np.ndarray, y_pred: np.ndarray, 
                                 target_rop: float = 20.0) -> float:
        """
        Score d'efficacité spécifique au forage (ROP)
        
        Args:
            y_true: ROP réelles
            y_pred: ROP prédites
            target_rop: ROP cible pour le calcul d'efficacité
            
        Returns:
            Score d'efficacité (0-1, 1 étant parfait)
        """
        # Calculer l'erreur relative par rapport à la cible
        relative_error = np.abs(y_true - y_pred) / target_rop
        
        # Score basé sur l'inverse de l'erreur relative
        efficiency_score = np.mean(1 / (1 + relative_error))
        
        return float(efficiency_score)

class ClassificationMetrics:
    """
    Classe pour calculer les métriques de classification
    """
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                             y_proba: Optional[np.ndarray] = None, 
                             labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calcule toutes les métriques de classification
        
        Args:
            y_true: Valeurs réelles
            y_pred: Prédictions
            y_proba: Probabilités (optionnel)
            labels: Labels des classes (optionnel)
            
        Returns:
            Dictionnaire avec toutes les métriques
        """
        metrics = {}
        
        # Métriques de base
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Matrice de confusion
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Métriques par classe
        if labels:
            class_report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
            metrics['classification_report'] = class_report
        
        # Métriques basées sur les probabilités
        if y_proba is not None:
            try:
                if y_proba.ndim == 1 or y_proba.shape[1] == 2:
                    # Classification binaire
                    if y_proba.ndim == 2:
                        y_proba_binary = y_proba[:, 1]
                    else:
                        y_proba_binary = y_proba
                    
                    metrics['auc_roc'] = roc_auc_score(y_true, y_proba_binary)
                    
                    # Calculer la courbe ROC
                    fpr, tpr, _ = roc_curve(y_true, y_proba_binary)
                    metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
                    
                    # Courbe Précision-Rappel
                    precision, recall, _ = precision_recall_curve(y_true, y_proba_binary)
                    metrics['pr_auc'] = auc(recall, precision)
                    metrics['pr_curve'] = {'precision': precision.tolist(), 'recall': recall.tolist()}
                
                else:
                    # Classification multi-classes
                    metrics['auc_roc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                    
            except Exception as e:
                logger.warning(f"Erreur lors du calcul des métriques probabilistes: {e}")
        
        return metrics
    
    @staticmethod
    def kick_detection_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                              y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Métriques spécialisées pour la détection de kicks
        
        Args:
            y_true: Vraies étiquettes (0: normal, 1: kick)
            y_pred: Prédictions
            y_proba: Probabilités de kick (optionnel)
            
        Returns:
            Dictionnaire avec métriques spécialisées
        """
        metrics = {}
        
        # Matrice de confusion
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Métriques de base
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        
        # Taux de détection (sensibilité)
        metrics['detection_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Taux de fausses alarmes
        metrics['false_alarm_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Spécificité
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Précision pour les kicks
        metrics['kick_precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Score F1 pour les kicks
        precision = metrics['kick_precision']
        recall = metrics['detection_rate']
        metrics['kick_f1'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Coût pondéré (les faux négatifs sont plus coûteux)
        fn_cost_weight = 5  # Un kick manqué coûte 5 fois plus qu'une fausse alarme
        metrics['weighted_cost'] = fp + fn_cost_weight * fn
        
        return metrics

class DrillingSpecificMetrics:
    """
    Métriques spécifiques au domaine du forage
    """
    
    @staticmethod
    def rop_prediction_accuracy(y_true: np.ndarray, y_pred: np.ndarray, 
                               tolerance_percentage: float = 10.0) -> float:
        """
        Calcule la précision de prédiction du ROP avec tolérance
        
        Args:
            y_true: ROP réelles
            y_pred: ROP prédites
            tolerance_percentage: Tolérance en pourcentage
            
        Returns:
            Pourcentage de prédictions dans la tolérance
        """
        relative_errors = np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8)) * 100
        within_tolerance = relative_errors <= tolerance_percentage
        
        return float(np.mean(within_tolerance)) * 100
    
    @staticmethod
    def formation_pressure_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        RMSE spécialisé pour la pression de formation
        
        Args:
            y_true: Pressions réelles
            y_pred: Pressions prédites
            
        Returns:
            Dictionnaire avec RMSE et métriques dérivées
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Statistiques des pressions
        pressure_range = np.max(y_true) - np.min(y_true)
        normalized_rmse = rmse / pressure_range if pressure_range > 0 else np.inf
        
        return {
            'rmse_psi': float(rmse),
            'normalized_rmse': float(normalized_rmse),
            'pressure_range_psi': float(pressure_range),
            'max_prediction_error_psi': float(np.max(np.abs(y_true - y_pred)))
        }
    
    @staticmethod
    def drilling_optimization_score(actual_params: Dict[str, np.ndarray], 
                                   predicted_params: Dict[str, np.ndarray],
                                   weights: Optional[Dict[str, float]] = None) -> float:
        """
        Score global d'optimisation du forage
        
        Args:
            actual_params: Paramètres réels {param_name: values}
            predicted_params: Paramètres prédits {param_name: values}
            weights: Poids pour chaque paramètre
            
        Returns:
            Score d'optimisation (0-100)
        """
        if weights is None:
            weights = {param: 1.0 for param in actual_params.keys()}
        
        param_scores = {}
        
        for param_name in actual_params.keys():
            if param_name in predicted_params:
                # Calculer le score normalisé pour ce paramètre
                mape = RegressionMetrics.mean_absolute_percentage_error(
                    actual_params[param_name], 
                    predicted_params[param_name]
                )
                
                # Convertir MAPE en score (100 - MAPE, plafonné à 0)
                param_score = max(0, 100 - mape)
                param_scores[param_name] = param_score
        
        # Score pondéré global
        total_weight = sum(weights.get(param, 1.0) for param in param_scores.keys())
        if total_weight == 0:
            return 0.0
        
        weighted_score = sum(
            param_scores[param] * weights.get(param, 1.0) 
            for param in param_scores.keys()
        ) / total_weight
        
        return float(weighted_score)

class ModelComparisonMetrics:
    """
    Métriques pour comparer les modèles
    """
    
    @staticmethod
    def calculate_improvement_metrics(baseline_metrics: Dict[str, float],
                                    new_model_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Calcule les métriques d'amélioration par rapport à un modèle de base
        
        Args:
            baseline_metrics: Métriques du modèle de base
            new_model_metrics: Métriques du nouveau modèle
            
        Returns:
            Métriques d'amélioration
        """
        improvements = {}
        
        for metric in baseline_metrics.keys():
            if metric in new_model_metrics:
                baseline_val = baseline_metrics[metric]
                new_val = new_model_metrics[metric]
                
                if baseline_val != 0:
                    # Amélioration en pourcentage
                    if metric in ['mse', 'rmse', 'mae', 'mape']:  # Métriques à minimiser
                        improvement = ((baseline_val - new_val) / baseline_val) * 100
                    else:  # Métriques à maximiser (r2, accuracy, etc.)
                        improvement = ((new_val - baseline_val) / abs(baseline_val)) * 100
                    
                    improvements[f'{metric}_improvement_%'] = improvement
                
                improvements[f'{metric}_baseline'] = baseline_val
                improvements[f'{metric}_new'] = new_val
        
        return improvements
    
    @staticmethod
    def statistical_significance_test(y_true: np.ndarray, 
                                    pred1: np.ndarray, 
                                    pred2: np.ndarray) -> Dict[str, Any]:
        """
        Test de signification statistique entre deux modèles
        
        Args:
            y_true: Valeurs réelles
            pred1: Prédictions modèle 1
            pred2: Prédictions modèle 2
            
        Returns:
            Résultats du test statistique
        """
        from scipy.stats import wilcoxon, ttest_rel
        
        # Erreurs absolues
        errors1 = np.abs(y_true - pred1)
        errors2 = np.abs(y_true - pred2)
        
        # Test de Wilcoxon (non-paramétrique)
        try:
            wilcoxon_stat, wilcoxon_p = wilcoxon(errors1, errors2)
        except:
            wilcoxon_stat, wilcoxon_p = np.nan, np.nan
        
        # Test t apparié (paramétrique)
        try:
            ttest_stat, ttest_p = ttest_rel(errors1, errors2)
        except:
            ttest_stat, ttest_p = np.nan, np.nan
        
        return {
            'wilcoxon_statistic': float(wilcoxon_stat) if not np.isnan(wilcoxon_stat) else None,
            'wilcoxon_p_value': float(wilcoxon_p) if not np.isnan(wilcoxon_p) else None,
            'ttest_statistic': float(ttest_stat) if not np.isnan(ttest_stat) else None,
            'ttest_p_value': float(ttest_p) if not np.isnan(ttest_p) else None,
            'model1_mean_error': float(np.mean(errors1)),
            'model2_mean_error': float(np.mean(errors2)),
            'significant_difference': float(wilcoxon_p) < 0.05 if not np.isnan(wilcoxon_p) else False
        }

class MetricsReporter:
    """
    Classe pour générer des rapports de métriques
    """
    
    @staticmethod
    def generate_regression_report(y_true: np.ndarray, y_pred: np.ndarray,
                                  model_name: str = "Model") -> str:
        """
        Génère un rapport de métriques pour la régression
        
        Args:
            y_true: Valeurs réelles
            y_pred: Prédictions
            model_name: Nom du modèle
            
        Returns:
            Rapport formaté
        """
        metrics = RegressionMetrics.calculate_all_metrics(y_true, y_pred)
        
        report = f"""
📊 RAPPORT DE MÉTRIQUES - {model_name}
{'='*50}

🎯 MÉTRIQUES PRINCIPALES:
   • R² Score:           {metrics['r2']:.4f}
   • RMSE:              {metrics['rmse']:.4f}
   • MAE:               {metrics['mae']:.4f}
   • MAPE:              {metrics['mape']:.2f}%

📈 MÉTRIQUES DÉTAILLÉES:
   • MSE:               {metrics['mse']:.4f}
   • SMAPE:             {metrics['smape']:.2f}%
   • Max Error:         {metrics['max_error']:.4f}
   • Median AE:         {metrics['median_ae']:.4f}

🔍 ANALYSE DES RÉSIDUS:
   • Moyenne:           {metrics['residual_mean']:.4f}
   • Écart-type:        {metrics['residual_std']:.4f}
   • Asymétrie:         {metrics['residual_skew']:.4f}
        """
        
        # Ajouter métriques spécifiques au forage si applicable
        if np.mean(y_true) > 0:  # Probablement du ROP
            efficiency = RegressionMetrics.drilling_efficiency_score(y_true, y_pred)
            accuracy_10 = DrillingSpecificMetrics.rop_prediction_accuracy(y_true, y_pred, 10.0)
            accuracy_20 = DrillingSpecificMetrics.rop_prediction_accuracy(y_true, y_pred, 20.0)
            
            report += f"""
⚡ MÉTRIQUES FORAGE:
   • Efficacité:        {efficiency:.4f}
   • Précision ±10%:    {accuracy_10:.2f}%
   • Précision ±20%:    {accuracy_20:.2f}%
            """
        
        return report
    
    @staticmethod
    def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                     y_proba: Optional[np.ndarray] = None,
                                     model_name: str = "Model") -> str:
        """
        Génère un rapport de métriques pour la classification
        
        Args:
            y_true: Valeurs réelles
            y_pred: Prédictions
            y_proba: Probabilités (optionnel)
            model_name: Nom du modèle
            
        Returns:
            Rapport formaté
        """
        metrics = ClassificationMetrics.calculate_all_metrics(y_true, y_pred, y_proba)
        
        report = f"""
📊 RAPPORT DE CLASSIFICATION - {model_name}
{'='*50}

🎯 MÉTRIQUES PRINCIPALES:
   • Accuracy:          {metrics['accuracy']:.4f}
   • Precision (macro):  {metrics['precision_macro']:.4f}
   • Recall (macro):     {metrics['recall_macro']:.4f}
   • F1-Score (macro):   {metrics['f1_macro']:.4f}
        """
        
        if 'auc_roc' in metrics:
            report += f"   • AUC-ROC:           {metrics['auc_roc']:.4f}\n"
        
        if 'pr_auc' in metrics:
            report += f"   • AUC-PR:            {metrics['pr_auc']:.4f}\n"
        
        # Matrice de confusion
        cm = np.array(metrics['confusion_matrix'])
        if cm.shape == (2, 2):  # Classification binaire
            tn, fp, fn, tp = cm.ravel()
            report += f"""
🔍 MATRICE DE CONFUSION:
   • True Negatives:    {tn}
   • False Positives:   {fp}
   • False Negatives:   {fn}
   • True Positives:    {tp}
            """
        
        return report

def create_metrics_dashboard_data(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Crée un DataFrame pour un dashboard de métriques
    
    Args:
        results: Résultats de plusieurs modèles {model_name: metrics_dict}
        
    Returns:
        DataFrame avec les métriques organisées
    """
    rows = []
    
    for model_name, metrics in results.items():
        row = {'Model': model_name}
        
        # Extraire les métriques principales
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                row[metric_name] = value
        
        rows.append(row)
    
    return pd.DataFrame(rows)
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calcule toutes les métriques de régression
        
        Args:
            y_true: Valeurs réelles
            y_pred: Prédictions
            
        Returns:
            Dictionnaire avec toutes les métriques
        """
        metrics = {}
        
        # Métriques de base
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Métriques personnalisées
        metrics['mape'] = RegressionMetrics.mean_absolute_percentage_error(y_true, y_pred)
        metrics['smape'] = RegressionMetrics.symmetric_mean_absolute_percentage_error(y_true, y_pred)
        metrics['max_error'] = np.max(np.abs(y_true - y_pred))
        metrics['median_ae'] = np.median(np.abs(y_true - y_pred))
        
        # Métriques de distribution des erreurs
        residuals = y_true - y_pred
        metrics['residual_mean'] = np.mean(residuals)
        metrics['residual_std'] = np.std(residuals)
        metrics['residual_skew'] = RegressionMetrics.calculate_skewness(residuals)
        
        return metrics
    
    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcule le MAPE (Mean Absolute Percentage Error)
        
        Args:
            y_true: Valeurs réelles
            y_pred: Prédictions
            
        Returns:
            MAPE en pourcentage
        """
        # Éviter la division par zéro
        mask = y_true != 0
        if not np.any(mask):
            return np.inf
        
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcule le SMAPE (Symmetric Mean Absolute Percentage Error)
        
        Args:
            y_true: Valeurs réelles
            y_pred: Prédictions
            
        Returns:
            SMAPE en pourcentage
        """
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        
        # Éviter la division par zéro
        mask = denominator != 0
        if not np.any(mask):
            return 0.0
        
        return np.mean(numerator[mask] / denominator[mask]) * 100