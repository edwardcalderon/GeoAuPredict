#!/usr/bin/env python3
"""
GeoAuPredict Phase 3: Predictive Modeling - Model Evaluation

This module implements comprehensive evaluation metrics for geospatial models:
- Geographic performance analysis
- Spatial autocorrelation assessment
- Precision@k for exploration targeting
- Geographic confusion matrices
- Feature importance analysis

Author: GeoAuPredict Team
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class GeospatialModelEvaluator:
    """
    Comprehensive evaluation framework for geospatial prediction models.

    Key Features:
    - Geographic performance metrics
    - Spatial autocorrelation analysis
    - Precision@k for exploration targeting
    - Geographic confusion matrices
    - Feature importance with spatial context
    """

    def __init__(self, model_name: str = "Unknown"):
        """
        Initialize evaluator.

        Args:
            model_name: Name of the model being evaluated
        """
        self.model_name = model_name
        self.evaluation_results = {}

        logger.info(f"Initialized evaluator for {model_name}")

    def calculate_basic_metrics(self, y_true: np.ndarray,
                              y_pred: np.ndarray,
                              y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate standard classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities

        Returns:
            Dictionary with basic metrics
        """
        logger.info("Calculating basic classification metrics")

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }

        logger.info(f"Basic metrics: {metrics}")

        return metrics

    def calculate_precision_at_k(self, y_true: np.ndarray,
                               y_pred_proba: np.ndarray,
                               coordinates: np.ndarray,
                               k_values: List[int] = [10, 50, 100]) -> Dict[str, float]:
        """
        Calculate Precision@k for exploration targeting.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            coordinates: Array of (lat, lon) coordinates
            k_values: List of k values for evaluation

        Returns:
            Dictionary with precision@k values
        """
        logger.info(f"Calculating Precision@k for k={k_values}")

        # Convert to numpy arrays to avoid pandas index issues
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)

        # If probabilities are 2D (predict_proba output), take positive class column
        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] > 1:
            y_pred_proba = y_pred_proba[:, 1]

        # Sort by predicted probability (descending)
        sorted_indices = np.argsort(y_pred_proba)[::-1]

        precision_results = {}

        for k in k_values:
            if k > len(y_true):
                logger.warning(f"k={k} larger than dataset size {len(y_true)}")
                continue

            # Get top-k predictions
            top_k_indices = sorted_indices[:k]
            top_k_true = y_true[top_k_indices]

            # Calculate precision@k
            precision_k = np.mean(top_k_true)

            precision_results[f'precision@{k}'] = precision_k

            logger.info(f"Precision@{k}: {precision_k:.3f} ({np.sum(top_k_true)}/{k} true positives)")

        return precision_results

    def create_geographic_confusion_matrix(self, y_true: np.ndarray,
                                         y_pred: np.ndarray,
                                         coordinates: np.ndarray,
                                         grid_resolution: float = 1.0) -> pd.DataFrame:
        """
        Create confusion matrix binned by geographic location.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            coordinates: Array of (lat, lon) coordinates
            grid_resolution: Size of geographic bins in degrees

        Returns:
            DataFrame with geographic confusion matrix
        """
        logger.info(f"Creating geographic confusion matrix with {grid_resolution}° resolution")

        # Create geographic bins
        lat_bins = np.arange(coordinates[:, 0].min(), coordinates[:, 0].max() + grid_resolution, grid_resolution)
        lon_bins = np.arange(coordinates[:, 1].min(), coordinates[:, 1].max() + grid_resolution, grid_resolution)

        # Assign samples to bins
        lat_indices = np.digitize(coordinates[:, 0], lat_bins) - 1
        lon_indices = np.digitize(coordinates[:, 1], lon_bins) - 1

        # Create confusion matrix for each geographic bin
        n_lat_bins = len(lat_bins) - 1
        n_lon_bins = len(lon_bins) - 1

        geo_cm = np.zeros((n_lat_bins, n_lon_bins, 2, 2))  # (lat_bin, lon_bin, true_class, pred_class)

        for i in range(len(y_true)):
            lat_idx = lat_indices[i]
            lon_idx = lon_indices[i]
            true_class = int(y_true[i])
            pred_class = int(y_pred[i])

            geo_cm[lat_idx, lon_idx, true_class, pred_class] += 1

        # Convert to DataFrame
        results = []
        for lat_idx in range(n_lat_bins):
            for lon_idx in range(n_lon_bins):
                bin_data = {
                    'lat_bin': lat_idx,
                    'lon_bin': lon_idx,
                    'lat_center': (lat_bins[lat_idx] + lat_bins[lat_idx + 1]) / 2,
                    'lon_center': (lon_bins[lon_idx] + lon_bins[lon_idx + 1]) / 2,
                    'true_negative': geo_cm[lat_idx, lon_idx, 0, 0],
                    'false_positive': geo_cm[lat_idx, lon_idx, 0, 1],
                    'false_negative': geo_cm[lat_idx, lon_idx, 1, 0],
                    'true_positive': geo_cm[lat_idx, lon_idx, 1, 1],
                    'total_samples': geo_cm[lat_idx, lon_idx].sum()
                }

                if bin_data['total_samples'] > 0:
                    bin_data['accuracy'] = (bin_data['true_negative'] + bin_data['true_positive']) / bin_data['total_samples']
                    bin_data['precision'] = bin_data['true_positive'] / (bin_data['true_positive'] + bin_data['false_positive']) if (bin_data['true_positive'] + bin_data['false_positive']) > 0 else 0
                    bin_data['recall'] = bin_data['true_positive'] / (bin_data['true_positive'] + bin_data['false_negative']) if (bin_data['true_positive'] + bin_data['false_negative']) > 0 else 0

                results.append(bin_data)

        geo_cm_df = pd.DataFrame(results)

        logger.info(f"Created geographic confusion matrix with {len(geo_cm_df)} bins")

        return geo_cm_df

    def calculate_spatial_autocorrelation(self, y_true: np.ndarray,
                                        y_pred: np.ndarray,
                                        coordinates: np.ndarray,
                                        max_distance: float = 10000) -> Dict[str, float]:
        """
        Calculate spatial autocorrelation in predictions and errors.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            coordinates: Array of (lat, lon) coordinates
            max_distance: Maximum distance for spatial analysis (meters)

        Returns:
            Dictionary with spatial autocorrelation metrics
        """
        logger.info("Calculating spatial autocorrelation metrics")

        from scipy.spatial.distance import pdist, squareform

        # Calculate distance matrix
        distances = squareform(pdist(coordinates))

        # Create binary distance mask
        distance_mask = distances <= max_distance

        # Calculate Moran's I for predictions
        y_pred_centered = y_pred - np.mean(y_pred)
        y_true_centered = y_true - np.mean(y_true)

        # Spatial weights matrix (inverse distance)
        weights = 1.0 / (distances + 1e-10)  # Avoid division by zero
        weights[~distance_mask] = 0  # Only consider nearby points

        # Normalize weights by row
        row_sums = np.sum(weights, axis=1, keepdims=True)
        weights = weights / row_sums

        # Calculate Moran's I
        n = len(y_pred)
        numerator = np.sum(weights * np.outer(y_pred_centered, y_pred_centered))
        denominator = np.sum(y_pred_centered ** 2)

        morans_i_pred = (n / np.sum(weights)) * (numerator / denominator) if denominator > 0 else 0

        # Moran's I for errors
        errors = y_pred - y_true
        errors_centered = errors - np.mean(errors)

        numerator_error = np.sum(weights * np.outer(errors_centered, errors_centered))
        morans_i_error = (n / np.sum(weights)) * (numerator_error / np.sum(errors_centered ** 2)) if np.sum(errors_centered ** 2) > 0 else 0

        results = {
            'morans_i_predictions': morans_i_pred,
            'morans_i_errors': morans_i_error,
            'spatial_autocorrelation_present': abs(morans_i_pred) > 0.1,
            'error_spatial_autocorrelation': abs(morans_i_error) > 0.1
        }

        logger.info(f"Moran's I (predictions): {morans_i_pred:.3f}")
        logger.info(f"Moran's I (errors): {morans_i_error:.3f}")

        return results

    def calculate_exploration_metrics(self, y_true: np.ndarray,
                                    y_pred_proba: np.ndarray,
                                    coordinates: np.ndarray,
                                    area_thresholds: List[float] = [0.1, 0.25, 0.5]) -> Dict[str, float]:
        """
        Calculate exploration-specific metrics.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            coordinates: Array of (lat, lon) coordinates
            area_thresholds: Area thresholds for exploration targeting

        Returns:
            Dictionary with exploration metrics
        """
        logger.info("Calculating exploration-specific metrics")

        metrics = {}

        for threshold in area_thresholds:
            # Sort by probability and select top percentage
            n_select = max(1, int(len(y_true) * threshold))
            top_indices = np.argsort(y_pred_proba)[::-1][:n_select]

            # Calculate metrics for selected area
            selected_true = y_true[top_indices]
            selected_proba = y_pred_proba[top_indices]

            precision = np.mean(selected_true)
            recall = np.sum(selected_true) / np.sum(y_true) if np.sum(y_true) > 0 else 0

            metrics[f'precision_at_{int(threshold*100)}pct'] = precision
            metrics[f'recall_at_{int(threshold*100)}pct'] = recall
            metrics[f'gold_found_at_{int(threshold*100)}pct'] = np.sum(selected_true)

            logger.info(f"At {int(threshold*100)}% area: Precision={precision:.3f}, Recall={recall:.3f}")

        return metrics

    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      save_path: Optional[str] = None):
        """Plot ROC curve."""
        logger.info("Creating ROC curve plot")

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_score(y_true, y_pred_proba):.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                  save_path: Optional[str] = None):
        """Plot Precision-Recall curve."""
        logger.info("Creating Precision-Recall curve plot")

        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {self.model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_geographic_performance(self, geo_cm_df: pd.DataFrame,
                                  metric: str = 'accuracy',
                                  save_path: Optional[str] = None):
        """
        Plot geographic variation in model performance.

        Args:
            geo_cm_df: Geographic confusion matrix DataFrame
            metric: Metric to plot
            save_path: Optional save path
        """
        logger.info(f"Creating geographic performance plot for {metric}")

        plt.figure(figsize=(12, 8))

        # Filter bins with data
        plot_data = geo_cm_df[geo_cm_df['total_samples'] > 0]

        if len(plot_data) == 0:
            logger.warning("No geographic data to plot")
            return

        scatter = plt.scatter(plot_data['lon_center'], plot_data['lat_center'],
                            c=plot_data[metric], cmap='viridis',
                            s=plot_data['total_samples'] * 2, alpha=0.7,
                            edgecolors='black', linewidth=0.5)

        plt.colorbar(scatter, label=metric.title())
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'Geographic Variation in {metric.title()} - {self.model_name}')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def calculate_feature_importance_spatial(self, model,
                                           X: pd.DataFrame,
                                           y: pd.Series,
                                           coordinates: np.ndarray) -> pd.DataFrame:
        """
        Calculate feature importance with spatial context.

        Args:
            model: Trained model
            X: Feature matrix
            y: Target variable
            coordinates: Array of (lat, lon) coordinates

        Returns:
            DataFrame with feature importance and spatial analysis
        """
        logger.info("Calculating spatially-aware feature importance")

        # Standard permutation importance
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
        else:
            feature_names = X.columns

        perm_importance = permutation_importance(
            model, X, y, n_repeats=10, random_state=42
        )

        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)

        # Add spatial correlation analysis
        spatial_correlations = []
        for feature in feature_names:
            if feature in ['lat', 'lon']:
                spatial_correlations.append(0)  # Skip coordinate features
                continue

            # Calculate correlation between feature and coordinates
            lat_corr = abs(stats.pearsonr(X[feature], coordinates[:, 0])[0])
            lon_corr = abs(stats.pearsonr(X[feature], coordinates[:, 1])[0])

            spatial_correlations.append(max(lat_corr, lon_corr))

        importance_df['spatial_correlation'] = spatial_correlations

        logger.info("Feature importance calculated")
        logger.info(f"\n{importance_df.head(10)}")

        return importance_df

    def create_comprehensive_evaluation(self, model,
                                     X_test: pd.DataFrame,
                                     y_test: np.ndarray,
                                     coordinates: np.ndarray,
                                     model_predictions: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create comprehensive evaluation report.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            coordinates: Test coordinates
            model_predictions: Optional pre-computed predictions

        Returns:
            Dictionary with comprehensive evaluation results
        """
        logger.info("Creating comprehensive evaluation report")

        # Get predictions if not provided
        if model_predictions is None:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
        else:
            y_pred = model_predictions['predictions']
            y_pred_proba = model_predictions['probabilities']

        # Basic metrics
        basic_metrics = self.calculate_basic_metrics(y_test, y_pred, y_pred_proba)

        # Precision@k
        precision_k = self.calculate_precision_at_k(y_test, y_pred_proba, coordinates)

        # Geographic confusion matrix
        geo_cm = self.create_geographic_confusion_matrix(y_test, y_pred, coordinates)

        # Spatial autocorrelation
        spatial_autocorr = self.calculate_spatial_autocorrelation(y_test, y_pred, coordinates)

        # Exploration metrics
        exploration_metrics = self.calculate_exploration_metrics(y_test, y_pred_proba, coordinates)

        # Feature importance (if available)
        feature_importance = None
        if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
            feature_importance = self.calculate_feature_importance_spatial(model, X_test, y_test, coordinates)

        # Compile results
        evaluation_results = {
            'model_name': self.model_name,
            'basic_metrics': basic_metrics,
            'precision_k': precision_k,
            'geographic_confusion_matrix': geo_cm,
            'spatial_autocorrelation': spatial_autocorr,
            'exploration_metrics': exploration_metrics,
            'feature_importance': feature_importance,
            'n_test_samples': len(y_test),
            'test_class_distribution': pd.Series(y_test).value_counts().to_dict()
        }

        self.evaluation_results = evaluation_results

        logger.info("Comprehensive evaluation completed")

        return evaluation_results

    def save_evaluation_report(self, output_path: str):
        """
        Save evaluation results to file.

        Args:
            output_path: Path to save results
        """
        logger.info(f"Saving evaluation report to {output_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON for structured data
        import json

        # Convert DataFrames to dict for JSON serialization
        results_to_save = self.evaluation_results.copy()

        for key, value in results_to_save.items():
            if isinstance(value, pd.DataFrame):
                results_to_save[key] = value.to_dict('records')

        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)

        logger.info(f"Evaluation report saved to {output_path}")


def compare_model_evaluations(evaluations: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Compare evaluations from multiple models.

    Args:
        evaluations: List of evaluation dictionaries

    Returns:
        DataFrame with model comparison
    """
    logger.info("Comparing model evaluations")

    comparison_data = []

    for eval_result in evaluations:
        model_data = {
            'model_name': eval_result['model_name'],
            'accuracy': eval_result['basic_metrics']['accuracy'],
            'precision': eval_result['basic_metrics']['precision'],
            'recall': eval_result['basic_metrics']['recall'],
            'f1_score': eval_result['basic_metrics']['f1_score'],
            'roc_auc': eval_result['basic_metrics']['roc_auc'],
            'precision@50': eval_result['precision_k'].get('precision@50', 0),
            'precision@100': eval_result['precision_k'].get('precision@100', 0),
            'spatial_autocorr_pred': eval_result['spatial_autocorrelation']['morans_i_predictions'],
            'spatial_autocorr_error': eval_result['spatial_autocorrelation']['morans_i_errors'],
            'n_test_samples': eval_result['n_test_samples']
        }

        comparison_data.append(model_data)

    comparison_df = pd.DataFrame(comparison_data).set_index('model_name')
    comparison_df = comparison_df.sort_values('roc_auc', ascending=False)

    logger.info("Model comparison completed")
    logger.info(f"\n{comparison_df.round(4)}")

    return comparison_df
