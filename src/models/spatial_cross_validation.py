#!/usr/bin/env python3
"""
GeoAuPredict Phase 3: Predictive Modeling - Spatial Cross-Validation

This module implements spatial cross-validation strategies for geographic data:
- Geographic blocking (spatial blocks)
- Distance-based splitting
- K-means spatial clustering
- Geographic coordinate-based stratification

Author: GeoAuPredict Team
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


class SpatialCrossValidator:
    """
    Implements spatial cross-validation for geospatial data.

    Key Features:
    - Geographic blocking to prevent data leakage
    - Distance-based train/test splitting
    - Spatial clustering for validation folds
    - Geographic performance visualization
    """

    def __init__(self, method: str = 'geographic_blocks',
                 n_splits: int = 5,
                 block_size: float = 1.0,
                 distance_threshold: float = 1000,
                 random_state: int = 42):
        """
        Initialize spatial cross-validator.

        Args:
            method: Validation method ('geographic_blocks', 'distance_based', 'kmeans_clusters')
            n_splits: Number of CV folds
            block_size: Size of geographic blocks in degrees
            distance_threshold: Minimum distance for distance-based splitting (meters)
            random_state: Random seed
        """
        self.method = method
        self.n_splits = n_splits
        self.block_size = block_size
        self.distance_threshold = distance_threshold
        self.random_state = random_state

        logger.info(f"Initialized spatial CV with {method} method")

    def create_geographic_blocks(self, coordinates: np.ndarray,
                               block_size: float) -> np.ndarray:
        """
        Create geographic blocks for spatial cross-validation.

        Args:
            coordinates: Array of (lat, lon) coordinates
            block_size: Size of blocks in degrees

        Returns:
            Array of block identifiers
        """
        lat_blocks = (coordinates[:, 0] // block_size).astype(int)
        lon_blocks = (coordinates[:, 1] // block_size).astype(int)

        # Create unique block IDs
        block_ids = lat_blocks * 1000 + lon_blocks  # Simple encoding

        return block_ids

    def create_spatial_clusters(self, coordinates: np.ndarray,
                              n_clusters: int) -> np.ndarray:
        """
        Create spatial clusters using K-means.

        Args:
            coordinates: Array of (lat, lon) coordinates
            n_clusters: Number of clusters

        Returns:
            Array of cluster assignments
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        clusters = kmeans.fit_predict(coordinates)

        return clusters

    def create_distance_based_folds(self, coordinates: np.ndarray,
                                  n_splits: int) -> List[np.ndarray]:
        """
        Create folds based on distance thresholds.

        Args:
            coordinates: Array of (lat, lon) coordinates
            n_splits: Number of folds

        Returns:
            List of fold assignments
        """
        from scipy.spatial.distance import pdist, squareform

        # Calculate distance matrix
        distances = squareform(pdist(coordinates))

        # Create folds by selecting distant points
        n_samples = len(coordinates)
        fold_size = n_samples // n_splits

        # Initialize folds
        folds = [np.array([]) for _ in range(n_splits)]
        remaining_indices = np.arange(n_samples)

        for fold_idx in range(n_splits):
            if fold_idx == n_splits - 1:
                # Last fold gets remaining samples
                folds[fold_idx] = remaining_indices
            else:
                # Select samples that are far from already selected samples
                selected_indices = []

                for _ in range(min(fold_size, len(remaining_indices))):
                    if len(selected_indices) == 0:
                        # First selection - pick random sample
                        idx = np.random.choice(remaining_indices)
                    else:
                        # Select sample farthest from already selected samples
                        min_distances = distances[remaining_indices][:, selected_indices].min(axis=1)
                        farthest_idx = np.argmax(min_distances)
                        idx = remaining_indices[farthest_idx]

                    selected_indices.append(idx)
                    remaining_indices = np.setdiff1d(remaining_indices, [idx])

                folds[fold_idx] = np.array(selected_indices)

        return folds

    def get_spatial_folds(self, coordinates: np.ndarray) -> List[np.ndarray]:
        """
        Get spatial cross-validation folds.

        Args:
            coordinates: Array of (lat, lon) coordinates

        Returns:
            List of arrays with sample indices for each fold
        """
        if self.method == 'geographic_blocks':
            # Use geographic blocking
            block_ids = self.create_geographic_blocks(coordinates, self.block_size)

            # Create folds ensuring each block is in only one fold
            unique_blocks = np.unique(block_ids)
            n_blocks = len(unique_blocks)

            if n_blocks < self.n_splits:
                logger.warning(f"Only {n_blocks} blocks available, using {n_blocks} folds")
                n_folds = min(n_blocks, self.n_splits)
            else:
                n_folds = self.n_splits

            # Distribute blocks across folds
            block_folds = np.random.RandomState(self.random_state).choice(
                n_folds, size=n_blocks, replace=True
            )

            folds = []
            for fold_idx in range(n_folds):
                fold_blocks = unique_blocks[block_folds == fold_idx]
                fold_indices = np.where(np.isin(block_ids, fold_blocks))[0]
                folds.append(fold_indices)

        elif self.method == 'kmeans_clusters':
            # Use spatial clustering
            clusters = self.create_spatial_clusters(coordinates, self.n_splits)

            folds = []
            for fold_idx in range(self.n_splits):
                fold_indices = np.where(clusters == fold_idx)[0]
                folds.append(fold_indices)

        elif self.method == 'distance_based':
            # Use distance-based splitting
            folds = self.create_distance_based_folds(coordinates, self.n_splits)

        else:
            raise ValueError(f"Unknown spatial CV method: {self.method}")

        # Ensure all samples are assigned to folds
        all_assigned = np.concatenate(folds)
        if len(all_assigned) != len(coordinates):
            logger.warning(f"Not all samples assigned to folds. Missing: {len(coordinates) - len(all_assigned)}")

        logger.info(f"Created {len(folds)} spatial folds")
        for i, fold in enumerate(folds):
            logger.info(f"Fold {i}: {len(fold)} samples")

        return folds

    def cross_validate_spatial(self, model, X: pd.DataFrame, y: pd.Series,
                             coordinates: np.ndarray) -> Dict[str, float]:
        """
        Perform spatial cross-validation.

        Args:
            model: Model instance
            X: Feature matrix
            y: Target variable
            coordinates: Array of (lat, lon) coordinates

        Returns:
            Dictionary with CV results
        """
        logger.info(f"Starting spatial cross-validation with {self.method}")

        # Get spatial folds
        folds = self.get_spatial_folds(coordinates)

        # Perform cross-validation
        cv_scores = []
        fold_metrics = []

        for fold_idx, test_indices in enumerate(folds):
            logger.info(f"Training fold {fold_idx + 1}/{len(folds)}")

            # Split data
            train_indices = np.setdiff1d(np.arange(len(X)), test_indices)

            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

            # Train model
            if hasattr(model, 'fit'):
                model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            # Calculate metrics
            fold_metric = {
                'fold': fold_idx,
                'test_samples': len(test_indices),
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }

            fold_metrics.append(fold_metric)
            cv_scores.append(fold_metric['roc_auc'])

        # Calculate summary statistics
        cv_scores = np.array(cv_scores)
        results = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_min': cv_scores.min(),
            'cv_max': cv_scores.max(),
            'fold_metrics': fold_metrics,
            'n_folds': len(folds)
        }

        logger.info(f"Spatial CV results: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")

        return results

    def plot_spatial_folds(self, coordinates: np.ndarray,
                          fold_assignments: List[np.ndarray],
                          save_path: Optional[str] = None):
        """
        Plot spatial distribution of cross-validation folds.

        Args:
            coordinates: Array of (lat, lon) coordinates
            fold_assignments: List of arrays with fold indices
            save_path: Optional path to save plot
        """
        logger.info("Creating spatial folds visualization")

        plt.figure(figsize=(12, 8))

        # Create color map for folds
        colors = plt.cm.tab10(np.linspace(0, 1, len(fold_assignments)))

        for fold_idx, test_indices in enumerate(fold_assignments):
            # Plot training points (light color)
            train_indices = np.setdiff1d(np.arange(len(coordinates)), test_indices)
            plt.scatter(coordinates[train_indices, 1], coordinates[train_indices, 0],
                       c=[colors[fold_idx]], alpha=0.3, s=20, label=f'Train Fold {fold_idx}')

            # Plot test points (dark color)
            plt.scatter(coordinates[test_indices, 1], coordinates[test_indices, 0],
                       c=[colors[fold_idx]], alpha=0.8, s=40, label=f'Test Fold {fold_idx}')

        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'Spatial Cross-Validation Folds ({self.method})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved spatial folds plot to {save_path}")

        plt.show()

    def create_geographic_confusion_matrix(self, y_true: np.ndarray,
                                         y_pred: np.ndarray,
                                         coordinates: np.ndarray,
                                         bins: int = 10) -> pd.DataFrame:
        """
        Create geographically-binned confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            coordinates: Array of (lat, lon) coordinates
            bins: Number of geographic bins

        Returns:
            DataFrame with geographic confusion matrix
        """
        logger.info("Creating geographic confusion matrix")

        # Create geographic bins
        lat_bins = np.linspace(coordinates[:, 0].min(), coordinates[:, 0].max(), bins + 1)
        lon_bins = np.linspace(coordinates[:, 1].min(), coordinates[:, 1].max(), bins + 1)

        # Assign samples to geographic bins
        lat_indices = np.digitize(coordinates[:, 0], lat_bins) - 1
        lon_indices = np.digitize(coordinates[:, 1], lon_bins) - 1

        # Clip to valid range
        lat_indices = np.clip(lat_indices, 0, bins - 1)
        lon_indices = np.clip(lon_indices, 0, bins - 1)

        # Create confusion matrix for each geographic bin
        geo_cm = np.zeros((bins, bins, 2, 2))  # (lat_bin, lon_bin, true_class, pred_class)

        for i in range(len(y_true)):
            lat_idx = lat_indices[i]
            lon_idx = lon_indices[i]
            true_class = int(y_true[i])
            pred_class = int(y_pred[i])

            geo_cm[lat_idx, lon_idx, true_class, pred_class] += 1

        # Convert to DataFrame for easier analysis
        results = []
        for lat_idx in range(bins):
            for lon_idx in range(bins):
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

    def plot_geographic_performance(self, geo_cm_df: pd.DataFrame,
                                  metric: str = 'accuracy',
                                  save_path: Optional[str] = None):
        """
        Plot geographic performance variation.

        Args:
            geo_cm_df: Geographic confusion matrix DataFrame
            metric: Metric to plot ('accuracy', 'precision', 'recall')
            save_path: Optional path to save plot
        """
        logger.info(f"Creating geographic performance plot for {metric}")

        plt.figure(figsize=(12, 8))

        # Filter out empty bins
        plot_data = geo_cm_df[geo_cm_df['total_samples'] > 0].copy()

        if len(plot_data) == 0:
            logger.warning("No data to plot for geographic performance")
            return

        scatter = plt.scatter(plot_data['lon_center'], plot_data['lat_center'],
                            c=plot_data[metric], cmap='viridis', s=plot_data['total_samples'],
                            alpha=0.7, edgecolors='black', linewidth=0.5)

        plt.colorbar(scatter, label=metric.title())
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'Geographic Variation in Model {metric.title()}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved geographic performance plot to {save_path}")

        plt.show()


def create_spatial_cv_comparison(models: List[Any],
                               X: pd.DataFrame,
                               y: pd.Series,
                               coordinates: np.ndarray) -> pd.DataFrame:
    """
    Compare different spatial CV methods.

    Args:
        models: List of model instances
        X: Feature matrix
        y: Target variable
        coordinates: Array of (lat, lon) coordinates

    Returns:
        DataFrame with CV method comparison
    """
    logger.info("Comparing spatial cross-validation methods")

    cv_methods = [
        'geographic_blocks',
        'kmeans_clusters',
        'distance_based'
    ]

    results = []

    for method in cv_methods:
        logger.info(f"Testing {method} method")

        cv = SpatialCrossValidator(
            method=method,
            n_splits=5,
            random_state=42
        )

        for model in models:
            # Reset model for fair comparison
            if hasattr(model, 'random_state'):
                model.random_state = 42

            try:
                cv_results = cv.cross_validate_spatial(model, X, y, coordinates)

                result = {
                    'cv_method': method,
                    'model': model.__class__.__name__,
                    'cv_mean': cv_results['cv_mean'],
                    'cv_std': cv_results['cv_std'],
                    'n_folds': cv_results['n_folds']
                }

                results.append(result)

            except Exception as e:
                logger.warning(f"Error with {method} and {model.__class__.__name__}: {e}")
                continue

    comparison_df = pd.DataFrame(results)

    logger.info("Spatial CV comparison completed")
    logger.info(f"\n{comparison_df.round(4)}")

    return comparison_df


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    n_samples = 100
    coordinates = np.random.uniform([4, -75], [12, -67], (n_samples, 2))
    y = np.random.choice([0, 1], n_samples)

    # Test spatial CV
    cv = SpatialCrossValidator(method='geographic_blocks', n_splits=5)
    folds = cv.get_spatial_folds(coordinates)

    logger.info(f"Created {len(folds)} spatial folds")
    for i, fold in enumerate(folds):
        logger.info(f"Fold {i}: {len(fold)} samples")
