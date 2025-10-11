#!/usr/bin/env python3
"""
GeoAuPredict Phase 3: Predictive Modeling - Prediction and Probability Mapping

This module creates probability maps and rasters for gold presence prediction:
- Convert point predictions to raster surfaces
- Create probability heatmaps
- Generate uncertainty maps
- Export prediction rasters for GIS integration

Author: GeoAuPredict Team
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import rasterio
from rasterio.transform import from_bounds
from scipy.interpolate import griddata
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


class ProbabilityMapper:
    """
    Creates probability maps and prediction surfaces from point predictions.

    Key Features:
    - Multiple interpolation methods (IDW, Kriging, Spline)
    - Raster creation and export
    - Uncertainty quantification
    - Visualization of prediction surfaces
    """

    def __init__(self,
                 interpolation_method: str = 'kriging',
                 pixel_size: float = 100.0,
                 uncertainty_estimation: bool = True):
        """
        Initialize probability mapper.

        Args:
            interpolation_method: Method for spatial interpolation ('idw', 'kriging', 'spline', 'nearest')
            pixel_size: Output raster pixel size in meters
            uncertainty_estimation: Whether to estimate prediction uncertainty
        """
        self.interpolation_method = interpolation_method
        self.pixel_size = pixel_size
        self.uncertainty_estimation = uncertainty_estimation

        logger.info(f"Initialized ProbabilityMapper with {interpolation_method} method")

    def create_prediction_grid(self, bounds: Tuple[float, float, float, float],
                             pixel_size: float) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Create regular grid for prediction.

        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat)
            pixel_size: Grid spacing in meters

        Returns:
            Tuple of (X_grid, Y_grid, width, height)
        """
        min_lon, min_lat, max_lon, max_lat = bounds

        # Calculate grid dimensions
        width = int((max_lon - min_lon) / (pixel_size / 111320))  # Rough degrees per meter
        height = int((max_lat - min_lat) / (pixel_size / 111320))

        # Ensure minimum dimensions
        width = max(width, 50)
        height = max(height, 50)

        # Create coordinate grids
        x_grid = np.linspace(min_lon, max_lon, width)
        y_grid = np.linspace(min_lat, max_lat, height)

        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

        logger.info(f"Created prediction grid: {width}x{height} = {width*height} pixels")

        return X_grid, Y_grid, width, height

    def inverse_distance_weighting(self, points: np.ndarray,
                                 values: np.ndarray,
                                 grid_x: np.ndarray,
                                 grid_y: np.ndarray,
                                 power: int = 2) -> np.ndarray:
        """
        Perform Inverse Distance Weighting interpolation.

        Args:
            points: Array of (x, y) coordinates
            values: Values at each point
            grid_x, grid_y: Grid coordinates for prediction
            power: Power parameter for IDW

        Returns:
            Interpolated values on grid
        """
        logger.info(f"Performing IDW interpolation with power={power}")

        grid_points = np.column_stack([grid_x.flatten(), grid_y.flatten()])
        n_grid = grid_points.shape[0]

        # Calculate distances
        distances = np.zeros((n_grid, len(points)))
        for i, grid_point in enumerate(grid_points):
            distances[i] = np.sqrt(np.sum((points - grid_point)**2, axis=1))

        # Avoid division by zero
        distances = np.where(distances == 0, 1e-10, distances)

        # Calculate weights (inverse distance)
        weights = 1.0 / (distances ** power)

        # Normalize weights
        weight_sums = np.sum(weights, axis=1, keepdims=True)
        weights = weights / weight_sums

        # Interpolate
        interpolated = np.sum(weights * values, axis=1)

        return interpolated.reshape(grid_y.shape)

    def kriging_interpolation(self, points: np.ndarray,
                            values: np.ndarray,
                            grid_x: np.ndarray,
                            grid_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Kriging interpolation with uncertainty estimation.

        Args:
            points: Array of (x, y) coordinates
            values: Values at each point
            grid_x, grid_y: Grid coordinates for prediction

        Returns:
            Tuple of (interpolated_values, uncertainty)
        """
        logger.info("Performing Kriging interpolation")

        # Set up Gaussian Process
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

        # Fit GP model
        gp.fit(points, values)

        # Predict on grid
        grid_points = np.column_stack([grid_x.flatten(), grid_y.flatten()])
        predictions = gp.predict(grid_points, return_std=True)

        interpolated = predictions[0].reshape(grid_y.shape)
        uncertainty = predictions[1].reshape(grid_y.shape)

        return interpolated, uncertainty

    def nearest_neighbor_interpolation(self, points: np.ndarray,
                                     values: np.ndarray,
                                     grid_x: np.ndarray,
                                     grid_y: np.ndarray) -> np.ndarray:
        """
        Perform nearest neighbor interpolation.

        Args:
            points: Array of (x, y) coordinates
            values: Values at each point
            grid_x, grid_y: Grid coordinates for prediction

        Returns:
            Interpolated values on grid
        """
        logger.info("Performing nearest neighbor interpolation")

        grid_points = np.column_stack([grid_x.flatten(), grid_y.flatten()])

        # Simple nearest neighbor - find closest point for each grid cell
        interpolated = np.zeros(grid_points.shape[0])

        for i, grid_point in enumerate(grid_points):
            distances = np.sqrt(np.sum((points - grid_point)**2, axis=1))
            closest_idx = np.argmin(distances)
            interpolated[i] = values[closest_idx]

        return interpolated.reshape(grid_y.shape)

    def create_probability_raster(self, predictions_df: pd.DataFrame,
                                bounds: Optional[Tuple[float, float, float, float]] = None,
                                method: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Create probability raster from point predictions.

        Args:
            predictions_df: DataFrame with lat, lon, probability columns
            bounds: Optional (min_lon, min_lat, max_lon, max_lat)
            method: Override interpolation method

        Returns:
            Dictionary with prediction and uncertainty rasters
        """
        logger.info("Creating probability raster from point predictions")

        if bounds is None:
            # Use data bounds with small buffer
            bounds = (
                predictions_df['lon'].min() - 0.01,
                predictions_df['lat'].min() - 0.01,
                predictions_df['lon'].max() + 0.01,
                predictions_df['lat'].max() + 0.01
            )

        method = method or self.interpolation_method

        # Extract coordinates and values
        points = predictions_df[['lat', 'lon']].values
        values = predictions_df['probability'].values

        # Create prediction grid
        x_grid, y_grid, width, height = self.create_prediction_grid(bounds, self.pixel_size)

        # Perform interpolation
        if method == 'idw':
            probability_raster = self.inverse_distance_weighting(points, values, x_grid, y_grid)

            # For IDW, use simple uncertainty estimate
            uncertainty_raster = np.ones_like(probability_raster) * 0.1

        elif method == 'kriging':
            probability_raster, uncertainty_raster = self.kriging_interpolation(points, values, x_grid, y_grid)

        elif method == 'nearest':
            probability_raster = self.nearest_neighbor_interpolation(points, values, x_grid, y_grid)
            uncertainty_raster = np.ones_like(probability_raster) * 0.2

        else:
            raise ValueError(f"Unknown interpolation method: {method}")

        # Clip probabilities to [0, 1]
        probability_raster = np.clip(probability_raster, 0, 1)

        logger.info(f"Created probability raster: {probability_raster.shape}")

        return {
            'probability': probability_raster,
            'uncertainty': uncertainty_raster,
            'x_grid': x_grid,
            'y_grid': y_grid,
            'bounds': bounds
        }

    def save_probability_raster(self, raster_data: Dict[str, np.ndarray],
                              output_path: str,
                              crs: str = 'EPSG:4326') -> str:
        """
        Save probability raster to GeoTIFF file.

        Args:
            raster_data: Dictionary with raster data
            output_path: Output file path
            crs: Coordinate reference system

        Returns:
            Path to saved file
        """
        logger.info(f"Saving probability raster to {output_path}")

        probability = raster_data['probability']
        bounds = raster_data['bounds']
        height, width = probability.shape

        # Create transform
        transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)

        # Prepare output directory
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save probability raster
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype='float32',
            crs=crs,
            transform=transform,
            nodata=-9999
        ) as dst:
            dst.write(probability.astype(np.float32), 1)

        # Save uncertainty raster if available
        if 'uncertainty' in raster_data:
            uncertainty_path = output_path.parent / f"{output_path.stem}_uncertainty.tif"
            with rasterio.open(
                uncertainty_path, 'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype='float32',
                crs=crs,
                transform=transform,
                nodata=-9999
            ) as dst:
                dst.write(raster_data['uncertainty'].astype(np.float32), 1)

            logger.info(f"Saved uncertainty raster: {uncertainty_path}")

        logger.info(f"Saved probability raster: {output_path}")

        return str(output_path)

    def create_probability_categories(self, probability_raster: np.ndarray,
                                    thresholds: List[float] = [0.3, 0.5, 0.7, 0.9]) -> np.ndarray:
        """
        Categorize probability values into classes.

        Args:
            probability_raster: Probability values
            thresholds: Threshold values for categorization

        Returns:
            Categorized probability raster
        """
        categories = np.zeros_like(probability_raster, dtype=int)

        for i, threshold in enumerate(sorted(thresholds)):
            categories[probability_raster >= threshold] = i + 1

        return categories

    def plot_probability_map(self, raster_data: Dict[str, np.ndarray],
                           title: str = "Gold Presence Probability",
                           save_path: Optional[str] = None,
                           show_uncertainty: bool = False):
        """
        Plot probability map.

        Args:
            raster_data: Dictionary with raster data
            title: Plot title
            save_path: Optional path to save plot
            show_uncertainty: Whether to show uncertainty overlay
        """
        logger.info("Creating probability map visualization")

        fig, ax = plt.subplots(figsize=(12, 8))

        probability = raster_data['probability']

        # Plot probability heatmap
        im = ax.imshow(probability, extent=raster_data['bounds'],
                      origin='lower', cmap='YlOrRd', alpha=0.8)

        # Add uncertainty overlay if requested
        if show_uncertainty and 'uncertainty' in raster_data:
            uncertainty = raster_data['uncertainty']
            # Create uncertainty mask (darker where more uncertain)
            uncertainty_mask = np.ma.masked_where(uncertainty < 0.2, uncertainty)
            im2 = ax.imshow(uncertainty_mask, extent=raster_data['bounds'],
                          origin='lower', cmap='Greys', alpha=0.3, vmin=0, vmax=0.5)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Probability')

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved probability map to {save_path}")

        plt.show()

    def create_exploration_priority_map(self, probability_raster: np.ndarray,
                                      accessibility_raster: Optional[np.ndarray] = None,
                                      threshold: float = 0.5) -> np.ndarray:
        """
        Create exploration priority map combining probability with accessibility.

        Args:
            probability_raster: Gold probability values
            accessibility_raster: Accessibility/cost surface (lower = more accessible)
            threshold: Minimum probability threshold

        Returns:
            Priority raster (higher = higher priority)
        """
        logger.info("Creating exploration priority map")

        # Start with probability (only above threshold)
        priority = np.where(probability_raster >= threshold, probability_raster, 0)

        # Incorporate accessibility if provided
        if accessibility_raster is not None:
            # Lower accessibility cost = higher priority
            # Normalize accessibility to [0, 1] (1 = most accessible)
            accessibility_norm = 1 - (accessibility_raster - accessibility_raster.min()) / (accessibility_raster.max() - accessibility_raster.min())
            accessibility_norm = np.clip(accessibility_norm, 0, 1)

            # Combine probability and accessibility
            priority = priority * accessibility_norm

        return priority

    def extract_raster_statistics(self, raster: np.ndarray,
                                mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract statistics from probability raster.

        Args:
            raster: Input raster
            mask: Optional mask for valid areas

        Returns:
            Dictionary with raster statistics
        """
        if mask is not None:
            valid_pixels = raster[mask]
        else:
            valid_pixels = raster[~np.isnan(raster)]

        if len(valid_pixels) == 0:
            return {'error': 'No valid pixels'}

        stats = {
            'mean': float(np.mean(valid_pixels)),
            'std': float(np.std(valid_pixels)),
            'min': float(np.min(valid_pixels)),
            'max': float(np.max(valid_pixels)),
            'median': float(np.median(valid_pixels)),
            'q25': float(np.percentile(valid_pixels, 25)),
            'q75': float(np.percentile(valid_pixels, 75)),
            'q90': float(np.percentile(valid_pixels, 90)),
            'n_pixels': len(valid_pixels),
            'area_above_05': float(np.sum(valid_pixels >= 0.5)),
            'area_above_07': float(np.sum(valid_pixels >= 0.7)),
            'area_above_09': float(np.sum(valid_pixels >= 0.9))
        }

        logger.info(f"Raster statistics: mean={stats['mean']:.3f}, max={stats['max']:.3f}")

        return stats


def create_sample_predictions(n_samples: int = 100) -> pd.DataFrame:
    """
    Create sample prediction data for testing.

    Args:
        n_samples: Number of sample points

    Returns:
        DataFrame with sample predictions
    """
    logger.info(f"Creating {n_samples} sample predictions")

    np.random.seed(42)

    # Create sample locations in Colombia
    sample_data = {
        'lat': np.random.uniform(4.0, 12.0, n_samples),
        'lon': np.random.uniform(-75.0, -67.0, n_samples),
        'probability': np.random.uniform(0.1, 0.9, n_samples),
        'uncertainty': np.random.uniform(0.05, 0.3, n_samples)
    }

    # Add some spatial correlation
    # Higher probabilities in certain regions
    for i in range(n_samples):
        lat, lon = sample_data['lat'][i], sample_data['lon'][i]

        # Add regional effects
        if lat > 7.0 and lon < -72.0:  # Northwestern region
            sample_data['probability'][i] += 0.2
        elif lat < 6.0 and lon > -70.0:  # Southeastern region
            sample_data['probability'][i] += 0.15

    # Clip to [0, 1]
    sample_data['probability'] = np.clip(sample_data['probability'], 0, 1)

    return pd.DataFrame(sample_data)


# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_predictions = create_sample_predictions(200)

    # Initialize mapper
    mapper = ProbabilityMapper(interpolation_method='kriging', pixel_size=1000)

    # Define bounds
    bounds = (-75.0, 4.0, -67.0, 12.0)  # Colombia bounds

    # Create probability raster
    raster_data = mapper.create_probability_raster(sample_predictions, bounds)

    # Save raster
    output_path = "sample_probability_map.tif"
    saved_path = mapper.save_probability_raster(raster_data, output_path)

    # Plot map
    mapper.plot_probability_map(raster_data, "Sample Gold Probability Map")

    # Get statistics
    stats = mapper.extract_raster_statistics(raster_data['probability'])
    logger.info(f"Sample map statistics: {stats}")
