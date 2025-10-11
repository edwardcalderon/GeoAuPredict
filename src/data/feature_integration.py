"""
Feature Integration Module for Phase 2 Geospatial Feature Engineering

This module integrates spectral indices, terrain variables, and geological data
into a unified tabular dataset for machine learning applications.

Features include:
- Spectral indices (NDVI, NBR, Clay Index, Iron Index)
- Terrain variables (elevation, slope, curvature, aspect, TWI)
- Geological variables (geochemistry, distances to features)
- Spatial coordinates and metadata
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.transform import rowcol
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
import warnings

from .spectral_indices import SpectralIndicesCalculator
from .terrain_analysis import TerrainAnalyzer
from .geological_processing import GeologicalProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureIntegrator:
    """Integrator for combining multiple geospatial feature types."""

    def __init__(self, pixel_size: float = 100.0):
        """
        Initialize the feature integrator.

        Args:
            pixel_size: Pixel size in meters for raster sampling
        """
        self.pixel_size = pixel_size
        self.spectral_calculator = SpectralIndicesCalculator()
        self.terrain_analyzer = TerrainAnalyzer()
        self.geological_processor = GeologicalProcessor()

    def sample_raster_at_points(self, raster_path: str,
                               points_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Sample raster values at point locations.

        Args:
            raster_path: Path to raster file
            points_gdf: GeoDataFrame with point locations

        Returns:
            DataFrame with sampled values
        """
        logger.info(f"Sampling raster {raster_path} at {len(points_gdf)} points...")

        if not Path(raster_path).exists():
            logger.warning(f"Raster file not found: {raster_path}")
            # Return NaN values for missing raster
            return pd.DataFrame({
                Path(raster_path).stem: [np.nan] * len(points_gdf)
            })

        try:
            with rasterio.open(raster_path) as src:
                # Convert points to raster coordinates
                sampled_values = []

                for _, point in points_gdf.iterrows():
                    # Get row, col for point
                    try:
                        row, col = rowcol(src.transform, point.geometry.x, point.geometry.y)
                        if 0 <= row < src.height and 0 <= col < src.width:
                            value = src.read(1)[row, col]
                            if value == src.nodata:
                                sampled_values.append(np.nan)
                            else:
                                sampled_values.append(float(value))
                        else:
                            sampled_values.append(np.nan)
                    except:
                        sampled_values.append(np.nan)

                logger.info(f"Sampled {len([v for v in sampled_values if not np.isnan(v)])} valid values")

                return pd.DataFrame({
                    Path(raster_path).stem: sampled_values
                })

        except Exception as e:
            logger.error(f"Error sampling raster {raster_path}: {e}")
            return pd.DataFrame({
                Path(raster_path).stem: [np.nan] * len(points_gdf)
            })

    def integrate_spectral_features(self, spectral_indices: Dict[str, str],
                                  sample_points: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Integrate spectral indices at sample point locations.

        Args:
            spectral_indices: Dictionary mapping index names to raster file paths
            sample_points: GeoDataFrame with sample locations

        Returns:
            DataFrame with spectral features
        """
        logger.info("Integrating spectral features...")

        spectral_features = []

        for index_name, raster_path in spectral_indices.items():
            sampled_df = self.sample_raster_at_points(raster_path, sample_points)
            spectral_features.append(sampled_df)

        # Combine all spectral features
        if spectral_features:
            combined_spectral = pd.concat(spectral_features, axis=1)
            logger.info(f"Integrated {len(combined_spectral.columns)} spectral features")
            return combined_spectral
        else:
            logger.warning("No spectral features integrated")
            return pd.DataFrame()

    def integrate_terrain_features(self, terrain_variables: Dict[str, str],
                                 sample_points: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Integrate terrain variables at sample point locations.

        Args:
            terrain_variables: Dictionary mapping variable names to raster file paths
            sample_points: GeoDataFrame with sample locations

        Returns:
            DataFrame with terrain features
        """
        logger.info("Integrating terrain features...")

        terrain_features = []

        for var_name, raster_path in terrain_variables.items():
            sampled_df = self.sample_raster_at_points(raster_path, sample_points)
            terrain_features.append(sampled_df)

        # Combine all terrain features
        if terrain_features:
            combined_terrain = pd.concat(terrain_features, axis=1)
            logger.info(f"Integrated {len(combined_terrain.columns)} terrain features")
            return combined_terrain
        else:
            logger.warning("No terrain features integrated")
            return pd.DataFrame()

    def integrate_geological_features(self, geological_data_path: str) -> pd.DataFrame:
        """
        Load and return geological features data.

        Args:
            geological_data_path: Path to processed geological data CSV

        Returns:
            DataFrame with geological features
        """
        logger.info(f"Loading geological features from {geological_data_path}...")

        if not Path(geological_data_path).exists():
            logger.warning(f"Geological data file not found: {geological_data_path}")
            return pd.DataFrame()

        try:
            geological_df = pd.read_csv(geological_data_path)

            # Remove geometry column if it exists (not needed in final dataset)
            if 'geometry' in geological_df.columns:
                geological_df = geological_df.drop('geometry', axis=1)

            logger.info(f"Loaded {len(geological_df.columns)} geological features")
            return geological_df

        except Exception as e:
            logger.error(f"Error loading geological data: {e}")
            return pd.DataFrame()

    def create_spatial_grid(self, bounds: Tuple[float, float, float, float],
                          pixel_size: float) -> gpd.GeoDataFrame:
        """
        Create a regular grid of points within the study area.

        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat)
            pixel_size: Grid spacing in meters

        Returns:
            GeoDataFrame with grid points
        """
        logger.info(f"Creating spatial grid with {pixel_size}m spacing...")

        min_lon, min_lat, max_lon, max_lat = bounds

        # Approximate degrees per meter (varies by latitude)
        meters_per_degree = 111320  # Approximate at equator

        # Calculate number of points
        lon_points = int((max_lon - min_lon) * meters_per_degree / pixel_size) + 1
        lat_points = int((max_lat - min_lat) * meters_per_degree / pixel_size) + 1

        # Create coordinate arrays
        lon_coords = np.linspace(min_lon, max_lon, lon_points)
        lat_coords = np.linspace(min_lat, max_lat, lat_points)

        # Create grid points
        grid_points = []
        for lat in lat_coords:
            for lon in lon_coords:
                grid_points.append(Point(lon, lat))

        # Create GeoDataFrame
        grid_gdf = gpd.GeoDataFrame(
            {'grid_id': range(len(grid_points))},
            geometry=grid_points,
            crs='EPSG:4326'
        )

        logger.info(f"Created {len(grid_gdf)} grid points")
        return grid_gdf

    def create_comprehensive_dataset(self,
                                   spectral_indices: Dict[str, str],
                                   terrain_variables: Dict[str, str],
                                   geological_data_path: str,
                                   bounds: Optional[Tuple[float, float, float, float]] = None,
                                   use_grid: bool = True,
                                   sample_points: Optional[gpd.GeoDataFrame] = None) -> pd.DataFrame:
        """
        Create comprehensive dataset integrating all feature types.

        Args:
            spectral_indices: Dictionary of spectral index rasters
            terrain_variables: Dictionary of terrain variable rasters
            geological_data_path: Path to geological data CSV
            bounds: Study area bounds for grid creation
            use_grid: Whether to create a regular grid or use provided points
            sample_points: Custom sample points (if not using grid)

        Returns:
            Comprehensive feature dataset
        """
        logger.info("Creating comprehensive geospatial dataset...")

        # Determine sampling points
        if use_grid and bounds:
            sample_points = self.create_spatial_grid(bounds, self.pixel_size)
        elif sample_points is None:
            raise ValueError("Either provide bounds for grid creation or custom sample points")

        # Integrate features
        spectral_df = self.integrate_spectral_features(spectral_indices, sample_points)
        terrain_df = self.integrate_terrain_features(terrain_variables, sample_points)
        geological_df = self.integrate_geological_features(geological_data_path)

        # Combine all features
        feature_dfs = []

        # Add spatial coordinates
        coords_df = pd.DataFrame({
            'longitude': [p.x for p in sample_points.geometry],
            'latitude': [p.y for p in sample_points.geometry],
            'grid_id': getattr(sample_points, 'grid_id', range(len(sample_points)))
        })
        feature_dfs.append(coords_df)

        # Add other feature types if available
        if not spectral_df.empty:
            feature_dfs.append(spectral_df)
        if not terrain_df.empty:
            feature_dfs.append(terrain_df)
        if not geological_df.empty:
            feature_dfs.append(geological_df)

        # Combine all features
        if feature_dfs:
            combined_df = pd.concat(feature_dfs, axis=1)

            # Remove duplicate columns
            combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

            logger.info(f"Created dataset with {len(combined_df)} samples and {len(combined_df.columns)} features")
            return combined_df
        else:
            logger.warning("No features available for dataset creation")
            return pd.DataFrame()

    def save_dataset(self, dataset: pd.DataFrame, output_path: str,
                    format: str = 'csv') -> str:
        """
        Save the integrated dataset to file.

        Args:
            dataset: DataFrame to save
            output_path: Output file path
            format: File format ('csv', 'parquet', 'feather')

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if format == 'csv':
                dataset.to_csv(output_path, index=False)
            elif format == 'parquet':
                dataset.to_parquet(output_path, index=False)
            elif format == 'feather':
                dataset.to_feather(output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Saved dataset ({len(dataset)} samples, {len(dataset.columns)} features) to {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            raise

    def create_sample_integration_data(self, output_dir: str = "sample_data") -> Dict[str, str]:
        """
        Create sample data for all feature types for integration testing.

        Args:
            output_dir: Directory to save sample data

        Returns:
            Dictionary of created file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Create sample spectral indices
        from src.data.spectral_indices import create_sample_spectral_data
        sample_bands = create_sample_spectral_data(str(output_path))

        spectral_calc = SpectralIndicesCalculator()

        spectral_indices = {
            'ndvi': str(output_path / "ndvi.tif"),
            'nbr': str(output_path / "nbr.tif"),
            'clay_index': str(output_path / "clay_index.tif"),
            'iron_index': str(output_path / "iron_index.tif")
        }

        # Create sample terrain variables
        from src.data.terrain_analysis import create_sample_dem
        dem_file = create_sample_dem(str(output_path))

        terrain_analyzer = TerrainAnalyzer()
        terrain_vars, profile = terrain_analyzer.calculate_all_terrain_variables(dem_file)
        terrain_analyzer.save_terrain_variables(terrain_vars, str(output_path), profile)

        terrain_variables = {
            var_name: str(output_path / f"{var_name}.tif")
            for var_name in terrain_vars.keys()
        }

        # Create sample geological data
        geo_processor = GeologicalProcessor()
        geo_files = geo_processor.create_sample_geological_data(str(output_path))

        logger.info(f"Created sample integration data in {output_path}")
        return {
            'spectral_indices': spectral_indices,
            'terrain_variables': terrain_variables,
            'geological_data': geo_files['geochemical_csv'],
            'sample_points': str(output_path / "sample_points.geojson")
        }


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    integrator = FeatureIntegrator(pixel_size=1000)  # 1km grid

    sample_files = integrator.create_sample_integration_data()

    # Create comprehensive dataset
    bounds = (-75.0, 4.0, -67.0, 12.0)  # Colombia bounds

    dataset = integrator.create_comprehensive_dataset(
        spectral_indices=sample_files['spectral_indices'],
        terrain_variables=sample_files['terrain_variables'],
        geological_data_path=sample_files['geological_data'],
        bounds=bounds,
        use_grid=True
    )

    print("Created comprehensive dataset:")
    print(f"  Samples: {len(dataset)}")
    print(f"  Features: {len(dataset.columns)}")
    print(f"  Feature types: {dataset.dtypes.unique()}")

    # Save dataset
    saved_path = integrator.save_dataset(dataset, "integrated_dataset.csv")
    print(f"Saved dataset to: {saved_path}")

    # Display feature summary
    print("\nFeature summary:")
    print(dataset.info())
