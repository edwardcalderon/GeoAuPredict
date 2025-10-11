"""
Geological Variables Processing Module for Phase 2 Geospatial Feature Engineering

This module processes geological data for gold exploration:
- Geochemical analysis (elements associated with gold: Fe, As, Cu, Pb, Zn)
- Distance calculations to geological features (faults, rivers, geological contacts)
- Geological unit classification and proximity analysis

Integrates with geopandas and rasterio for spatial analysis.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
from scipy.spatial.distance import cdist
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeologicalProcessor:
    """Processor for geological variables in mineral exploration."""

    def __init__(self, study_area_bounds: Optional[Tuple[float, float, float, float]] = None):
        """
        Initialize the geological processor.

        Args:
            study_area_bounds: Optional (min_lon, min_lat, max_lon, max_lat) bounds
        """
        self.study_area_bounds = study_area_bounds
        self.gold_associated_elements = ['Fe', 'As', 'Cu', 'Pb', 'Zn', 'Au', 'Ag']

    def load_geochemical_data(self, csv_path: str) -> gpd.GeoDataFrame:
        """
        Load geochemical sample data from CSV file.

        Args:
            csv_path: Path to CSV file with geochemical data

        Returns:
            GeoDataFrame with geochemical samples
        """
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"Geochemical data file not found: {csv_path}")

        try:
            # Load CSV data
            df = pd.read_csv(csv_path)

            # Check for required columns
            required_cols = ['lat', 'lon']
            element_cols = [col for col in df.columns if col in self.gold_associated_elements]

            if not required_cols:
                raise ValueError("CSV must contain 'lat' and 'lon' columns")

            # Create GeoDataFrame
            geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

            logger.info(f"Loaded {len(gdf)} geochemical samples")
            logger.info(f"Available elements: {', '.join(element_cols)}")

            return gdf

        except Exception as e:
            logger.error(f"Error loading geochemical data: {e}")
            raise

    def load_geological_features(self, shapefile_path: str) -> gpd.GeoDataFrame:
        """
        Load geological features (faults, contacts, etc.) from shapefile.

        Args:
            shapefile_path: Path to shapefile or directory containing shapefiles

        Returns:
            GeoDataFrame with geological features
        """
        shapefile = Path(shapefile_path)

        if shapefile.is_dir():
            # Look for .shp files in directory
            shp_files = list(shapefile.glob("*.shp"))
            if not shp_files:
                raise FileNotFoundError(f"No shapefiles found in {shapefile}")
            shapefile = shp_files[0]  # Use first shapefile found

        try:
            gdf = gpd.read_file(shapefile)
            logger.info(f"Loaded {len(gdf)} geological features from {shapefile}")
            logger.info(f"Feature types: {gdf.geometry.type.unique()}")

            return gdf

        except Exception as e:
            logger.error(f"Error loading geological features: {e}")
            raise

    def calculate_element_ratios(self, geochemical_data: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Calculate geochemical ratios useful for gold exploration.

        Args:
            geochemical_data: GeoDataFrame with geochemical samples

        Returns:
            DataFrame with calculated ratios
        """
        logger.info("Calculating geochemical ratios...")

        ratios = {}
        element_cols = [col for col in geochemical_data.columns
                       if col in self.gold_associated_elements]

        # Path finder ratios (common in gold exploration)
        if 'As' in element_cols and 'Sb' in element_cols:
            ratios['as_sb_ratio'] = geochemical_data['As'] / geochemical_data['Sb']

        if 'Fe' in element_cols and 'Mg' in element_cols:
            ratios['fe_mg_ratio'] = geochemical_data['Fe'] / geochemical_data['Mg']

        if 'Cu' in element_cols and 'Zn' in element_cols:
            ratios['cu_zn_ratio'] = geochemical_data['Cu'] / geochemical_data['Zn']

        # Alteration indices
        if all(elem in element_cols for elem in ['Na', 'K', 'Ca']):
            # Ishikawa alteration index
            ratios['ishikawa_index'] = (geochemical_data['Na'] + geochemical_data['K']) / geochemical_data['Ca']

        # Arsenic pathfinder ratio
        if 'As' in element_cols and 'Au' in element_cols:
            ratios['as_au_ratio'] = geochemical_data['As'] / geochemical_data['Au']

        # Convert to DataFrame
        ratios_df = pd.DataFrame(ratios)

        # Handle infinite and NaN values
        ratios_df = ratios_df.replace([np.inf, -np.inf], np.nan)
        ratios_df = ratios_df.fillna(-9999)

        logger.info(f"Calculated {len(ratios_df.columns)} geochemical ratios")
        return ratios_df

    def calculate_distances_to_features(self, sample_points: gpd.GeoDataFrame,
                                     geological_features: gpd.GeoDataFrame,
                                     max_distance: float = 10000) -> pd.DataFrame:
        """
        Calculate distances from sample points to geological features.

        Args:
            sample_points: GeoDataFrame with sample locations
            geological_features: GeoDataFrame with geological features
            max_distance: Maximum distance to calculate (meters)

        Returns:
            DataFrame with distance calculations
        """
        logger.info("Calculating distances to geological features...")

        distances = {}

        # Convert to projected CRS for distance calculations (UTM Zone 18N for Colombia)
        sample_points_utm = sample_points.to_crs('EPSG:32618')
        features_utm = geological_features.to_crs('EPSG:32618')

        # Get coordinates
        sample_coords = np.array([(p.x, p.y) for p in sample_points_utm.geometry])
        
        # Handle different geometry types
        feature_coords = []
        for geom in features_utm.geometry:
            if geom.geom_type == 'Point':
                feature_coords.append((geom.x, geom.y))
            elif geom.geom_type == 'LineString':
                # For LineStrings, use centroid or representative point
                centroid = geom.centroid
                feature_coords.append((centroid.x, centroid.y))
            elif geom.geom_type == 'Polygon':
                # For Polygons, use centroid
                centroid = geom.centroid
                feature_coords.append((centroid.x, centroid.y))
            else:
                # Fallback: use bounds center
                bounds = geom.bounds
                center_x = (bounds[0] + bounds[2]) / 2
                center_y = (bounds[1] + bounds[3]) / 2
                feature_coords.append((center_x, center_y))
        
        feature_coords = np.array(feature_coords)

        if len(feature_coords) > 0:
            # Calculate distance matrix
            distance_matrix = cdist(sample_coords, feature_coords)

            # Get minimum distance to any feature
            distances['dist_to_nearest_feature'] = distance_matrix.min(axis=1)

            # Filter by maximum distance
            distances['dist_to_nearest_feature'] = np.where(
                distances['dist_to_nearest_feature'] > max_distance,
                max_distance,
                distances['dist_to_nearest_feature']
            )

            # Calculate distance statistics
            distances['dist_to_features_mean'] = distance_matrix.mean(axis=1)
            distances['dist_to_features_std'] = distance_matrix.std(axis=1)
            distances['num_features_within_1km'] = (distance_matrix <= 1000).sum(axis=1)
            distances['num_features_within_5km'] = (distance_matrix <= 5000).sum(axis=1)

        else:
            # No features available
            distances['dist_to_nearest_feature'] = np.full(len(sample_points), max_distance)
            distances['dist_to_features_mean'] = np.full(len(sample_points), max_distance)
            distances['dist_to_features_std'] = np.full(len(sample_points), 0)
            distances['num_features_within_1km'] = np.zeros(len(sample_points))
            distances['num_features_within_5km'] = np.zeros(len(sample_points))

        # Convert to DataFrame
        distances_df = pd.DataFrame(distances)

        logger.info(f"Distance statistics - Min: {distances_df['dist_to_nearest_feature'].min():.0f}m, "
                   f"Max: {distances_df['dist_to_nearest_feature'].max():.0f}m")

        return distances_df

    def create_geological_raster(self, geological_features: gpd.GeoDataFrame,
                               output_path: str, pixel_size: float = 100,
                               bounds: Optional[Tuple[float, float, float, float]] = None) -> str:
        """
        Create raster from geological features for proximity analysis.

        Args:
            geological_features: GeoDataFrame with geological features
            output_path: Path for output raster
            pixel_size: Pixel size in meters
            bounds: Optional (min_x, min_y, max_x, max_y) bounds

        Returns:
            Path to created raster file
        """
        logger.info("Creating geological raster...")

        if bounds is None:
            # Use feature bounds
            bounds = geological_features.total_bounds

        # Calculate raster dimensions
        width = int((bounds[2] - bounds[0]) / pixel_size)
        height = int((bounds[3] - bounds[1]) / pixel_size)
        
        # Ensure minimum dimensions
        width = max(width, 10)
        height = max(height, 10)
        
        logger.info(f"Raster bounds: {bounds}")
        logger.info(f"Pixel size: {pixel_size}, Width: {width}, Height: {height}")

        # Create transform
        transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)

        # Create geological raster (distance to features)
        geo_raster = np.full((height, width), np.inf, dtype=np.float32)

        # Convert features to UTM for distance calculations
        features_utm = geological_features.to_crs('EPSG:32618')

        # Create grid of coordinates
        rows, cols = np.indices((height, width))
        xs = bounds[0] + (cols + 0.5) * pixel_size
        ys = bounds[1] + (rows + 0.5) * pixel_size

        # Convert grid coordinates to UTM
        grid_points = gpd.GeoSeries([Point(x, y) for x, y in zip(xs.flatten(), ys.flatten())],
                                   crs='EPSG:4326').to_crs('EPSG:32618')

        grid_coords = np.array([(p.x, p.y) for p in grid_points])

        if len(features_utm) > 0:
            # Get feature coordinates (handle different geometry types)
            feature_coords = []
            for geom in features_utm.geometry:
                if geom.geom_type == 'Point':
                    feature_coords.append((geom.x, geom.y))
                elif geom.geom_type == 'LineString':
                    # For LineStrings, use centroid
                    centroid = geom.centroid
                    feature_coords.append((centroid.x, centroid.y))
                elif geom.geom_type == 'Polygon':
                    # For Polygons, use centroid
                    centroid = geom.centroid
                    feature_coords.append((centroid.x, centroid.y))
                else:
                    # Fallback: use bounds center
                    bounds = geom.bounds
                    center_x = (bounds[0] + bounds[2]) / 2
                    center_y = (bounds[1] + bounds[3]) / 2
                    feature_coords.append((center_x, center_y))
            
            feature_coords = np.array(feature_coords)

            # Debug: check array shape
            logger.info(f"Feature coords shape: {feature_coords.shape}")
            logger.info(f"Grid coords shape: {grid_coords.shape}")

            if len(feature_coords) == 0:
                logger.warning("No valid feature coordinates found")
            else:
                # Calculate distances
                distance_matrix = cdist(grid_coords, feature_coords)
                min_distances = distance_matrix.min(axis=1)

                # Reshape back to raster
                geo_raster = min_distances.reshape(height, width)

        # Save raster
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype='float32',
            crs='EPSG:32618',
            transform=transform,
            nodata=-9999
        ) as dst:
            dst.write(geo_raster.astype(np.float32), 1)

        logger.info(f"Created geological raster: {output_path}")
        return str(output_path)

    def process_geological_variables(self, geochemical_path: str,
                                   features_path: str,
                                   output_dir: str = "geological_output") -> Dict[str, Union[pd.DataFrame, str]]:
        """
        Process all geological variables from input data.

        Args:
            geochemical_path: Path to geochemical CSV data
            features_path: Path to geological features shapefile
            output_dir: Output directory for results

        Returns:
            Dictionary with processed geological data
        """
        logger.info("Starting geological variables processing...")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # Load geochemical data
        geochemical_gdf = self.load_geochemical_data(geochemical_path)
        results['geochemical_data'] = geochemical_gdf

        # Load geological features
        features_gdf = self.load_geological_features(features_path)
        results['geological_features'] = features_gdf

        # Calculate geochemical ratios
        ratios_df = self.calculate_element_ratios(geochemical_gdf)
        results['geochemical_ratios'] = ratios_df

        # Calculate distances to geological features
        distances_df = self.calculate_distances_to_features(geochemical_gdf, features_gdf)
        results['distances_to_features'] = distances_df

        # Create geological raster
        raster_path = self.create_geological_raster(
            features_gdf,
            output_path / "geological_raster.tif"
        )
        results['geological_raster'] = raster_path

        # Combine all geological variables
        combined_df = pd.concat([
            geochemical_gdf[['lat', 'lon', 'geometry']].reset_index(drop=True),
            ratios_df,
            distances_df
        ], axis=1)

        # Save combined data
        combined_path = output_path / "geological_variables.csv"
        combined_df.to_csv(combined_path, index=False)
        results['combined_geological_data'] = combined_path

        logger.info(f"Processed geological variables saved to {output_path}")
        return results

    def create_sample_geological_data(self, output_dir: str = "sample_data") -> Dict[str, str]:
        """
        Create sample geological data for testing purposes.

        Args:
            output_dir: Directory to save sample data

        Returns:
            Dictionary of created file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Create sample geochemical data
        np.random.seed(42)
        n_samples = 100

        sample_data = {
            'lat': np.random.uniform(4.0, 12.0, n_samples),  # Colombia bounds
            'lon': np.random.uniform(-75.0, -67.0, n_samples),
            'Au_ppm': np.random.exponential(0.1, n_samples),
            'As_ppm': np.random.exponential(5, n_samples),
            'Cu_ppm': np.random.exponential(50, n_samples),
            'Pb_ppm': np.random.exponential(20, n_samples),
            'Zn_ppm': np.random.exponential(80, n_samples),
            'Fe_pct': np.random.uniform(1, 10, n_samples),
            'label_gold': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        }

        geochemical_path = output_path / "sample_geochemical.csv"
        pd.DataFrame(sample_data).to_csv(geochemical_path, index=False)

        # Create sample geological features (faults)
        # Create a simple fault line
        fault_coords = [
            (-72.0, 6.0), (-71.0, 6.5), (-70.0, 7.0), (-69.0, 7.5)
        ]

        coords_str = ', '.join(f'{x} {y}' for x, y in fault_coords)
        features_data = {
            'geometry': [f'LINESTRING({coords_str})'],
            'type': ['fault'],
            'name': ['sample_fault']
        }

        features_path = output_path / "sample_features.geojson"
        features_gdf = gpd.GeoDataFrame(features_data, geometry=gpd.GeoSeries.from_wkt(features_data['geometry']), crs='EPSG:4326')
        features_gdf.to_file(features_path, driver='GeoJSON')

        logger.info(f"Created sample geological data in {output_path}")
        return {
            'geochemical_csv': str(geochemical_path),
            'features_geojson': str(features_path)
        }


# Example usage
if __name__ == "__main__":
    # Create sample geological data for testing
    sample_files = GeologicalProcessor().create_sample_geological_data()

    # Process geological variables
    processor = GeologicalProcessor()
    results = processor.process_geological_variables(
        sample_files['geochemical_csv'],
        sample_files['features_geojson']
    )

    print("Processed geological variables:")
    print(f"  Geochemical samples: {len(results['geochemical_data'])}")
    print(f"  Geological features: {len(results['geological_features'])}")
    print(f"  Geochemical ratios: {results['geochemical_ratios'].shape[1]}")
    print(f"  Distance metrics: {results['distances_to_features'].shape[1]}")
    print(f"  Geological raster: {results['geological_raster']}")
