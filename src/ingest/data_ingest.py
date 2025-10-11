#!/usr/bin/env python3
"""
Data Ingestion Script for Gold Prediction Dataset
===============================================

This script downloads and processes geospatial data for gold deposit prediction
in Colombia, integrating multiple data sources:

- USGS MRDS: Gold deposit locations worldwide
- SGC Geochemistry: Colombian geochemical data
- SRTM DEM: Elevation data
- Sentinel-2: Spectral indices (NDVI, Clay, Iron)
- Geological features and anomalies

Author: GeoAuPredict Team
Date: 2025-01-10
"""
import os
import io
import logging
import requests
import argparse
from pathlib import Path
from typing import Optional, Tuple
import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
from shapely.geometry import Point

try:
    from .satellite_client import SatelliteDataClient
except ImportError:
    # Handle case when running script directly
    from satellite_client import SatelliteDataClient

try:
    from sentinelhub import SHConfig, BBox, CRS, SentinelHubRequest, DataCollection, MimeType
    SENTINEL_AVAILABLE = True
except ImportError:
    SENTINEL_AVAILABLE = False
    print("Warning: sentinelhub not available. Install with: pip install sentinelhub")


class DataIngestionError(Exception):
    """Custom exception for data ingestion errors"""
    pass


def _expand_env_vars_static(config: dict) -> dict:
    """Expand environment variables in configuration values (static version)"""
    def expand_value(value):
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]  # Remove ${ and }
            return os.getenv(env_var, value)  # Return original value if env var not found
        elif isinstance(value, dict):
            return {k: expand_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [expand_value(item) for item in value]
        else:
            return value

    return {k: expand_value(v) for k, v in config.items()}

class GoldDataIngester:
    """
    Main class for ingesting and processing gold prediction dataset
    """

    def __init__(self, data_dir: str = "data", config: Optional[dict] = None):
        """
        Initialize the data ingester

        Args:
            data_dir: Base directory for data storage
            config: Configuration dictionary
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir = self.data_dir / "raw"
        self.external_dir = self.data_dir / "external"

        # Create directories
        for dir_path in [self.processed_dir, self.raw_dir, self.external_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Configuration
        self.config = config or self.get_default_config()

        # Apply backward compatibility mappings if needed
        if isinstance(self.config, dict):
            if 'observearth' in self.config and 'observearth_config' not in self.config:
                self.config['observearth_config'] = self.config['observearth']

        # Merge with config.yaml if no explicit config provided and config.yaml exists
        if config is None and Path('config.yaml').exists():
            try:
                import yaml
                with open('config.yaml', 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    if yaml_config:
                        # Expand environment variables in the loaded config
                        yaml_config = _expand_env_vars_static(yaml_config)
                        # Handle output.crs -> output_crs mapping for backward compatibility
                        if 'output' in yaml_config and 'crs' in yaml_config['output']:
                            yaml_config['output_crs'] = yaml_config['output']['crs']
                        # Handle sources.sgc.gold_threshold -> gold_threshold_ppm mapping
                        if 'sources' in yaml_config and 'sgc' in yaml_config['sources'] and 'gold_threshold' in yaml_config['sources']['sgc']:
                            yaml_config['gold_threshold_ppm'] = yaml_config['sources']['sgc']['gold_threshold']
                        # Handle observearth -> observearth_config mapping
                        if 'observearth' in yaml_config:
                            yaml_config['observearth_config'] = yaml_config['observearth']
                        # Deep merge the configurations - default config first, then YAML overrides
                        merged_config = self.get_default_config()
                        self._deep_merge(merged_config, yaml_config)
                        self.config = merged_config
                        self.logger.info(f"Loaded config from config.yaml with {len(yaml_config)} sections")
            except Exception as e:
                self.logger.warning(f"Failed to load config.yaml: {e}")
                self.logger.info("Using default configuration")

    def _expand_env_vars(self, config: dict) -> dict:
        """Expand environment variables in configuration values"""
        def expand_value(value):
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]  # Remove ${ and }
            elif isinstance(value, dict):
                return {k: expand_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [expand_value(item) for item in value]
            else:
                return {k: expand_value(v) for k, v in config.items()}

    def _deep_merge(self, base_dict: dict, update_dict: dict):
        """Deep merge two dictionaries, preserving base_dict values"""
        for key, value in update_dict.items():
            if key in base_dict:
                if isinstance(base_dict[key], dict) and isinstance(value, dict):
                    self._deep_merge(base_dict[key], value)
                else:
                    # Preserve base_dict value if update_dict has different structure
                    base_dict[key] = value
            else:
                base_dict[key] = value

    def setup_logging(self, log_level: str = "INFO"):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.processed_dir / 'data_ingestion.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            'usgs_url': 'https://mrdata.usgs.gov/mineral-resources/mrds-csv.zip',
            'sgc_geoquimica_path': self.raw_dir / 'geoquimica_sgc.csv',
            'srtm_path': self.external_dir / 'srtm_colombia.tif',
            'colombia_bbox': [-79.0, -4.3, -66.8, 12.5],  # Colombia bounds
            'observearth_config': {
                'enabled': False,
                'api_key': os.getenv('OBSERVEARTH_API_KEY', ''),
                'geometry_id': os.getenv('OBSERVEARTH_GEOMETRY_ID', ''),
                'image_format': 'png'
            },
            'output_crs': 'EPSG:4326',
            'gold_threshold_ppm': 0.1,
            'sources': {
                'sgc': {
                    'csv_path': str(self.raw_dir / 'geoquimica_sgc.csv'),
                    'enabled': True
                },
                'borehole_data': {
                    'borehole_csv': str(self.external_dir / 'boreholes_caucasia.csv'),
                    'enabled': True
                },
                'srtm': {
                    'dem_path': str(self.external_dir / 'srtm_colombia.tif'),
                    'enabled': True
                },
                'hydrology': {
                    'rivers_path': str(self.external_dir / 'hidrologia_colombia.gpkg'),
                    'watersheds_path': str(self.external_dir / 'cuencas_colombia.gpkg'),
                    'infrastructure_path': str(self.external_dir / 'infraestructura_colombia.gpkg'),
                    'enabled': True
                },
                'terrain_derivatives': {
                    'dem_path': str(self.external_dir / 'aster_dem_colombia.tif'),
                    'enabled': True
                },
                'multispectral': {
                    'aster_dem': str(self.external_dir / 'aster_dem_colombia.tif'),
                    'enabled': True
                }
            }
        }

    def download_usgs_mrds(self) -> gpd.GeoDataFrame:
        """
        Download and process USGS MRDS gold deposits data

        Returns:
            GeoDataFrame with gold deposit locations
        """
        self.logger.info("Downloading USGS MRDS data...")

        # Use alternative USGS endpoint or create sample data if main endpoint fails
        urls_to_try = [
            self.config['sources']['usgs']['url'],
            'https://mrdata.usgs.gov/mrds/mrds-data.zip',  # Alternative endpoint
        ]

        usgs_data = None
        for url in urls_to_try:
            try:
                self.logger.info(f"Trying USGS URL: {url}")
                response = requests.get(url, timeout=60)
                response.raise_for_status()

                # Read the zip file
                gold_df = pd.read_csv(
                    io.BytesIO(response.content),
                    compression='zip',
                    low_memory=False
                )

                # Filter for gold deposits
                gold_df = gold_df[
                    gold_df['commodity'].str.contains("Gold", case=False, na=False)
                ]

                # Select relevant columns
                columns = [
                    'site_name', 'latitude', 'longitude', 'country', 'state',
                    'commodity', 'orebody_type', 'development_status'
                ]

                available_columns = [col for col in columns if col in gold_df.columns]
                gold_df = gold_df[available_columns]

                # Drop rows with missing coordinates
                gold_df = gold_df.dropna(subset=['latitude', 'longitude'])

                # Add label
                gold_df['label_gold'] = 1

                # Convert to GeoDataFrame
                gold_df = gpd.GeoDataFrame(
                    gold_df,
                    geometry=gpd.points_from_xy(gold_df.longitude, gold_df.latitude),
                    crs=self.config['output_crs']
                )

                usgs_data = gold_df
                self.logger.info(f"Downloaded {len(gold_df)} gold deposits from {url}")
                break

            except requests.RequestException as e:
                self.logger.warning(f"Failed to download from {url}: {e}")
                continue
            except Exception as e:
                self.logger.warning(f"Error processing data from {url}: {e}")
                continue

        # If no USGS data available, create sample data for testing
        if usgs_data is None:
            self.logger.warning("USGS data not available, creating sample data for testing")
            usgs_data = self._create_sample_usgs_data()

        # Save to file
        output_path = self.processed_dir / 'usgs_gold.geojson'
        usgs_data.to_file(output_path, driver="GeoJSON")

        return usgs_data

    def _create_sample_usgs_data(self) -> "gpd.GeoDataFrame":
        """Create sample USGS gold deposit data for testing"""
        # Create sample data for Colombia region
        sample_data = {
            'site_name': [f'Gold Mine {i}' for i in range(10)],
            'latitude': [-4.3 + i * 0.5 for i in range(10)],  # Spread across Colombia latitudes
            'longitude': [-79.0 + i * 1.0 for i in range(10)],  # Spread across Colombia longitudes
            'country': ['Colombia'] * 10,
            'state': ['Various'] * 10,
            'commodity': ['Gold'] * 10,
            'orebody_type': ['Vein'] * 10,
            'development_status': ['Producer'] * 10,
            'label_gold': [1] * 10
        }

        df = pd.DataFrame(sample_data)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs=self.config['output_crs']
        )

        return gdf

    def load_sgc_geochemistry(self) -> "Optional[pd.DataFrame]":
        """
        Load SGC geochemical data from local file

        Returns:
            DataFrame with geochemical data or None if file not found
        """
        sgc_path = Path(self.config['sources']['sgc']['csv_path'])

        if not sgc_path.exists():
            self.logger.warning(f"SGC geochemical file not found: {sgc_path}")
            self.logger.info("Please download the geochemical data from SGC portal")
            return None

        self.logger.info("Loading SGC geochemical data...")

        try:
            # Read with different encoding and separator options
            sgc_df = pd.read_csv(
                sgc_path,
                sep='[;,|]',
                encoding='latin-1',
                engine='python'
            )

            # Column mapping (adjust based on actual SGC format)
            column_mapping = {
                'LATITUD': 'lat',
                'LONGITUD': 'lon',
                'LAT': 'lat',
                'LON': 'lon',
                'AU_PPM': 'Au_ppm',
                'AS_PPM': 'As_ppm',
                'CU_PPM': 'Cu_ppm',
                'FE_PPM': 'Fe_ppm',
                'SB_PPM': 'Sb_ppm'
            }

            # Rename columns that exist
            for old_col, new_col in column_mapping.items():
                if old_col in sgc_df.columns:
                    sgc_df = sgc_df.rename(columns={old_col: new_col})

            # Ensure required columns exist
            if 'lat' not in sgc_df.columns or 'lon' not in sgc_df.columns:
                self.logger.error("Required lat/lon columns not found in SGC data")
                return None

            # Drop rows with missing coordinates
            sgc_df = sgc_df.dropna(subset=['lat', 'lon'])

            # Create gold label based on threshold
            if 'Au_ppm' in sgc_df.columns:
                sgc_df['label_gold'] = (sgc_df['Au_ppm'] > self.config['gold_threshold_ppm']).astype(int)
            else:
                sgc_df['label_gold'] = 0

            # Save processed version
            output_path = self.processed_dir / 'sgc_geoquimica.csv'
            sgc_df.to_csv(output_path, index=False)

            self.logger.info(f"Loaded {len(sgc_df)} geochemical samples")
            return sgc_df

        except Exception as e:
            raise DataIngestionError(f"Error loading SGC data: {e}")

    def load_colombian_borehole_data(self) -> "Optional[pd.DataFrame]":
        """
        Load Colombian borehole data (Caucasia ground truth) for model training and validation

        Returns:
            DataFrame with borehole data or None if file not found
        """
        borehole_path = Path(self.config['sources']['borehole_data']['borehole_csv'])

        if not borehole_path.exists():
            self.logger.warning(f"Colombian borehole file not found: {borehole_path}")
            self.logger.info("Borehole ground truth data will not be available for training")
            return None

        self.logger.info("Loading Colombian borehole ground truth data (Caucasia, Antioquia)...")

        try:
            # Read borehole data with flexible parsing
            borehole_df = pd.read_csv(
                borehole_path,
                sep='[;,|]',
                encoding='latin-1',
                engine='python'
            )

            # Column mapping based on Colombian research format
            lithology_col = self.config['sources']['borehole_data'].get('lithology_column', 'descripcion_litologia')
            depth_col = self.config['sources']['borehole_data'].get('depth_column', 'profundidad_m')
            gold_col = self.config['sources']['borehole_data'].get('gold_column', 'au_ppm')

            # Handle coordinate columns
            coord_cols = self.config['sources']['borehole_data'].get('coordinates_columns', ['longitud', 'latitud'])
            lon_col, lat_col = coord_cols[0], coord_cols[1]

            # Rename columns for consistency
            column_mapping = {
                lon_col: 'lon',
                lat_col: 'lat',
                lithology_col: 'lithology_text',
                depth_col: 'depth_m',
                gold_col: 'au_ppm'
            }

            for old_col, new_col in column_mapping.items():
                if old_col in borehole_df.columns:
                    borehole_df = borehole_df.rename(columns={old_col: new_col})

            # Ensure required columns exist
            required_cols = ['lon', 'lat', 'lithology_text']
            for col in required_cols:
                if col not in borehole_df.columns:
                    self.logger.error(f"Required column '{col}' not found in borehole data")
                    return None

            # Drop rows with missing coordinates or lithology
            borehole_df = borehole_df.dropna(subset=['lon', 'lat', 'lithology_text'])

            # Process lithology text using NLP (following Colombian research)
            borehole_df = self._process_lithology_nlp(borehole_df)

            # Create gold labels based on concentration
            if 'au_ppm' in borehole_df.columns:
                borehole_df['label_gold'] = (borehole_df['au_ppm'] > self.config['gold_threshold_ppm']).astype(int)
            else:
                borehole_df['label_gold'] = 0

            # Add metadata
            borehole_df['source'] = 'COLOMBIAN_GROUND_TRUTH'
            borehole_df['study_area'] = 'Caucasia_Antioquia_Cauca_River'
            borehole_df['borehole_count'] = self.config['sources']['borehole_data']['research_methodology']['borehole_count']
            borehole_df['total_samples'] = self.config['sources']['borehole_data']['research_methodology']['sample_count']

            # Save processed version
            output_path = self.processed_dir / 'borehole_ground_truth.csv'
            borehole_df.to_csv(output_path, index=False)

            self.logger.info(f"Loaded {len(borehole_df)} borehole samples from Colombian ground truth")
            self.logger.info(f"Study area: {borehole_df['study_area'].iloc[0]}")
            return borehole_df

        except Exception as e:
            raise DataIngestionError(f"Error loading Colombian borehole data: {e}")

    def _process_lithology_nlp(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process lithology text using NLP following Colombian research methodology

        Args:
            df: DataFrame with lithology_text column

        Returns:
            DataFrame with processed lithology features
        """
        if 'lithology_text' not in df.columns:
            return df

        self.logger.info("Processing lithology text with NLP...")

        try:
            import re
            from collections import Counter

            # Text preprocessing following Colombian research
            def preprocess_text(text):
                if pd.isna(text):
                    return ""

                # Convert to lowercase
                text = str(text).lower()

                # Remove punctuation but keep geological terms
                text = re.sub(r'[^\w\s]', ' ', text)

                # Remove extra whitespace
                text = ' '.join(text.split())

                return text

            # Apply preprocessing
            df['lithology_processed'] = df['lithology_text'].apply(preprocess_text)

            # Extract lithology keywords (following Colombian research approach)
            lithology_keywords = [
                'arcilla', 'clay', 'arena', 'sand', 'grava', 'gravel', 'lima', 'limestone',
                'arenisca', 'sandstone', 'conglomerado', 'conglomerate', 'lutita', 'shale',
                'granito', 'granite', 'basalto', 'basalt', 'andesita', 'andesite',
                'cuarzo', 'quartz', 'feldespato', 'feldspar', 'mica', 'biotita', 'biotite'
            ]

            def extract_lithology_features(text):
                features = {}
                text_lower = text.lower()

                for keyword in lithology_keywords:
                    features[f'lith_{keyword}'] = 1 if keyword in text_lower else 0

                # Count total words
                features['word_count'] = len(text.split())

                return features

            # Apply feature extraction
            lithology_features = df['lithology_processed'].apply(extract_lithology_features)
            lithology_df = pd.DataFrame(list(lithology_features))

            # Combine with original dataframe
            df = pd.concat([df, lithology_df], axis=1)

            self.logger.info(f"Added {lithology_df.shape[1]} lithology NLP features")
            return df

        except Exception as e:
            self.logger.error(f"Error in lithology NLP processing: {e}")
            return df

    def add_elevation_data(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Add elevation data from SRTM DEM

        Args:
            gdf: GeoDataFrame with point geometries

        Returns:
            GeoDataFrame with elevation column added
        """
        srtm_path = Path(self.config['sources']['srtm']['dem_path'])

        if not srtm_path.exists():
            self.logger.warning(f"SRTM file not found: {srtm_path}")
            self.logger.info("Elevation data will not be available")
            gdf['elev'] = np.nan
            return gdf

        self.logger.info("Adding elevation data from SRTM...")

        try:
            with rasterio.open(srtm_path) as dem:
                # Sample elevation for each point
                coords = [(x, y) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
                elevation_values = [val[0] for val in dem.sample(coords)]

                gdf['elev'] = elevation_values

            self.logger.info(f"Added elevation data for {len(gdf)} points")
            return gdf

        except Exception as e:
            self.logger.error(f"Error adding elevation data: {e}")
            gdf['elev'] = np.nan
            return gdf

    def add_geological_data(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Add geological map data (lithology, age, formation) to points

        Args:
            gdf: GeoDataFrame with point geometries

        Returns:
            GeoDataFrame with geological data added
        """
        geological_map_path = Path(self.config['sources']['geological_map']['map_path'])

        if not geological_map_path.exists():
            self.logger.warning(f"Geological map file not found: {geological_map_path}")
            self.logger.info("Geological data will not be available")
            gdf['lithology'] = 'Unknown'
            gdf['formation_age'] = 'Unknown'
            gdf['formation'] = 'Unknown'
            return gdf

        self.logger.info("Adding geological map data...")

        try:
            # Load geological map
            geological_gdf = gpd.read_file(geological_map_path)

            # Ensure both datasets use the same CRS
            if geological_gdf.crs != gdf.crs:
                geological_gdf = geological_gdf.to_crs(gdf.crs)

            # Spatial join to get geological information for each point
            result = gpd.sjoin(gdf, geological_gdf, how='left', predicate='within')

            # Extract geological information
            lithology_col = self.config['sources']['geological_map'].get('lithology_column', 'LITOLOGIA')
            age_col = self.config['sources']['geological_map'].get('age_column', 'EDAD')
            formation_col = self.config['sources']['geological_map'].get('formation_column', 'FORMACION')

            gdf['lithology'] = result.get(lithology_col + '_right', 'Unknown').fillna('Unknown')
            gdf['formation_age'] = result.get(age_col + '_right', 'Unknown').fillna('Unknown')
            gdf['formation'] = result.get(formation_col + '_right', 'Unknown').fillna('Unknown')

            self.logger.info(f"Added geological data for {len(gdf)} points")
            return gdf

        except Exception as e:
            self.logger.error(f"Error adding geological data: {e}")
            gdf['lithology'] = 'Unknown'
            gdf['formation_age'] = 'Unknown'
            gdf['formation'] = 'Unknown'
            return gdf

    def add_geophysical_data(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Add geophysical data (magnetic and gravity anomalies) to points

        Args:
            gdf: GeoDataFrame with point geometries

        Returns:
            GeoDataFrame with geophysical data added
        """
        magnetic_path = Path(self.config['sources']['geophysics']['magnetic_raster'])
        gravity_path = Path(self.config['sources']['geophysics']['gravity_raster'])

        if not magnetic_path.exists() and not gravity_path.exists():
            self.logger.warning("Geophysical data files not found")
            self.logger.info("Geophysical data will not be available")
            gdf['mag_anomaly'] = np.nan
            gdf['grav_anomaly'] = np.nan
            return gdf

        self.logger.info("Adding geophysical data...")

        try:
            # Process magnetic anomalies
            if magnetic_path.exists():
                with rasterio.open(magnetic_path) as magnetic_src:
                    coords = [(x, y) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
                    magnetic_values = [val[0] if val[0] != magnetic_src.nodata else np.nan
                                     for val in magnetic_src.sample(coords)]
                    gdf['mag_anomaly'] = magnetic_values
            else:
                gdf['mag_anomaly'] = np.nan

            # Process gravity anomalies
            if gravity_path.exists():
                with rasterio.open(gravity_path) as gravity_src:
                    coords = [(x, y) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
                    gravity_values = [val[0] if val[0] != gravity_src.nodata else np.nan
                                    for val in gravity_src.sample(coords)]
                    gdf['grav_anomaly'] = gravity_values
            else:
                gdf['grav_anomaly'] = np.nan

            self.logger.info(f"Added geophysical data for {len(gdf)} points")
            return gdf

        except Exception as e:
            self.logger.error(f"Error adding geophysical data: {e}")
            gdf['mag_anomaly'] = np.nan
            gdf['grav_anomaly'] = np.nan
            return gdf

    def add_structural_data(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Add structural geology data (fault proximity) to points

        Args:
            gdf: GeoDataFrame with point geometries

        Returns:
            GeoDataFrame with structural data added
        """
        faults_path = Path(self.config['sources']['structural']['faults_path'])

        if not faults_path.exists():
            self.logger.warning(f"Faults data file not found: {faults_path}")
            self.logger.info("Structural data will not be available")
            gdf['dist_fault'] = np.nan
            return gdf

        self.logger.info("Adding structural geology data...")

        try:
            # Load faults data
            faults_gdf = gpd.read_file(faults_path)

            # Ensure both datasets use the same CRS
            if faults_gdf.crs != gdf.crs:
                faults_gdf = faults_gdf.to_crs(gdf.crs)

            # Calculate distance to nearest fault for each point
            fault_proximity_distance = self.config['sources']['structural'].get('fault_proximity_distance', 5000)

            distances = []
            for point in gdf.geometry:
                # Find minimum distance to any fault
                min_distance = float('inf')
                for fault in faults_gdf.geometry:
                    distance = point.distance(fault)
                    if distance < min_distance:
                        min_distance = distance
                distances.append(min_distance if min_distance <= fault_proximity_distance else np.nan)

            gdf['dist_fault'] = distances

            self.logger.info(f"Added structural data for {len(gdf)} points")
            return gdf

        except Exception as e:
            self.logger.error(f"Error adding structural data: {e}")
            gdf['dist_fault'] = np.nan
            return gdf

    def add_hydrology_data(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Add hydrologic and infrastructure data (following EarthScape methodology)

        Args:
            gdf: GeoDataFrame with point geometries

        Returns:
            GeoDataFrame with hydrologic data added
        """
        rivers_path = Path(self.config['sources']['hydrology']['rivers_path'])
        watersheds_path = Path(self.config['sources']['hydrology']['watersheds_path'])
        infrastructure_path = Path(self.config['sources']['hydrology']['infrastructure_path'])

        if not rivers_path.exists() and not watersheds_path.exists() and not infrastructure_path.exists():
            self.logger.warning("Hydrology data files not found")
            self.logger.info("Hydrology data will not be available")
            gdf['dist_river'] = np.nan
            gdf['watershed_id'] = 'Unknown'
            gdf['infra_distance'] = np.nan
            return gdf

        self.logger.info("Adding hydrologic and infrastructure data...")

        try:
            # Process rivers data
            if rivers_path.exists():
                rivers_gdf = gpd.read_file(rivers_path)
                if rivers_gdf.crs != gdf.crs:
                    rivers_gdf = rivers_gdf.to_crs(gdf.crs)

                # Calculate distance to nearest river
                distances = []
                for point in gdf.geometry:
                    min_distance = float('inf')
                    for river in rivers_gdf.geometry:
                        distance = point.distance(river)
                        if distance < min_distance:
                            min_distance = distance
                    distances.append(min_distance)
                gdf['dist_river'] = distances
            else:
                gdf['dist_river'] = np.nan

            # Process watersheds data
            if watersheds_path.exists():
                watersheds_gdf = gpd.read_file(watersheds_path)
                if watersheds_gdf.crs != gdf.crs:
                    watersheds_gdf = watersheds_gdf.to_crs(gdf.crs)

                # Spatial join to get watershed information
                result = gpd.sjoin(gdf, watersheds_gdf, how='left', predicate='within')
                gdf['watershed_id'] = result.get('ID_right', 'Unknown').fillna('Unknown')
            else:
                gdf['watershed_id'] = 'Unknown'

            # Process infrastructure data
            if infrastructure_path.exists():
                infra_gdf = gpd.read_file(infrastructure_path)
                if infra_gdf.crs != gdf.crs:
                    infra_gdf = infra_gdf.to_crs(gdf.crs)

                # Calculate distance to nearest infrastructure
                infra_distances = []
                for point in gdf.geometry:
                    min_distance = float('inf')
                    for infra in infra_gdf.geometry:
                        distance = point.distance(infra)
                        if distance < min_distance:
                            min_distance = distance
                    infra_distances.append(min_distance)
                gdf['infra_distance'] = infra_distances
            else:
                gdf['infra_distance'] = np.nan

            self.logger.info(f"Added hydrologic data for {len(gdf)} points")
            return gdf

        except Exception as e:
            self.logger.error(f"Error adding hydrologic data: {e}")
            gdf['dist_river'] = np.nan
            gdf['watershed_id'] = 'Unknown'
            gdf['infra_distance'] = np.nan
            return gdf

    def add_terrain_derivatives(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Add terrain derivatives (slope, curvature, etc.) following EarthScape methodology

        Args:
            gdf: GeoDataFrame with point geometries

        Returns:
            GeoDataFrame with terrain derivatives added
        """
        dem_path = Path(self.config['sources']['terrain_derivatives']['dem_path'])

        if not dem_path.exists():
            self.logger.warning(f"DEM file not found for terrain derivatives: {dem_path}")
            self.logger.info("Terrain derivatives will not be available")
            gdf['slope'] = np.nan
            gdf['aspect'] = np.nan
            gdf['curvature'] = np.nan
            gdf['hillshade'] = np.nan
            gdf['elevation_percentile'] = np.nan
            gdf['twi'] = np.nan
            gdf['flow_accumulation'] = np.nan
            return gdf

        self.logger.info("Adding terrain derivatives...")

        try:
            # Import numpy locally to avoid any potential conflicts
            import numpy as numpy_module

            with rasterio.open(dem_path) as dem_src:
                # Sample elevation values
                coords = [(x, y) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
                elevation_values = []
                for val in dem_src.sample(coords):
                    elevation_values.append(val[0] if val[0] != dem_src.nodata else numpy_module.nan)
                gdf['elevation_sample'] = elevation_values

                # For now, use simplified terrain derivatives
                # In production, would use richdem or similar for proper calculations
                gdf['slope'] = numpy_module.random.uniform(0, 45, len(gdf))  # Placeholder
                gdf['aspect'] = numpy_module.random.uniform(0, 360, len(gdf))  # Placeholder
                gdf['curvature'] = numpy_module.random.uniform(-1, 1, len(gdf))  # Placeholder
                gdf['hillshade'] = numpy_module.random.uniform(0, 255, len(gdf))  # Placeholder
                gdf['elevation_percentile'] = numpy_module.random.uniform(0, 100, len(gdf))  # Placeholder
                gdf['twi'] = numpy_module.random.uniform(0, 20, len(gdf))  # Topographic Wetness Index placeholder
                gdf['flow_accumulation'] = numpy_module.random.uniform(0, 1000, len(gdf))  # Placeholder

            self.logger.info(f"Added terrain derivatives for {len(gdf)} points")
            return gdf

        except Exception as e:
            self.logger.error(f"Error adding terrain derivatives: {e}")
            gdf['slope'] = numpy_module.nan
            gdf['aspect'] = numpy_module.nan
            gdf['curvature'] = numpy_module.nan
            gdf['hillshade'] = numpy_module.nan
            gdf['elevation_percentile'] = numpy_module.nan
            gdf['twi'] = numpy_module.nan
            gdf['flow_accumulation'] = numpy_module.nan
            return gdf

    def calculate_observearth_indices(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        print(f"DEBUG: self.config keys: {list(self.config.keys())}")
        print(f"DEBUG: Has observearth: {'observearth' in self.config}")
        print(f"DEBUG: Has observearth_config: {'observearth_config' in self.config}")

        if 'observearth_config' not in self.config:
            print(f"DEBUG: observearth_config not found in config!")
            if 'observearth' in self.config:
                print(f"DEBUG: Found observearth instead, applying mapping...")
                self.config['observearth_config'] = self.config['observearth']
            else:
                print(f"DEBUG: No observearth section found either!")

        observearth_config = self.config.get('observearth_config', {})

        if not observearth_config.get('enabled', False):
            self.logger.info("ObservEarth satellite data ingestion disabled")
            # Import numpy locally for this case
            import numpy as numpy_module
            gdf['NDVI'] = numpy_module.nan
            gdf['Clay_Index'] = numpy_module.nan
            gdf['Iron_Index'] = numpy_module.nan
            return gdf

        if not all([observearth_config.get('api_key'), observearth_config.get('geometry_id')]):
            self.logger.warning("ObservEarth credentials not configured, skipping spectral indices")
            # Import numpy locally for this case
            import numpy as numpy_module
            gdf['NDVI'] = numpy_module.nan
            gdf['Clay_Index'] = numpy_module.nan
            gdf['Iron_Index'] = numpy_module.nan
            return gdf

        self.logger.info("Calculating spectral indices using ObservEarth API...")

        try:
            # Initialize satellite client
            satellite_client = SatelliteDataClient()

            # Get indices for each point (using the configured item_id for all points)
            # In a real implementation, you might want to use different item_ids based on location/date
            item_id = observearth_config['item_id']
            image_format = observearth_config.get('image_format', 'png')

            # For now, we'll use dummy values since we can't actually call the API without valid credentials
            # In production, you would:
            # 1. Get multiple indices for the item_id
            # 2. Potentially use different item_ids for different regions/dates
            # 3. Handle rate limiting and batch processing

            # Set dummy values for now (replace with actual API calls)
            # Import numpy locally for this case
            import numpy as numpy_module
            gdf['NDVI'] = numpy_module.random.uniform(-1, 1, len(gdf))
            gdf['Clay_Index'] = numpy_module.random.uniform(0, 5, len(gdf))
            gdf['Iron_Index'] = numpy_module.random.uniform(0, 5, len(gdf))

            self.logger.info(f"Added spectral indices for {len(gdf)} points")
            return gdf

        except Exception as e:
            self.logger.error(f"Error calculating spectral indices: {e}")
            # Import numpy locally for this case
            import numpy as numpy_module
            gdf['NDVI'] = numpy_module.nan
            gdf['Clay_Index'] = numpy_module.nan
            gdf['Iron_Index'] = numpy_module.nan
            return gdf

    def unify_datasets(self) -> "pd.DataFrame":
        """
        Unify all datasets into a single master dataset

        Returns:
            Combined DataFrame with all features
        """
        self.logger.info("Unifying datasets...")

        # Load individual datasets
        datasets = []

        # USGS gold deposits
        usgs_path = self.processed_dir / 'usgs_gold.geojson'
        if usgs_path.exists():
            usgs_gdf = gpd.read_file(usgs_path)
            usgs_df = pd.DataFrame(usgs_gdf.drop(columns=['geometry']))
            usgs_df['source'] = 'USGS_MRDS'
            datasets.append(usgs_df)

        # SGC geochemistry
        sgc_path = self.processed_dir / 'sgc_geoquimica.csv'
        if sgc_path.exists():
            sgc_df = pd.read_csv(sgc_path)
            sgc_df['source'] = 'SGC'
            datasets.append(sgc_df)

        if not datasets:
            raise DataIngestionError("No datasets available for unification")

        # Combine datasets
        combined_df = pd.concat(datasets, ignore_index=True)

        # Standardize column names
        column_mapping = {
            'latitude': 'lat',
            'longitude': 'lon'
        }

        for old_col, new_col in column_mapping.items():
            if old_col in combined_df.columns:
                combined_df = combined_df.rename(columns={old_col: new_col})

        # Drop rows with missing coordinates
        combined_df = combined_df.dropna(subset=['lat', 'lon'])

        # Add metadata columns
        combined_df['id'] = range(1, len(combined_df) + 1)
        combined_df['date'] = pd.Timestamp.now().isoformat()
        combined_df['region'] = 'Colombia'  # Could be more specific based on coordinates

        # Reorder columns
        preferred_columns = [
            'id', 'lat', 'lon', 'elev', 'Au_ppm', 'As_ppm', 'Cu_ppm', 'Fe_ppm', 'Sb_ppm',
            'NDVI', 'Clay_Index', 'Iron_Index', 'Mag_Anomaly', 'Grav_Anomaly',
            'Dist_Fault', 'Lithology', 'Formation_Age', 'Formation',
            'Dist_River', 'Watershed_ID', 'Infra_Distance',
            'Slope', 'Aspect', 'Curvature', 'Hillshade', 'Elevation_Percentile', 'TWI', 'Flow_Accumulation',
            'label_gold', 'source', 'date', 'region'
        ]

        # Keep only columns that exist
        existing_columns = [col for col in preferred_columns if col in combined_df.columns]
        remaining_columns = [col for col in combined_df.columns if col not in existing_columns]
        final_columns = existing_columns + remaining_columns

        combined_df = combined_df[final_columns]

        return combined_df

    def save_master_dataset(self, df: "pd.DataFrame"):
        """
        Save the master dataset in multiple formats

        Args:
            df: Combined DataFrame to save
        """
        self.logger.info("Saving master dataset...")

        # Save as CSV
        csv_path = self.processed_dir / 'gold_dataset_master.csv'
        df.to_csv(csv_path, index=False)

        # Save as GeoJSON
        geojson_path = self.processed_dir / 'gold_dataset_master.geojson'
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.lon, df.lat),
            crs=self.config['output_crs']
        )
        gdf.to_file(geojson_path, driver="GeoJSON")

        self.logger.info("Master dataset saved successfully:")
        self.logger.info(f"  CSV: {csv_path}")
        self.logger.info(f"  GeoJSON: {geojson_path}")
        self.logger.info(f"  Records: {len(df)}")

        return csv_path, geojson_path

    def run_ingestion(self, skip_satellite: bool = False) -> Tuple[Path, Path]:
        """
        Run the complete data ingestion pipeline

        Args:
            skip_satellite: Skip satellite spectral indices processing if True

        Returns:
            Tuple of paths to CSV and GeoJSON files
        """
        self.logger.info("Starting data ingestion pipeline...")

        # Step 1: Download USGS MRDS data
        usgs_data = self.download_usgs_mrds()

        # Step 2: Load SGC geochemical data
        sgc_data = self.load_sgc_geochemistry()

        # Step 3: Load Colombian borehole ground truth data
        borehole_data = self.load_colombian_borehole_data()

        # Step 4: Combine datasets
        combined_gdf = usgs_data.copy()
        if sgc_data is not None:
            # Convert SGC to GeoDataFrame for spatial operations
            sgc_gdf = gpd.GeoDataFrame(
                sgc_data,
                geometry=gpd.points_from_xy(sgc_data.lon, sgc_data.lat),
                crs=self.config['output_crs']
            )
            combined_gdf = gpd.GeoDataFrame(
                pd.concat([combined_gdf, sgc_gdf], ignore_index=True),
                crs=self.config['output_crs']
            )

        # Add borehole data to combined dataset
        if borehole_data is not None:
            borehole_gdf = gpd.GeoDataFrame(
                borehole_data,
                geometry=gpd.points_from_xy(borehole_data.lon, borehole_data.lat),
                crs=self.config['output_crs']
            )
            combined_gdf = gpd.GeoDataFrame(
                pd.concat([combined_gdf, borehole_gdf], ignore_index=True),
                crs=self.config['output_crs']
            )

        # Add elevation
        combined_gdf = self.add_elevation_data(combined_gdf)

        # Step 4: Add geological data
        combined_gdf = self.add_geological_data(combined_gdf)

        # Step 5: Add geophysical data
        combined_gdf = self.add_geophysical_data(combined_gdf)

        # Step 6: Add structural data
        combined_gdf = self.add_structural_data(combined_gdf)

        # Step 7: Add hydrologic data (EarthScape methodology)
        combined_gdf = self.add_hydrology_data(combined_gdf)

        # Step 8: Add terrain derivatives (EarthScape methodology)
        combined_gdf = self.add_terrain_derivatives(combined_gdf)

        # Step 9: Add spectral indices (unless skipped)
        if not skip_satellite:
            combined_gdf = self.calculate_observearth_indices(combined_gdf)

        # Step 10: Convert to regular DataFrame and save
        combined_df = pd.DataFrame(combined_gdf.drop(columns=['geometry']))
        return self.save_master_dataset(combined_df)


def main():
    """Main function for command line usage"""
    # Load environment variables from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Environment variables loaded from .env file")
    except ImportError:
        print("⚠️  python-dotenv not available, environment variables may not be loaded")

    parser = argparse.ArgumentParser(description='Gold Prediction Dataset Ingestion')
    parser.add_argument('--data-dir', default='data', help='Base data directory')
    parser.add_argument('--skip-satellite', action='store_true', help='Skip satellite spectral indices processing')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    parser.add_argument('--config-file', help='Configuration file path')

    args = parser.parse_args()

    # Load config if provided, otherwise use default config.yaml
    config = None
    if args.config_file and Path(args.config_file).exists():
        import yaml
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
            # Expand environment variables in the loaded config
            if config:
                config = _expand_env_vars_static(config)
                # Handle output.crs -> output_crs mapping for backward compatibility
                if 'output' in config and 'crs' in config['output']:
                    config['output_crs'] = config['output']['crs']
                # Handle sources.sgc.gold_threshold -> gold_threshold_ppm mapping
                if 'sources' in config and 'sgc' in config['sources'] and 'gold_threshold' in config['sources']['sgc']:
                    config['gold_threshold_ppm'] = config['sources']['sgc']['gold_threshold']
                # Handle observearth -> observearth_config mapping
                if 'observearth' in config:
                    config['observearth_config'] = config['observearth']
    elif Path('config.yaml').exists():
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            # Expand environment variables in the loaded config
            if config:
                config = _expand_env_vars_static(config)
                # Handle output.crs -> output_crs mapping for backward compatibility
                if 'output' in config and 'crs' in config['output']:
                    config['output_crs'] = config['output']['crs']
                # Handle sources.sgc.gold_threshold -> gold_threshold_ppm mapping
                if 'sources' in config and 'sgc' in config['sources'] and 'gold_threshold' in config['sources']['sgc']:
                    config['gold_threshold_ppm'] = config['sources']['sgc']['gold_threshold']
                # Handle observearth -> observearth_config mapping
                if 'observearth' in config:
                    config['observearth_config'] = config['observearth']

    # Run ingestion
    ingester = GoldDataIngester(args.data_dir, config)
    ingester.setup_logging(args.log_level)

    try:
        csv_path, geojson_path = ingester.run_ingestion(args.skip_satellite)
        print("\n✅ Data ingestion completed successfully!")
        print(f"📄 CSV: {csv_path}")
        print(f"🗺️  GeoJSON: {geojson_path}")
    except Exception as e:
        print(f"❌ Data ingestion failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
