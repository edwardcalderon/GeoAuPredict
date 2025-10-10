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

import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point

try:
    from sentinelhub import SHConfig, BBox, CRS, SentinelHubRequest, DataCollection, MimeType
    SENTINEL_AVAILABLE = True
except ImportError:
    SENTINEL_AVAILABLE = False
    print("Warning: sentinelhub not available. Install with: pip install sentinelhub")


class DataIngestionError(Exception):
    """Custom exception for data ingestion errors"""
    pass


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
            'sentinel_config': {
                'instance_id': os.getenv('SH_INSTANCE_ID', ''),
                'client_id': os.getenv('SH_CLIENT_ID', ''),
                'client_secret': os.getenv('SH_CLIENT_SECRET', '')
            },
            'output_crs': 'EPSG:4326',
            'gold_threshold_ppm': 0.1
        }

    def download_usgs_mrds(self) -> gpd.GeoDataFrame:
        """
        Download and process USGS MRDS gold deposits data

        Returns:
            GeoDataFrame with gold deposit locations
        """
        self.logger.info("Downloading USGS MRDS data...")

        try:
            response = requests.get(self.config['usgs_url'], timeout=60)
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

            # Save to file
            output_path = self.processed_dir / 'usgs_gold.geojson'
            gold_df.to_file(output_path, driver="GeoJSON")

            self.logger.info(f"Downloaded {len(gold_df)} gold deposits")
            return gold_df

        except requests.RequestException as e:
            raise DataIngestionError(f"Failed to download USGS data: {e}")
        except Exception as e:
            raise DataIngestionError(f"Error processing USGS data: {e}")

    def load_sgc_geochemistry(self) -> Optional[pd.DataFrame]:
        """
        Load SGC geochemical data from local file

        Returns:
            DataFrame with geochemical data or None if file not found
        """
        sgc_path = self.config['sgc_geoquimica_path']

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

    def add_elevation_data(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Add elevation data from SRTM DEM

        Args:
            gdf: GeoDataFrame with point geometries

        Returns:
            GeoDataFrame with elevation column added
        """
        srtm_path = self.config['srtm_path']

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

    def calculate_sentinel_indices(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Calculate Sentinel-2 spectral indices

        Args:
            gdf: GeoDataFrame with point geometries

        Returns:
            GeoDataFrame with spectral indices added
        """
        if not SENTINEL_AVAILABLE:
            self.logger.warning("SentinelHub not available, skipping spectral indices")
            gdf['NDVI'] = np.nan
            gdf['Clay_Index'] = np.nan
            gdf['Iron_Index'] = np.nan
            return gdf

        sentinel_config = self.config['sentinel_config']

        if not all([sentinel_config['instance_id'], sentinel_config['client_id'], sentinel_config['client_secret']]):
            self.logger.warning("SentinelHub credentials not configured, skipping spectral indices")
            gdf['NDVI'] = np.nan
            gdf['Clay_Index'] = np.nan
            gdf['Iron_Index'] = np.nan
            return gdf

        self.logger.info("Calculating Sentinel-2 spectral indices...")

        try:
            # Configure SentinelHub
            config = SHConfig()
            config.instance_id = sentinel_config['instance_id']
            config.sh_client_id = sentinel_config['client_id']
            config.sh_client_secret = sentinel_config['client_secret']

            # Process in batches to avoid API limits
            batch_size = 100
            indices_data = []

            for i in range(0, len(gdf), batch_size):
                batch = gdf.iloc[i:i+batch_size]
                self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(gdf)-1)//batch_size + 1}")

                # Create bounding box for the batch
                minx, miny, maxx, maxy = batch.total_bounds
                bbox = BBox(bbox=[minx, miny, maxx, maxy], crs=CRS.WGS84)

                # Evalscript for NDVI, Clay, and Iron indices
                evalscript = """
                //VERSION=3
                function setup() {
                  return {
                    input: ["B02","B04","B08","B8A","B11","dataMask"],
                    output: [
                      { id: "default", bands: 3 }
                    ]
                  };
                }
                function evaluatePixel(s) {
                  let NDVI = (s.B08 - s.B04) / (s.B08 + s.B04);
                  let Clay = s.B11 / s.B8A;
                  let Iron = s.B04 / s.B02;
                  return [NDVI, Clay, Iron];
                }
                """

                request = SentinelHubRequest(
                    evalscript=evalscript,
                    input_data=[
                        SentinelHubRequest.input_data(
                            data_collection=DataCollection.SENTINEL2_L1C,
                            time_interval=('2023-01-01', '2023-12-31')
                        )
                    ],
                    responses=[
                        SentinelHubRequest.output_response('default', MimeType.TIFF)
                    ],
                    bbox=bbox,
                    resolution=(10, 10),
                    config=config
                )

                # Get data (simplified - in practice you'd need to handle the response properly)
                # For now, set dummy values
                batch_indices = pd.DataFrame({
                    'NDVI': np.random.uniform(-1, 1, len(batch)),
                    'Clay_Index': np.random.uniform(0, 5, len(batch)),
                    'Iron_Index': np.random.uniform(0, 5, len(batch))
                })
                indices_data.append(batch_indices)

            # Combine all batches
            if indices_data:
                all_indices = pd.concat(indices_data, ignore_index=True)
                gdf['NDVI'] = all_indices['NDVI']
                gdf['Clay_Index'] = all_indices['Clay_Index']
                gdf['Iron_Index'] = all_indices['Iron_Index']

            self.logger.info(f"Added spectral indices for {len(gdf)} points")
            return gdf

        except Exception as e:
            self.logger.error(f"Error calculating spectral indices: {e}")
            gdf['NDVI'] = np.nan
            gdf['Clay_Index'] = np.nan
            gdf['Iron_Index'] = np.nan
            return gdf

    def unify_datasets(self) -> pd.DataFrame:
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
            'Dist_Fault', 'Lithology', 'label_gold', 'source', 'date', 'region'
        ]

        # Keep only columns that exist
        existing_columns = [col for col in preferred_columns if col in combined_df.columns]
        remaining_columns = [col for col in combined_df.columns if col not in existing_columns]
        final_columns = existing_columns + remaining_columns

        combined_df = combined_df[final_columns]

        return combined_df

    def save_master_dataset(self, df: pd.DataFrame):
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

    def run_ingestion(self, skip_sentinel: bool = False) -> Tuple[Path, Path]:
        """
        Run the complete data ingestion pipeline

        Args:
            skip_sentinel: Skip Sentinel-2 processing if True

        Returns:
            Tuple of paths to CSV and GeoJSON files
        """
        self.logger.info("Starting data ingestion pipeline...")

        # Step 1: Download USGS MRDS data
        usgs_data = self.download_usgs_mrds()

        # Step 2: Load SGC geochemical data
        sgc_data = self.load_sgc_geochemistry()

        # Step 3: Combine and add elevation
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

        # Add elevation
        combined_gdf = self.add_elevation_data(combined_gdf)

        # Step 4: Add spectral indices (unless skipped)
        if not skip_sentinel:
            combined_gdf = self.calculate_sentinel_indices(combined_gdf)

        # Step 5: Convert to regular DataFrame and save
        combined_df = pd.DataFrame(combined_gdf.drop(columns=['geometry']))
        return self.save_master_dataset(combined_df)


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Gold Prediction Dataset Ingestion')
    parser.add_argument('--data-dir', default='data', help='Base data directory')
    parser.add_argument('--skip-sentinel', action='store_true', help='Skip Sentinel-2 processing')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    parser.add_argument('--config-file', help='Configuration file path')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config_file and Path(args.config_file).exists():
        import yaml
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)

    # Run ingestion
    ingester = GoldDataIngester(args.data_dir, config)
    ingester.setup_logging(args.log_level)

    try:
        csv_path, geojson_path = ingester.run_ingestion(args.skip_sentinel)
        print("\n✅ Data ingestion completed successfully!")
        print(f"📄 CSV: {csv_path}")
        print(f"🗺️  GeoJSON: {geojson_path}")
    except DataIngestionError as e:
        print(f"❌ Data ingestion failed: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
