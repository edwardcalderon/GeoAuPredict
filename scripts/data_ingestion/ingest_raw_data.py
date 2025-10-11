#!/usr/bin/env python3
"""
GeoAuPredict Data Lake Ingestion Script

This script manages the geospatial data lake by:
1. Downloading and organizing raw data from various sources
2. Standardizing data formats for AI compatibility
3. Creating validation and quality checks
4. Maintaining data lineage and metadata

Usage:
    python scripts/data_ingestion/ingest_raw_data.py --config config.yaml
"""

import argparse
import logging
import os
import sys
import yaml
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Optional
import requests
import zipfile
import shutil
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ingest.data_ingest import GoldDataIngester


class DataLakeManager:
    """Manages the geospatial data lake structure and ingestion."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the data lake manager.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.data_dir = Path(self.config.get('data_dir', 'data'))
        self.setup_logging()

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"⚠️  Configuration file not found: {self.config_path}")
            print("Using default configuration...")
            return self._get_default_config()

    def _get_default_config(self) -> dict:
        """Get default configuration if config file not found."""
        return {
            'data_dir': 'data',
            'sources': {
                'usgs': {'enabled': True, 'url': 'https://mrdata.usgs.gov/mineral-resources/mrds-csv.zip'},
                'sgc': {'enabled': True, 'csv_path': 'data/raw/sgc/geoquimica_sgc.csv'},
                'srtm': {'enabled': True, 'dem_path': 'data/external/dem/srtm_colombia.tif'}
            }
        }

    def setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_config.get('file', 'data/processed/data_lake.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def create_data_lake_structure(self):
        """Create the complete data lake folder structure."""
        self.logger.info("🏗️  Creating data lake structure...")

        # Create main directories
        directories = [
            "data/raw/sgc",
            "data/raw/usgs",
            "data/raw/sentinel",
            "data/raw/srtm",
            "data/raw/perforaciones",
            "data/raw/magnetic",
            "data/raw/gravity",
            "data/raw/lithology",
            "data/processed/datasets",
            "data/processed/interim",
            "data/processed/cache",
            "data/external/dem",
            "data/external/satellite",
            "data/external/geological",
            "scripts/data_ingestion",
            "docs/data_lineage"
        ]

        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"✅ Created directory: {dir_path}")

        # Create metadata files
        self._create_metadata_files()

    def _create_metadata_files(self):
        """Create metadata and documentation files."""
        metadata = {
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "description": "GeoAuPredict Geospatial Data Lake",
            "structure": {
                "raw": {
                    "sgc": "Servicio Geológico Colombiano geochemical data",
                    "usgs": "USGS Mineral Resource Data System",
                    "sentinel": "Sentinel-2 satellite imagery",
                    "srtm": "SRTM Digital Elevation Model",
                    "perforaciones": "Drilling/perforation data",
                    "magnetic": "Magnetic anomaly data",
                    "gravity": "Gravity anomaly data",
                    "lithology": "Lithological/geological maps"
                },
                "processed": {
                    "datasets": "Final processed datasets for ML",
                    "interim": "Intermediate processing results",
                    "cache": "Cached data and temporary files"
                },
                "external": {
                    "dem": "Digital Elevation Models",
                    "satellite": "Satellite imagery data",
                    "geological": "Geological maps and data"
                }
            }
        }

        # Save metadata
        with open("data/data_lake_metadata.json", 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False, indent=2)

        self.logger.info("📋 Created data lake metadata file")

    def ingest_sgc_data(self):
        """Ingest SGC geochemical data."""
        self.logger.info("🔬 Ingesting SGC geochemical data...")

        sgc_config = self.config['sources']['sgc']
        if not sgc_config.get('enabled', False):
            self.logger.info("⏭️  SGC data ingestion disabled")
            return

        source_path = sgc_config['csv_path']
        target_dir = "data/raw/sgc"

        if not os.path.exists(source_path):
            self.logger.error(f"❌ SGC data file not found: {source_path}")
            print(f"📥 Please download SGC geochemical data from: https://www.sgc.gov.co/")
            print(f"   And place it at: {source_path}")
            return

        # Copy file to organized location with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"geoquimica_sgc_{timestamp}.csv"
        target_path = Path(target_dir) / filename

        shutil.copy2(source_path, target_path)
        self.logger.info(f"✅ Copied SGC data to: {target_path}")

        # Create data quality report
        self._validate_sgc_data(target_path)

    def _validate_sgc_data(self, file_path: Path):
        """Validate SGC data quality and create report."""
        try:
            df = pd.read_csv(file_path, sep=';')

            # Basic validation checks
            validation_report = {
                "file_path": str(file_path),
                "total_records": len(df),
                "columns": list(df.columns),
                "missing_data": df.isnull().sum().to_dict(),
                "data_types": df.dtypes.to_dict(),
                "validation_date": datetime.now().isoformat()
            }

            # Save validation report
            report_path = file_path.parent / f"validation_report_{file_path.stem}.json"
            with open(report_path, 'w') as f:
                yaml.dump(validation_report, f, default_flow_style=False, indent=2)

            self.logger.info(f"📊 Created validation report: {report_path}")

        except Exception as e:
            self.logger.error(f"❌ Validation failed: {e}")

    def ingest_usgs_data(self):
        """Download and ingest USGS MRDS data."""
        self.logger.info("🌍 Ingesting USGS MRDS data...")

        usgs_config = self.config['sources']['usgs']
        if not usgs_config.get('enabled', False):
            self.logger.info("⏭️  USGS data ingestion disabled")
            return

        url = usgs_config['url']
        target_dir = "data/raw/usgs"

        try:
            # Download ZIP file
            self.logger.info(f"📥 Downloading from: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            zip_path = Path(target_dir) / "mrds_data.zip"
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Extract ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)

            # Find CSV file and organize it
            extracted_files = list(Path(target_dir).glob("*.csv"))
            if extracted_files:
                csv_file = extracted_files[0]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_name = f"usgs_mrds_{timestamp}.csv"
                organized_path = Path(target_dir) / new_name
                csv_file.rename(organized_path)

                self.logger.info(f"✅ Organized USGS data: {organized_path}")
            else:
                self.logger.warning("⚠️  No CSV file found in USGS download")

            # Clean up ZIP file
            zip_path.unlink()

        except Exception as e:
            self.logger.error(f"❌ USGS data ingestion failed: {e}")

    def create_data_catalog(self):
        """Create a catalog of all available datasets."""
        self.logger.info("📚 Creating data catalog...")

        catalog = {
            "generated_at": datetime.now().isoformat(),
            "data_lake_version": "1.0.0",
            "datasets": {}
        }

        # Scan raw data directories
        for source_dir in Path("data/raw").iterdir():
            if source_dir.is_dir():
                datasets = list(source_dir.glob("*.csv")) + list(source_dir.glob("*.geojson")) + list(source_dir.glob("*.tif"))
                if datasets:
                    catalog["datasets"][source_dir.name] = {
                        "path": str(source_dir),
                        "files": [str(f.name) for f in datasets],
                        "file_count": len(datasets),
                        "last_updated": datetime.now().isoformat()
                    }

        # Save catalog
        catalog_path = "data/data_catalog.json"
        with open(catalog_path, 'w') as f:
            yaml.dump(catalog, f, default_flow_style=False, indent=2)

        self.logger.info(f"📋 Created data catalog: {catalog_path}")

    def run_full_ingestion(self):
        """Run complete data lake ingestion process."""
        self.logger.info("🚀 Starting complete data lake ingestion...")

        # Create structure
        self.create_data_lake_structure()

        # Ingest data sources
        self.ingest_sgc_data()
        self.ingest_usgs_data()

        # Create catalog
        self.create_data_catalog()

        self.logger.info("🎉 Data lake ingestion completed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="GeoAuPredict Data Lake Ingestion")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("--create-structure", action="store_true", help="Only create folder structure")
    parser.add_argument("--ingest-sgc", action="store_true", help="Only ingest SGC data")
    parser.add_argument("--ingest-usgs", action="store_true", help="Only ingest USGS data")
    parser.add_argument("--validate", action="store_true", help="Validate existing data")

    args = parser.parse_args()

    manager = DataLakeManager(args.config)

    if args.create_structure:
        manager.create_data_lake_structure()
    elif args.ingest_sgc:
        manager.ingest_sgc_data()
    elif args.ingest_usgs:
        manager.ingest_usgs_data()
    elif args.validate:
        manager.create_data_catalog()
    else:
        manager.run_full_ingestion()


if __name__ == "__main__":
    main()
