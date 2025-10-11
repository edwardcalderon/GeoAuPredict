#!/usr/bin/env python3
"""
GeoAuPredict Phase 2: Geospatial Feature Engineering Pipeline

This script orchestrates the complete Phase 2 pipeline for geospatial feature engineering:

1. Spectral Indices Calculation (NDVI, NBR, Clay Index, Iron Index)
2. Terrain Variables Extraction (elevation, slope, curvature, aspect, TWI)
3. Geological Variables Processing (geochemistry, distances to features)
4. Feature Integration into unified tabular dataset

Usage:
    python phase2_feature_engineering.py --config config.yaml
    python phase2_feature_engineering.py --demo  # Run with sample data
"""

import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.spectral_indices import SpectralIndicesCalculator, create_sample_spectral_data
from src.data.terrain_analysis import TerrainAnalyzer, create_sample_dem
from src.data.geological_processing import GeologicalProcessor
from src.data.feature_integration import FeatureIntegrator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase2_feature_engineering.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Phase2Pipeline:
    """Main pipeline for Phase 2 geospatial feature engineering."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Phase 2 pipeline.

        Args:
            config: Configuration dictionary with pipeline parameters
        """
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'phase2_output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize processors
        self.spectral_calculator = SpectralIndicesCalculator(
            satellite=config.get('satellite_type', 'sentinel2')
        )
        self.terrain_analyzer = TerrainAnalyzer(
            use_richdem=config.get('use_richdem', True)
        )
        self.geological_processor = GeologicalProcessor()
        self.feature_integrator = FeatureIntegrator(
            pixel_size=config.get('pixel_size', 100.0)
        )

        logger.info("Phase 2 pipeline initialized")

    def run_spectral_indices_calculation(self) -> Dict[str, str]:
        """
        Run spectral indices calculation phase.

        Returns:
            Dictionary of calculated spectral index file paths
        """
        logger.info("🚀 Starting spectral indices calculation...")

        spectral_config = self.config.get('spectral_indices', {})
        band_paths = spectral_config.get('band_paths', {})

        if not band_paths:
            logger.warning("No band paths provided, skipping spectral indices calculation")
            return {}

        try:
            # Calculate all indices
            indices = self.spectral_calculator.calculate_all_indices(band_paths)

            # Save indices
            output_subdir = self.output_dir / "spectral_indices"
            saved_files = self.spectral_calculator.save_indices(
                indices, str(output_subdir)
            )

            logger.info(f"✅ Spectral indices calculated: {list(saved_files.keys())}")
            return saved_files

        except Exception as e:
            logger.error(f"❌ Spectral indices calculation failed: {e}")
            raise

    def run_terrain_analysis(self) -> Dict[str, str]:
        """
        Run terrain analysis phase.

        Returns:
            Dictionary of calculated terrain variable file paths
        """
        logger.info("🏔️  Starting terrain analysis...")

        terrain_config = self.config.get('terrain_analysis', {})
        dem_path = terrain_config.get('dem_path')

        if not dem_path:
            logger.warning("No DEM path provided, skipping terrain analysis")
            return {}

        try:
            # Calculate terrain variables
            terrain_vars, profile = self.terrain_analyzer.calculate_all_terrain_variables(dem_path)

            # Save terrain variables
            output_subdir = self.output_dir / "terrain_variables"
            saved_files = self.terrain_analyzer.save_terrain_variables(
                terrain_vars, str(output_subdir), profile
            )

            logger.info(f"✅ Terrain variables calculated: {list(saved_files.keys())}")
            return saved_files

        except Exception as e:
            logger.error(f"❌ Terrain analysis failed: {e}")
            raise

    def run_geological_processing(self) -> Dict[str, Any]:
        """
        Run geological processing phase.

        Returns:
            Dictionary of processed geological data
        """
        logger.info("🗺️  Starting geological processing...")

        geological_config = self.config.get('geological_processing', {})
        geochemical_path = geological_config.get('geochemical_path')
        features_path = geological_config.get('features_path')

        if not geochemical_path or not features_path:
            logger.warning("Missing geological data paths, skipping geological processing")
            return {}

        try:
            # Process geological variables
            results = self.geological_processor.process_geological_variables(
                geochemical_path, features_path, str(self.output_dir / "geological_variables")
            )

            logger.info("✅ Geological variables processed")
            return results

        except Exception as e:
            logger.error(f"❌ Geological processing failed: {e}")
            raise

    def run_feature_integration(self, spectral_files: Dict[str, str],
                              terrain_files: Dict[str, str],
                              geological_results: Dict[str, Any]) -> str:
        """
        Run feature integration phase.

        Args:
            spectral_files: Dictionary of spectral index files
            terrain_files: Dictionary of terrain variable files
            geological_results: Dictionary of geological processing results

        Returns:
            Path to integrated dataset file
        """
        logger.info("🔗 Starting feature integration...")

        integration_config = self.config.get('feature_integration', {})
        geological_data_path = geological_results.get('combined_geological_data')
        bounds = integration_config.get('bounds')

        if not geological_data_path:
            logger.warning("No geological data for integration")
            return ""

        try:
            # Create comprehensive dataset
            dataset = self.feature_integrator.create_comprehensive_dataset(
                spectral_indices=spectral_files,
                terrain_variables=terrain_files,
                geological_data_path=geological_data_path,
                bounds=bounds,
                use_grid=integration_config.get('use_grid', True)
            )

            if dataset.empty:
                logger.warning("No data integrated")
                return ""

            # Save integrated dataset
            output_format = integration_config.get('output_format', 'csv')
            dataset_path = self.output_dir / f"integrated_geospatial_dataset.{output_format}"

            saved_path = self.feature_integrator.save_dataset(
                dataset, str(dataset_path), output_format
            )

            logger.info(f"✅ Feature integration completed: {saved_path}")
            return saved_path

        except Exception as e:
            logger.error(f"❌ Feature integration failed: {e}")
            raise

    def run_demo(self) -> str:
        """
        Run the complete pipeline with sample data.

        Returns:
            Path to final integrated dataset
        """
        logger.info("🎯 Running Phase 2 pipeline with sample data...")

        # Create sample data for all components
        from src.data.feature_integration import FeatureIntegrator
        sample_integrator = FeatureIntegrator()
        sample_data = sample_integrator.create_sample_integration_data(
            str(self.output_dir / "sample_data")
        )

        # Create sample geological data
        geo_sample_files = self.geological_processor.create_sample_geological_data(
            str(self.output_dir / "sample_data")
        )

        # Update sample data with geological files
        sample_data.update(geo_sample_files)

        # Run each phase with sample data
        spectral_files = {
            'ndvi': str(self.output_dir / "sample_data" / "ndvi.tif"),
            'nbr': str(self.output_dir / "sample_data" / "nbr.tif"),
            'clay_index': str(self.output_dir / "sample_data" / "clay_index.tif"),
            'iron_index': str(self.output_dir / "sample_data" / "iron_index.tif")
        }

        terrain_files = {
            'elevation': str(self.output_dir / "sample_data" / "elevation.tif"),
            'slope_degrees': str(self.output_dir / "sample_data" / "slope_degrees.tif"),
            'plan_curvature': str(self.output_dir / "sample_data" / "plan_curvature.tif"),
            'twi': str(self.output_dir / "sample_data" / "twi.tif")
        }

        # Run geological processing with sample data
        geological_results = self.geological_processor.process_geological_variables(
            geo_sample_files['geochemical_csv'],
            geo_sample_files['features_geojson'],
            str(self.output_dir / "geological_variables")
        )

        # Integrate all features
        bounds = (-75.0, 4.0, -67.0, 12.0)  # Colombia approximate bounds
        dataset_path = self.run_feature_integration(
            spectral_files, terrain_files, geological_results
        )

        logger.info(f"✅ Demo completed successfully: {dataset_path}")
        return dataset_path

    def run_pipeline(self) -> str:
        """
        Run the complete Phase 2 pipeline.

        Returns:
            Path to final integrated dataset
        """
        logger.info("🚀 Starting complete Phase 2 geospatial feature engineering pipeline...")

        # Run each phase
        spectral_files = self.run_spectral_indices_calculation()
        terrain_files = self.run_terrain_analysis()
        geological_results = self.run_geological_processing()

        # Integrate all features
        dataset_path = self.run_feature_integration(
            spectral_files, terrain_files, geological_results
        )

        logger.info("🎉 Phase 2 pipeline completed successfully!")
        return dataset_path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise


def create_default_config(output_dir: str = "phase2_output") -> Dict[str, Any]:
    """
    Create default configuration for Phase 2 pipeline.

    Args:
        output_dir: Output directory path

    Returns:
        Default configuration dictionary
    """
    return {
        'output_dir': output_dir,
        'satellite_type': 'sentinel2',
        'pixel_size': 100.0,
        'use_richdem': True,
        'spectral_indices': {
            'band_paths': {
                'blue': 'data/satellite/B02.tif',
                'green': 'data/satellite/B03.tif',
                'red': 'data/satellite/B04.tif',
                'nir': 'data/satellite/B08.tif',
                'swir1': 'data/satellite/B11.tif',
                'swir2': 'data/satellite/B12.tif'
            }
        },
        'terrain_analysis': {
            'dem_path': 'data/external/dem/colombia_dem.tif'
        },
        'geological_processing': {
            'geochemical_path': 'data/raw/geological/geochemical_samples.csv',
            'features_path': 'data/raw/geological/geological_features.shp'
        },
        'feature_integration': {
            'bounds': [-79.0, -4.3, -66.8, 12.5],  # Colombia bounds
            'use_grid': True,
            'output_format': 'csv'
        }
    }


def main():
    """Main function to run the Phase 2 pipeline."""
    parser = argparse.ArgumentParser(description="GeoAuPredict Phase 2: Geospatial Feature Engineering")
    parser.add_argument('--config', '-c', type=str, help='Path to configuration YAML file')
    parser.add_argument('--demo', '-d', action='store_true', help='Run with sample data')
    parser.add_argument('--output', '-o', type=str, default='phase2_output', help='Output directory')

    args = parser.parse_args()

    try:
        if args.demo:
            # Run demo with sample data
            logger.info("🎯 Running Phase 2 pipeline in demo mode...")

            config = create_default_config(args.output)
            pipeline = Phase2Pipeline(config)
            dataset_path = pipeline.run_demo()

        else:
            # Run with provided configuration
            if args.config:
                config = load_config(args.config)
            else:
                logger.info("No config provided, using default configuration")
                config = create_default_config(args.output)

            pipeline = Phase2Pipeline(config)
            dataset_path = pipeline.run_pipeline()

        if dataset_path:
            logger.info(f"✅ Pipeline completed successfully!")
            logger.info(f"📊 Final dataset: {dataset_path}")

            # Show dataset summary
            import pandas as pd
            df = pd.read_csv(dataset_path)
            logger.info(f"📈 Dataset summary: {len(df)} samples, {len(df.columns)} features")

        else:
            logger.error("❌ Pipeline failed to produce output dataset")
            sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
