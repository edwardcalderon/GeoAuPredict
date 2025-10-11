"""
Spectral Indices Calculation Module for Phase 2 Geospatial Feature Engineering

This module calculates various spectral indices from satellite imagery including:
- NDVI (Normalized Difference Vegetation Index) - vegetation health
- NBR (Normalized Burn Ratio) - burned areas detection
- Clay Index - clay mineral detection
- Iron Index - iron oxide detection

Supports both Sentinel-2 and ASTER data formats.
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpectralIndicesCalculator:
    """Calculator for various spectral indices used in mineral exploration."""

    def __init__(self, satellite: str = "sentinel2"):
        """
        Initialize the spectral indices calculator.

        Args:
            satellite: Type of satellite data ('sentinel2' or 'aster')
        """
        self.satellite = satellite.lower()
        self.band_mapping = self._get_band_mapping()

    def _get_band_mapping(self) -> Dict[str, Union[str, int]]:
        """Get band mappings for different satellite sensors."""
        if self.satellite == "sentinel2":
            return {
                'blue': 'B02', 'green': 'B03', 'red': 'B04', 'nir': 'B08',
                'swir1': 'B11', 'swir2': 'B12'
            }
        elif self.satellite == "aster":
            return {
                'green': 1, 'red': 2, 'nir': 3, 'swir1': 4, 'swir2': 5,
                'swir3': 6, 'swir4': 7, 'swir5': 8, 'swir6': 9
            }
        else:
            raise ValueError(f"Unsupported satellite type: {self.satellite}")

    def load_bands(self, band_paths: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Load satellite bands from file paths.

        Args:
            band_paths: Dictionary mapping band names to file paths

        Returns:
            Dictionary of loaded band arrays
        """
        bands = {}
        for band_name, file_path in band_paths.items():
            if not Path(file_path).exists():
                logger.warning(f"Band file not found: {file_path}")
                continue

            try:
                with rasterio.open(file_path) as src:
                    bands[band_name] = src.read(1).astype(np.float32)
                    logger.info(f"Loaded {band_name} band: {bands[band_name].shape}")
            except Exception as e:
                logger.error(f"Error loading {band_name} band: {e}")
                continue

        return bands

    def calculate_ndvi(self, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Difference Vegetation Index (NDVI).

        NDVI = (NIR - Red) / (NIR + Red)

        Args:
            red: Red band array
            nir: Near-infrared band array

        Returns:
            NDVI array
        """
        logger.info("Calculating NDVI...")
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = (nir - red) / (nir + red)
            ndvi = np.clip(ndvi, -1, 1)  # Valid range is -1 to 1

        logger.info(f"NDVI range: {ndvi.min():.3f} to {ndvi.max():.3f}")
        return ndvi

    def calculate_nbr(self, nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Burn Ratio (NBR).

        NBR = (NIR - SWIR2) / (NIR + SWIR2)

        Args:
            nir: Near-infrared band array
            swir2: Short-wave infrared band 2 array

        Returns:
            NBR array
        """
        logger.info("Calculating NBR...")
        with np.errstate(divide='ignore', invalid='ignore'):
            nbr = (nir - swir2) / (nir + swir2)
            nbr = np.clip(nbr, -1, 1)  # Valid range is -1 to 1

        logger.info(f"NBR range: {nbr.min():.3f} to {nbr.max():.3f}")
        return nbr

    def calculate_clay_index(self, swir1: np.ndarray, swir2: np.ndarray) -> np.ndarray:
        """
        Calculate Clay Index for clay mineral detection.

        Clay Index = (SWIR1 / SWIR2)

        Args:
            swir1: Short-wave infrared band 1 array
            swir2: Short-wave infrared band 2 array

        Returns:
            Clay Index array
        """
        logger.info("Calculating Clay Index...")
        with np.errstate(divide='ignore', invalid='ignore'):
            clay_index = swir1 / swir2

        logger.info(f"Clay Index range: {clay_index.min():.3f} to {clay_index.max():.3f}")
        return clay_index

    def calculate_iron_index(self, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
        """
        Calculate Iron Index for iron oxide detection.

        Iron Index = (Red / Blue)

        Args:
            red: Red band array
            blue: Blue band array

        Returns:
            Iron Index array
        """
        logger.info("Calculating Iron Index...")
        with np.errstate(divide='ignore', invalid='ignore'):
            iron_index = red / blue

        logger.info(f"Iron Index range: {iron_index.min():.3f} to {iron_index.max():.3f}")
        return iron_index

    def calculate_all_indices(self, band_paths: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Calculate all spectral indices from band file paths.

        Args:
            band_paths: Dictionary mapping band names to file paths

        Returns:
            Dictionary of all calculated indices
        """
        logger.info(f"Starting spectral indices calculation for {self.satellite}...")

        # Load required bands
        bands = self.load_bands(band_paths)

        # Verify we have all required bands
        required_bands = ['red', 'nir']
        if self.satellite == "sentinel2":
            required_bands.extend(['swir1', 'swir2'])
        else:  # ASTER
            required_bands.extend(['swir1', 'swir2'])

        missing_bands = [band for band in required_bands if band not in bands]
        if missing_bands:
            raise ValueError(f"Missing required bands: {missing_bands}")

        # Calculate indices
        indices = {}

        # NDVI (always calculated)
        if 'red' in bands and 'nir' in bands:
            indices['ndvi'] = self.calculate_ndvi(bands['red'], bands['nir'])

        # NBR (always calculated)
        if 'nir' in bands and 'swir2' in bands:
            indices['nbr'] = self.calculate_nbr(bands['nir'], bands['swir2'])

        # Clay Index (Sentinel-2 and ASTER)
        if 'swir1' in bands and 'swir2' in bands:
            indices['clay_index'] = self.calculate_clay_index(bands['swir1'], bands['swir2'])

        # Iron Index (Sentinel-2 only)
        if 'red' in bands and 'blue' in bands:
            indices['iron_index'] = self.calculate_iron_index(bands['red'], bands['blue'])

        logger.info(f"Successfully calculated {len(indices)} spectral indices")
        return indices

    def save_indices(self, indices: Dict[str, np.ndarray], output_dir: str,
                    reference_profile: Optional[dict] = None) -> Dict[str, str]:
        """
        Save calculated indices as GeoTIFF files.

        Args:
            indices: Dictionary of index arrays
            output_dir: Output directory path
            reference_profile: Rasterio profile for consistent output

        Returns:
            Dictionary mapping index names to saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        for index_name, index_array in indices.items():
            output_file = output_path / f"{index_name}.tif"

            if reference_profile is None:
                # Create a basic profile if none provided
                reference_profile = {
                    'driver': 'GTiff',
                    'dtype': 'float32',
                    'nodata': -9999,
                    'width': index_array.shape[1],
                    'height': index_array.shape[0],
                    'count': 1,
                    'crs': 'EPSG:4326',
                    'transform': from_bounds(-180, -90, 180, 90, index_array.shape[1], index_array.shape[0])
                }

            # Update profile for this index
            profile = reference_profile.copy()
            profile['count'] = 1

            try:
                with rasterio.open(output_file, 'w', **profile) as dst:
                    dst.write(index_array.astype(np.float32), 1)

                saved_files[index_name] = str(output_file)
                logger.info(f"Saved {index_name} to {output_file}")

            except Exception as e:
                logger.error(f"Error saving {index_name}: {e}")
                continue

        logger.info(f"Saved {len(saved_files)} spectral indices")
        return saved_files


def create_sample_spectral_data(output_dir: str = "sample_data") -> Dict[str, str]:
    """
    Create sample spectral data for testing purposes.

    Args:
        output_dir: Directory to save sample data

    Returns:
        Dictionary of created file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Create sample band data (100x100 pixels)
    np.random.seed(42)  # For reproducible results

    sample_bands = {
        'B02.tif': np.random.uniform(0, 0.3, (100, 100)).astype(np.float32),  # Blue
        'B03.tif': np.random.uniform(0, 0.4, (100, 100)).astype(np.float32),  # Green
        'B04.tif': np.random.uniform(0, 0.5, (100, 100)).astype(np.float32),  # Red
        'B08.tif': np.random.uniform(0, 0.8, (100, 100)).astype(np.float32),  # NIR
        'B11.tif': np.random.uniform(0, 0.6, (100, 100)).astype(np.float32),  # SWIR1
        'B12.tif': np.random.uniform(0, 0.4, (100, 100)).astype(np.float32),  # SWIR2
    }

    file_paths = {}
    for filename, data in sample_bands.items():
        file_path = output_path / filename
        with rasterio.open(
            file_path, 'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype='float32',
            crs='EPSG:4326',
            transform=from_bounds(0, 0, 1, 1, 100, 100)
        ) as dst:
            dst.write(data, 1)
        file_paths[filename] = str(file_path)

    logger.info(f"Created sample spectral data in {output_path}")
    return file_paths


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    sample_files = create_sample_spectral_data()

    # Map to band names for Sentinel-2
    band_paths = {
        'blue': sample_files['B02.tif'],
        'green': sample_files['B03.tif'],
        'red': sample_files['B04.tif'],
        'nir': sample_files['B08.tif'],
        'swir1': sample_files['B11.tif'],
        'swir2': sample_files['B12.tif']
    }

    # Calculate indices
    calculator = SpectralIndicesCalculator(satellite="sentinel2")
    indices = calculator.calculate_all_indices(band_paths)

    print("Calculated indices:")
    for name, data in indices.items():
        print(f"  {name}: {data.shape}, range: [{data.min():.3f}, {data.max():.3f}]")

    # Save indices
    saved_files = calculator.save_indices(indices, "output_indices")
    print(f"Saved {len(saved_files)} indices to output_indices/")
