#!/usr/bin/env python3
"""
Data Standardization and Validation Utilities for GeoAuPredict

This module provides utilities for:
1. Standardizing geospatial data formats
2. Validating data quality and consistency
3. Converting between different coordinate systems
4. Ensuring AI compatibility

Usage:
    from scripts.data_ingestion.data_utils import standardize_coordinates, validate_dataset
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from shapely.geometry import Point
import re


class DataStandardizer:
    """Standardizes geospatial data formats for AI compatibility."""

    def __init__(self):
        """Initialize the data standardizer."""
        self.logger = logging.getLogger(__name__)

        # Standard column names for AI compatibility
        self.standard_columns = {
            'coordinates': ['latitude', 'longitude', 'lat', 'lon', 'x', 'y'],
            'elements': ['au_ppm', 'ag_ppm', 'cu_ppm', 'pb_ppm', 'zn_ppm', 'fe_ppm', 'as_ppm', 'sb_ppm'],
            'spectral': ['ndvi', 'clay_index', 'iron_index', 'ndwi', 'savi'],
            'geophysical': ['magnetic_anomaly', 'gravity_anomaly', 'mag_rtp', 'bouguer'],
            'geological': ['lithology', 'formation', 'rock_type', 'fault_distance', 'fold_distance'],
            'topography': ['elevation', 'slope', 'aspect', 'dem', 'dtm'],
            'metadata': ['sample_id', 'source', 'date', 'project', 'method', 'quality']
        }

        # Colombia bounds for spatial filtering
        self.colombia_bounds = {
            'west': -79.0,
            'south': -4.3,
            'east': -66.8,
            'north': 12.5
        }

    def standardize_coordinates(self, df: pd.DataFrame, lat_col: str = None, lon_col: str = None) -> pd.DataFrame:
        """Standardize coordinate columns to lat/lon format.

        Args:
            df: Input DataFrame
            lat_col: Latitude column name (auto-detected if None)
            lon_col: Longitude column name (auto-detected if None)

        Returns:
            DataFrame with standardized coordinates
        """
        df = df.copy()

        # Auto-detect coordinate columns
        if lat_col is None or lon_col is None:
            lat_col, lon_col = self._detect_coordinate_columns(df)

        if lat_col and lon_col:
            # Rename to standard names
            df = df.rename(columns={lat_col: 'latitude', lon_col: 'longitude'})

            # Validate coordinate ranges
            df = self._validate_coordinates(df)

            self.logger.info(f"✅ Standardized coordinates: {lat_col}→latitude, {lon_col}→longitude")
        else:
            self.logger.warning("⚠️  Could not detect coordinate columns")

        return df

    def _detect_coordinate_columns(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """Auto-detect latitude and longitude columns."""
        # Common patterns for coordinate columns
        lat_patterns = [r'lat', r'latitude', r'y_coord', r'northing']
        lon_patterns = [r'lon', r'longitude', r'x_coord', r'easting']

        lat_col = None
        lon_col = None

        # Find latitude column
        for pattern in lat_patterns:
            matches = [col for col in df.columns if re.search(pattern, col, re.IGNORECASE)]
            if matches:
                lat_col = matches[0]
                break

        # Find longitude column
        for pattern in lon_patterns:
            matches = [col for col in df.columns if re.search(pattern, col, re.IGNORECASE)]
            if matches:
                lon_col = matches[0]
                break

        return lat_col, lon_col

    def _validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean coordinate data."""
        # Remove invalid coordinates
        valid_mask = (
            (df['latitude'].between(-90, 90)) &
            (df['longitude'].between(-180, 180)) &
            (df['latitude'].notna()) &
            (df['longitude'].notna())
        )

        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            self.logger.warning(f"⚠️  Removed {invalid_count} invalid coordinates")

        df = df[valid_mask].copy()

        # Filter to Colombia bounds if requested
        if self.colombia_bounds:
            colombia_mask = (
                (df['longitude'] >= self.colombia_bounds['west']) &
                (df['longitude'] <= self.colombia_bounds['east']) &
                (df['latitude'] >= self.colombia_bounds['south']) &
                (df['latitude'] <= self.colombia_bounds['north'])
            )

            outside_count = (~colombia_mask).sum()
            if outside_count > 0:
                self.logger.info(f"📍 Filtered {outside_count} points outside Colombia bounds")

            df = df[colombia_mask].copy()

        return df

    def standardize_elements(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize element concentration columns."""
        df = df.copy()

        # Element column mappings (common variations)
        element_mappings = {
            'au_ppm': ['AU_PPM', 'Au_ppm', 'AU', 'au', 'gold_ppm', 'oro_ppm'],
            'ag_ppm': ['AG_PPM', 'Ag_ppm', 'AG', 'ag', 'silver_ppm', 'plata_ppm'],
            'cu_ppm': ['CU_PPM', 'Cu_ppm', 'CU', 'cu', 'copper_ppm', 'cobre_ppm'],
            'pb_ppm': ['PB_PPM', 'Pb_ppm', 'PB', 'pb', 'lead_ppm', 'plomo_ppm'],
            'zn_ppm': ['ZN_PPM', 'Zn_ppm', 'ZN', 'zn', 'zinc_ppm'],
            'fe_ppm': ['FE_PPM', 'Fe_ppm', 'FE', 'fe', 'iron_ppm', 'hierro_ppm'],
            'as_ppm': ['AS_PPM', 'As_ppm', 'AS', 'as', 'arsenic_ppm', 'arsenico_ppm'],
            'sb_ppm': ['SB_PPM', 'Sb_ppm', 'SB', 'sb', 'antimony_ppm', 'antimonio_ppm']
        }

        # Apply mappings
        for standard_name, variations in element_mappings.items():
            for variation in variations:
                if variation in df.columns:
                    df = df.rename(columns={variation: standard_name})
                    self.logger.info(f"✅ Mapped {variation} → {standard_name}")
                    break

        # Convert to numeric and handle detection limits
        for col in df.columns:
            if col.endswith('_ppm') and df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')

                # Handle detection limits (<0.1, etc.)
                if df[col].dtype == 'object':
                    df[col] = df[col].str.replace('<', '').str.replace('>', '').astype(float)

        return df

    def create_gold_labels(self, df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
        """Create binary gold presence labels for machine learning.

        Args:
            df: Input DataFrame with gold concentration data
            threshold: Gold concentration threshold in ppm

        Returns:
            DataFrame with added gold_label column
        """
        df = df.copy()

        # Find gold column
        gold_cols = [col for col in df.columns if 'au_ppm' in col.lower() or 'gold' in col.lower()]
        gold_col = gold_cols[0] if gold_cols else None

        if not gold_col:
            self.logger.error("❌ No gold concentration column found")
            return df

        # Create binary labels
        df['gold_label'] = (df[gold_col] >= threshold).astype(int)

        # Add label metadata
        df['gold_threshold'] = threshold
        df['label_method'] = 'concentration_threshold'

        self.logger.info(f"🏷️  Created gold labels using {gold_col} >= {threshold} ppm")
        self.logger.info(f"   Positive samples: {df['gold_label'].sum()}/{len(df)} ({df['gold_label'].mean()*100:.1f}%)")

        return df

    def validate_dataset(self, df: pd.DataFrame) -> Dict:
        """Comprehensive dataset validation.

        Args:
            df: Input DataFrame to validate

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'total_records': len(df),
            'columns': len(df.columns),
            'missing_data': {},
            'data_types': {},
            'coordinate_issues': 0,
            'duplicate_coordinates': 0,
            'outliers': {},
            'recommendations': []
        }

        # Check missing data
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                validation_results['missing_data'][col] = missing_count

        # Check data types
        validation_results['data_types'] = df.dtypes.astype(str).to_dict()

        # Check for duplicate coordinates
        if 'latitude' in df.columns and 'longitude' in df.columns:
            coords = df[['latitude', 'longitude']].round(6)
            duplicates = coords.duplicated().sum()
            validation_results['duplicate_coordinates'] = int(duplicates)

        # Check coordinate validity
        if 'latitude' in df.columns and 'longitude' in df.columns:
            valid_coords = (
                (df['latitude'].between(-90, 90)) &
                (df['longitude'].between(-180, 180))
            )
            validation_results['coordinate_issues'] = (~valid_coords).sum()

        # Generate recommendations
        if validation_results['missing_data']:
            validation_results['recommendations'].append(
                "Consider handling missing data through imputation or removal"
            )

        if validation_results['duplicate_coordinates'] > 0:
            validation_results['recommendations'].append(
                f"Remove or consolidate {validation_results['duplicate_coordinates']} duplicate coordinates"
            )

        if validation_results['coordinate_issues'] > 0:
            validation_results['recommendations'].append(
                f"Fix {validation_results['coordinate_issues']} invalid coordinates"
            )

        return validation_results

    def convert_to_geodataframe(self, df: pd.DataFrame, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
        """Convert DataFrame to GeoDataFrame.

        Args:
            df: Input DataFrame with latitude/longitude columns
            crs: Coordinate reference system

        Returns:
            GeoDataFrame with geometry column
        """
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            self.logger.error("❌ Latitude/longitude columns required for GeoDataFrame conversion")
            return None

        # Create geometry column
        geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)

        self.logger.info(f"🌍 Created GeoDataFrame with {len(gdf)} features")

        return gdf

    def save_standardized_data(self, df: pd.DataFrame, output_path: Path,
                              formats: List[str] = ['csv', 'geojson']) -> List[str]:
        """Save standardized data in multiple formats.

        Args:
            df: Standardized DataFrame or GeoDataFrame
            output_path: Base output path (without extension)
            formats: List of output formats

        Returns:
            List of saved file paths
        """
        saved_files = []

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        for fmt in formats:
            file_path = output_path.with_suffix(f'.{fmt}')

            try:
                if fmt.lower() == 'csv':
                    df.to_csv(file_path, index=False)
                elif fmt.lower() == 'geojson':
                    if isinstance(df, gpd.GeoDataFrame):
                        df.to_file(file_path, driver='GeoJSON')
                    else:
                        # Convert to GeoDataFrame if needed
                        gdf = self.convert_to_geodataframe(df)
                        if gdf is not None:
                            gdf.to_file(file_path, driver='GeoJSON')
                elif fmt.lower() == 'parquet':
                    df.to_parquet(file_path, index=False)

                saved_files.append(str(file_path))
                self.logger.info(f"💾 Saved {fmt.upper()}: {file_path}")

            except Exception as e:
                self.logger.error(f"❌ Failed to save {fmt}: {e}")

        return saved_files


def standardize_coordinates(df: pd.DataFrame, lat_col: str = None, lon_col: str = None) -> pd.DataFrame:
    """Convenience function for coordinate standardization."""
    standardizer = DataStandardizer()
    return standardizer.standardize_coordinates(df, lat_col, lon_col)


def validate_dataset(df: pd.DataFrame) -> Dict:
    """Convenience function for dataset validation."""
    standardizer = DataStandardizer()
    return standardizer.validate_dataset(df)


def create_gold_labels(df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    """Convenience function for creating gold labels."""
    standardizer = DataStandardizer()
    return standardizer.create_gold_labels(df, threshold)
