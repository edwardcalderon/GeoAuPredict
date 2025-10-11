"""
Data preprocessing utilities for GeoAuPredict
Handles normalization, CRS conversion, feature extraction, and borehole data preprocessing
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing for geospatial gold prediction data.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data preprocessor.

        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = config or {}
        self.feature_scalers = {}

        logger.info("Initialized DataPreprocessor")

    def preprocess_borehole_data(self, borehole_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess borehole data for machine learning.

        Args:
            borehole_data: Raw borehole data DataFrame

        Returns:
            Preprocessed DataFrame with engineered features
        """
        logger.info("Preprocessing borehole data")

        if borehole_data is None or borehole_data.empty:
            raise ValueError("Borehole data is empty or None")

        # Create a copy to avoid modifying original data
        processed_data = borehole_data.copy()

        # Basic data cleaning
        processed_data = self._clean_data(processed_data)

        # Feature engineering
        processed_data = self._engineer_features(processed_data)

        # Handle missing values
        processed_data = self._handle_missing_values(processed_data)

        # Normalize features
        processed_data = self._normalize_features(processed_data)

        logger.info(f"Preprocessing completed. Shape: {processed_data.shape}")

        return processed_data

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data."""
        # Remove duplicates
        data = data.drop_duplicates()

        # Validate required columns
        required_cols = ['lat', 'lon', 'elev', 'Au_ppm', 'label_gold']
        missing_cols = [col for col in required_cols if col not in data.columns]

        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            # Add missing columns with default values
            for col in missing_cols:
                if col in ['lat', 'lon']:
                    data[col] = 0.0
                elif col == 'elev':
                    data[col] = 1000.0  # Default elevation
                elif col == 'Au_ppm':
                    data[col] = 0.01  # Default low concentration
                elif col == 'label_gold':
                    data[col] = 0  # Default negative

        return data

    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer new features from existing data."""

        # Geological features
        if 'Au_ppm' in data.columns:
            data['Au_log'] = np.log1p(data['Au_ppm'])
            data['Au_binary'] = (data['Au_ppm'] > 0.1).astype(int)

        # Terrain features
        if 'elev' in data.columns:
            data['elev_normalized'] = (data['elev'] - data['elev'].mean()) / data['elev'].std()
            data['elev_category'] = pd.cut(data['elev'], bins=[0, 500, 1000, 2000, 5000],
                                        labels=['lowland', 'midland', 'highland', 'mountain'])

        # Spatial features
        if 'lat' in data.columns and 'lon' in data.columns:
            # Create spatial clusters (simplified)
            data['spatial_cluster'] = self._create_spatial_clusters(data[['lat', 'lon']].values)

        # Statistical features
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col not in ['lat', 'lon', 'id']:
                # Add statistical transformations
                data[f'{col}_sqrt'] = np.sqrt(np.abs(data[col]))
                data[f'{col}_square'] = data[col] ** 2

        return data

    def _create_spatial_clusters(self, coordinates: np.ndarray, n_clusters: int = 10) -> np.ndarray:
        """Create spatial clusters using simple grid-based approach."""
        from sklearn.cluster import KMeans

        # Simple grid-based clustering for demo
        lat_bins = np.linspace(coordinates[:, 0].min(), coordinates[:, 0].max(), int(np.sqrt(n_clusters)))
        lon_bins = np.linspace(coordinates[:, 1].min(), coordinates[:, 1].max(), int(np.sqrt(n_clusters)))

        lat_indices = np.digitize(coordinates[:, 0], lat_bins) - 1
        lon_indices = np.digitize(coordinates[:, 1], lon_bins) - 1

        # Create cluster IDs
        cluster_ids = lat_indices * len(lon_bins) + lon_indices

        return cluster_ids

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data."""

        # For numeric columns, fill with median
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].isnull().sum() > 0:
                data[col] = data[col].fillna(data[col].median())

        # For categorical columns, fill with mode or 'unknown'
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if data[col].isnull().sum() > 0:
                if data[col].dtype.name == 'category':
                    data[col] = data[col].cat.add_categories(['unknown'])
                    data[col] = data[col].fillna('unknown')
                else:
                    data[col] = data[col].fillna(data[col].mode().iloc[0] if not data[col].mode().empty else 'unknown')

        return data

    def _normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize numeric features."""

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['lat', 'lon', 'id', 'label_gold']]

        # Standard normalization
        for col in numeric_cols:
            if col in data.columns and data[col].std() > 0:
                data[f'{col}_normalized'] = (data[col] - data[col].mean()) / data[col].std()

        return data

    def get_feature_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of engineered features."""

        original_features = len(data.columns)

        # Count feature types
        geological_features = len([col for col in data.columns if 'geo' in col.lower()])
        terrain_features = len([col for col in data.columns if any(t in col.lower() for t in ['elev', 'slope', 'aspect', 'curvature'])])
        spectral_features = len([col for col in data.columns if any(s in col.lower() for s in ['ndvi', 'ndwi', 'band'])])

        return {
            'total_features': original_features,
            'geological_features': geological_features,
            'terrain_features': terrain_features,
            'spectral_features': spectral_features,
            'engineered_features': geological_features + terrain_features + spectral_features
        }
