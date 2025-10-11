#!/usr/bin/env python3
"""
GeoAuPredict Phase 3: Predictive Modeling - Data Preprocessing Module

This module handles data loading, preprocessing, and feature engineering
for machine learning models in gold exploration prediction.

Key Features:
- Load integrated geospatial datasets from Phase 2
- Handle missing values and outliers
- Feature scaling and normalization
- Spatial feature engineering
- Train/test split with spatial awareness

Author: GeoAuPredict Team
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import logging

logger = logging.getLogger(__name__)


class GeospatialDataPreprocessor:
    """
    Handles preprocessing of geospatial data for machine learning.

    Features:
    - Missing value imputation
    - Outlier detection and handling
    - Feature scaling and normalization
    - Spatial feature engineering
    - Train/test splitting with spatial awareness
    """

    def __init__(self,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 spatial_split: bool = True,
                 scaler_type: str = 'standard'):
        """
        Initialize the preprocessor.

        Args:
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
            spatial_split: Whether to use spatial splitting for train/test
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
        """
        self.test_size = test_size
        self.random_state = random_state
        self.spatial_split = spatial_split

        # Initialize scaler
        self.scaler_type = scaler_type
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        # Feature information
        self.feature_columns = []
        self.spatial_columns = ['lat', 'lon']
        self.target_column = 'label_gold'
        self.numeric_features = []
        self.categorical_features = []

        logger.info(f"Initialized GeospatialDataPreprocessor with {scaler_type} scaling")

    def load_integrated_dataset(self, dataset_path: str) -> pd.DataFrame:
        """
        Load integrated geospatial dataset from Phase 2.

        Args:
            dataset_path: Path to integrated dataset CSV

        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading integrated dataset from {dataset_path}")

        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        # Determine format and load
        file_ext = Path(dataset_path).suffix.lower()

        if file_ext == '.csv':
            df = pd.read_csv(dataset_path)
        elif file_ext in ['.parquet', '.feather']:
            df = pd.read_parquet(dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        logger.info(f"Loaded dataset: {df.shape[0]} samples, {df.shape[1]} features")

        # Validate required columns
        required_cols = self.spatial_columns + [self.target_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return df

    def handle_missing_values(self, df: pd.DataFrame,
                            strategy: str = 'median') -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            df: Input DataFrame
            strategy: Imputation strategy ('mean', 'median', 'mode', 'interpolation')

        Returns:
            DataFrame with imputed missing values
        """
        logger.info(f"Handling missing values with {strategy} strategy")

        df_processed = df.copy()

        # Separate features by type
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns

        # Handle numeric columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                if strategy == 'mean':
                    fill_value = df[col].mean()
                elif strategy == 'median':
                    fill_value = df[col].median()
                elif strategy == 'mode':
                    fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else 0
                elif strategy == 'interpolation':
                    df_processed[col] = df[col].interpolate(method='linear')
                    continue
                else:
                    raise ValueError(f"Unknown imputation strategy: {strategy}")

                df_processed[col] = df[col].fillna(fill_value)
                logger.info(f"Imputed {df[col].isnull().sum()} missing values in {col}")

        # Handle categorical columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df_processed[col] = df[col].fillna(df[col].mode().iloc[0])

        missing_after = df_processed.isnull().sum().sum()
        logger.info(f"Remaining missing values: {missing_after}")

        return df_processed

    def detect_outliers(self, df: pd.DataFrame,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect and handle outliers in numeric features.
        Also handles infinite and NaN values.

        Args:
            df: Input DataFrame
            method: Outlier detection method ('iqr', 'zscore', 'isolation_forest')
            threshold: Threshold for outlier detection

        Returns:
            DataFrame with outliers handled
        """
        logger.info(f"Detecting outliers using {method} method")

        # First, handle infinite and NaN values
        df_processed = df.copy()

        # Replace infinite values with NaN
        df_processed = df_processed.replace([np.inf, -np.inf], np.nan)

        # Fill NaN values with median before outlier detection
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())

        for col in numeric_cols:
            if col in self.spatial_columns or col == self.target_column:
                continue  # Skip spatial and target columns

            original_count = len(df_processed)

            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                # Cap outliers
                df_processed[col] = np.clip(df[col], lower_bound, upper_bound)

            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > threshold

                # Use median for outliers
                median_val = df[col].median()
                df_processed.loc[outliers, col] = median_val

            logger.info(f"Processed outliers in {col}: {original_count - len(df_processed)} values adjusted")

        # Final check for any remaining infinite/NaN values
        df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
        for col in numeric_cols:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())

        return df_processed

    def create_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional spatial features for better modeling.
        Handles division by zero and infinite values properly.

        Args:
            df: Input DataFrame with lat/lon coordinates

        Returns:
            DataFrame with additional spatial features
        """
        logger.info("Creating spatial features")

        df_processed = df.copy()

        # Distance-based features (if multiple points)
        if len(df) > 1:
            coords = df[['lat', 'lon']].values

            # Nearest neighbor distances
            from scipy.spatial.distance import pdist, squareform
            distances = squareform(pdist(coords))

            # Remove self-distances (diagonal)
            np.fill_diagonal(distances, np.inf)

            # Add distance-based features
            df_processed['dist_to_nearest_sample'] = distances.min(axis=1)
            df_processed['mean_dist_to_samples'] = distances.mean(axis=1)

            # Handle potential infinite values in distance calculations
            df_processed['dist_to_nearest_sample'] = df_processed['dist_to_nearest_sample'].replace([np.inf, -np.inf], 1000)
            df_processed['mean_dist_to_samples'] = df_processed['mean_dist_to_samples'].replace([np.inf, -np.inf], 1000)

            # Spatial clustering features
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=min(10, len(df)//10), random_state=self.random_state)
            clusters = kmeans.fit_predict(coords)
            df_processed['spatial_cluster'] = clusters

        # Geographic region encoding (for Colombia)
        df_processed['region_north'] = (df['lat'] > 7.0).astype(int)
        df_processed['region_south'] = (df['lat'] <= 7.0).astype(int)
        df_processed['region_east'] = (df['lon'] > -72.0).astype(int)
        df_processed['region_west'] = (df['lon'] <= -72.0).astype(int)

        # Elevation-based features (if elevation column exists)
        if 'elevation' in df.columns:
            # Handle infinite/NaN values in elevation
            df_processed['elevation'] = df_processed['elevation'].replace([np.inf, -np.inf], np.nan)
            if df_processed['elevation'].isnull().sum() > 0:
                df_processed['elevation'] = df_processed['elevation'].fillna(df_processed['elevation'].median())

            df_processed['elevation_category'] = pd.cut(
                df_processed['elevation'],
                bins=[0, 500, 1000, 2000, np.inf],
                labels=['low', 'medium', 'high', 'very_high']
            )

        # Handle any infinite values in the final dataframe
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            df_processed[col] = df_processed[col].replace([np.inf, -np.inf], df_processed[col].median())

        logger.info(f"Added {len(df_processed.columns) - len(df.columns)} spatial features")

        return df_processed

    def select_features(self, df: pd.DataFrame,
                       method: str = 'mutual_info',
                       k: Optional[int] = None) -> pd.DataFrame:
        """
        Select most important features for modeling.

        Args:
            df: Input DataFrame
            method: Feature selection method ('mutual_info', 'f_classif', 'correlation')
            k: Number of features to select (None for all)

        Returns:
            DataFrame with selected features
        """
        logger.info(f"Selecting features using {method} method")

        # Separate features and target
        feature_cols = [col for col in df.columns
                       if col not in self.spatial_columns + [self.target_column]]

        X = df[feature_cols]
        y = df[self.target_column]

        if method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=k)
        elif method == 'f_classif':
            selector = SelectKBest(f_classif, k=k)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")

        X_selected = selector.fit_transform(X, y)

        # Get selected feature names
        selected_features = np.array(feature_cols)[selector.get_support()]
        logger.info(f"Selected {len(selected_features)} features: {list(selected_features)}")

        # Create result DataFrame
        df_selected = pd.DataFrame(X_selected, columns=selected_features, index=df.index)
        df_selected[self.spatial_columns] = df[self.spatial_columns]
        df_selected[self.target_column] = df[self.target_column]

        return df_selected

    def scale_features(self, df: pd.DataFrame,
                      fit: bool = True) -> pd.DataFrame:
        """
        Scale numeric features using the configured scaler.
        Also handles categorical features properly for model training.

        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler or just transform

        Returns:
            DataFrame with scaled features
        """
        logger.info(f"Scaling features with {self.scaler_type} scaler")

        # Identify different column types for better handling
        all_columns = set(df.columns)
        spatial_cols = set(self.spatial_columns)
        target_cols = {self.target_column}

        # Get numeric and categorical columns
        numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = set(df.select_dtypes(exclude=[np.number]).columns)

        # Columns to exclude from scaling (spatial, target, and categorical)
        exclude_cols = spatial_cols | target_cols | categorical_cols

        # Columns available for scaling (numeric columns not in exclude list)
        feature_cols = list(numeric_cols - exclude_cols)

        logger.info(f"Column analysis:")
        logger.info(f"  Total columns: {len(all_columns)}")
        logger.info(f"  Numeric columns: {len(numeric_cols)}")
        logger.info(f"  Categorical columns: {len(categorical_cols)}")
        logger.info(f"  Spatial columns: {len(spatial_cols)}")
        logger.info(f"  Target columns: {len(target_cols)}")

        if categorical_cols:
            logger.info(f"  Categorical columns found: {sorted(categorical_cols)}")
            logger.info("  These will be encoded for model compatibility")

        if not feature_cols:
            logger.warning("No numeric features found for scaling")
            # If no numeric features, we still need to handle categorical columns
            df_result = df.copy()
            if categorical_cols:
                df_result = self._encode_categorical_features(df_result)
            return df_result

        logger.info(f"Scaling {len(feature_cols)} numeric features: {feature_cols[:5]}{'...' if len(feature_cols) > 5 else ''}")

        X_features = df[feature_cols].values

        if fit:
            X_scaled = self.scaler.fit_transform(X_features)
        else:
            X_scaled = self.scaler.transform(X_features)

        # Create result DataFrame
        df_scaled = df.copy()
        df_scaled[feature_cols] = X_scaled

        # Handle categorical columns
        if categorical_cols:
            logger.info(f"Calling categorical encoding for columns: {sorted(categorical_cols)}")
            df_scaled = self._encode_categorical_features(df_scaled)
            logger.info("Categorical encoding completed")

        logger.info(f"✅ Successfully scaled {len(feature_cols)} numeric features")
        logger.info(f"✅ Encoded {len(categorical_cols)} categorical columns")

        return df_scaled

    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features for model compatibility.

        Args:
            df: DataFrame with categorical columns

        Returns:
            DataFrame with encoded categorical features
        """
        df_encoded = df.copy()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns

        logger.info(f"Encoding {len(categorical_cols)} categorical columns: {list(categorical_cols)}")

        for col in categorical_cols:
            if col in self.spatial_columns or col == self.target_column:
                logger.info(f"Skipping spatial/target column: {col}")
                continue  # Skip spatial and target columns

            logger.info(f"Encoding column: {col}")
            unique_values = df_encoded[col].unique()
            logger.info(f"  Unique values in {col}: {unique_values}")

            # Handle NaN values first
            if df_encoded[col].isnull().any():
                logger.info(f"  Handling NaN values in {col}")
                # Use most frequent value for NaN replacement
                most_frequent = df_encoded[col].mode().iloc[0] if not df_encoded[col].mode().empty else unique_values[0]
                df_encoded[col] = df_encoded[col].fillna(most_frequent)

            if len(unique_values) == 2:
                # Binary categorical - use 0/1 encoding
                df_encoded[col] = (df_encoded[col] == unique_values[0]).astype(int)
                logger.info(f"  Binary encoding applied to {col}")
            else:
                # Multi-class categorical - use label encoding
                value_to_int = {val: idx for idx, val in enumerate(sorted(unique_values))}
                df_encoded[col] = df_encoded[col].map(value_to_int)
                logger.info(f"  Label encoding applied to {col}: {value_to_int}")

        # Final verification - ensure no categorical columns remain
        remaining_categorical = df_encoded.select_dtypes(exclude=[np.number]).columns
        if len(remaining_categorical) > 0:
            logger.warning(f"Warning: {len(remaining_categorical)} categorical columns remain: {list(remaining_categorical)}")
            # Force conversion of remaining categorical columns
            for col in remaining_categorical:
                if col not in self.spatial_columns and col != self.target_column:
                    logger.info(f"Force encoding remaining categorical column: {col}")
                    unique_vals = df_encoded[col].unique()
                    val_to_int = {val: idx for idx, val in enumerate(sorted(unique_vals))}
                    df_encoded[col] = df_encoded[col].map(val_to_int).astype(float)

        return df_encoded

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Splitting data with test_size={self.test_size}")

        # Prepare features and target
        feature_cols = [col for col in df.columns
                       if col not in self.spatial_columns + [self.target_column, 'spatial_block']]

        X = df[feature_cols]
        y = df[self.target_column]

        if self.spatial_split:
            # Use spatial coordinates for stratified splitting
            # Create spatial blocks for better geographic representation
            df['spatial_block'] = self._create_spatial_blocks(df)

            # Split by spatial blocks to ensure geographic separation
            unique_blocks = df['spatial_block'].unique()
            train_blocks, test_blocks = train_test_split(
                unique_blocks,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=df.groupby('spatial_block')[self.target_column].mean().round()
            )

            train_mask = df['spatial_block'].isin(train_blocks)
            test_mask = df['spatial_block'].isin(test_blocks)

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

        else:
            # Standard random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y if len(y.unique()) < 20 else None  # Stratify if not too many classes
            )

        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Train class distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"Test class distribution: {y_test.value_counts().to_dict()}")

        return X_train, X_test, y_train, y_test

    def _create_spatial_blocks(self, df: pd.DataFrame,
                              block_size: float = 1.0) -> pd.Series:
        """
        Create spatial blocks for geographic splitting.

        Args:
            df: DataFrame with lat/lon coordinates
            block_size: Size of spatial blocks in degrees

        Returns:
            Series with block identifiers
        """
        # Create grid blocks
        lat_blocks = (df['lat'] // block_size).astype(str)
        lon_blocks = (df['lon'] // block_size).astype(str)

        return lat_blocks + '_' + lon_blocks

    def preprocess_complete_pipeline(self, dataset_path: str) -> Dict[str, pd.DataFrame]:
        """
        Run complete preprocessing pipeline.

        Args:
            dataset_path: Path to integrated dataset

        Returns:
            Dictionary with preprocessed datasets and metadata
        """
        logger.info("Starting complete preprocessing pipeline")

        # Load data
        df = self.load_integrated_dataset(dataset_path)

        # Handle missing values
        df = self.handle_missing_values(df)

        # Detect and handle outliers
        df = self.detect_outliers(df)

        # Create spatial features
        df = self.create_spatial_features(df)

        # Feature selection (optional, can be skipped for now)
        # df = self.select_features(df)

        # Scale features
        df = self.scale_features(df, fit=True)

        # Split data
        X_train, X_test, y_train, y_test = self.split_data(df)

        # Prepare final datasets
        train_df = pd.DataFrame(X_train)
        train_df[self.target_column] = y_train
        train_df[self.spatial_columns] = df.loc[y_train.index, self.spatial_columns]

        test_df = pd.DataFrame(X_test)
        test_df[self.target_column] = y_test
        test_df[self.spatial_columns] = df.loc[y_test.index, self.spatial_columns]

        results = {
            'train_df': train_df,
            'test_df': test_df,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_columns': X_train.columns.tolist(),
            'scaler': self.scaler
        }

        logger.info("✅ Complete preprocessing pipeline finished")

        return results


def create_sample_training_data(output_path: str = 'phase3_sample_data') -> Dict[str, str]:
    """
    Create sample training data for testing Phase 3 models.

    Args:
        output_path: Directory to save sample data

    Returns:
        Dictionary with paths to created files
    """
    logger.info("Creating sample training data for Phase 3")

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    n_samples = 1000

    # Create realistic sample data
    sample_data = {
        'lat': np.random.uniform(4.0, 12.0, n_samples),  # Colombia bounds
        'lon': np.random.uniform(-75.0, -67.0, n_samples),
        'elevation': np.random.exponential(500, n_samples) + 100,
        'slope_degrees': np.random.exponential(15, n_samples),
        'ndvi': np.random.normal(0.3, 0.1, n_samples),
        'nbr': np.random.normal(0.2, 0.15, n_samples),
        'clay_index': np.random.exponential(1.2, n_samples),
        'iron_index': np.random.exponential(0.8, n_samples),
        'twi': np.random.exponential(8, n_samples),
        'flow_accumulation': np.random.exponential(1000, n_samples),
        'dist_to_faults': np.random.exponential(2000, n_samples),
        'Au_ppm': np.random.exponential(0.05, n_samples),
        'As_ppm': np.random.exponential(3, n_samples),
        'Cu_ppm': np.random.exponential(25, n_samples),
        'dist_to_nearest_sample': np.random.exponential(500, n_samples),
        'study_area': np.random.choice(['Antioquia', 'Cauca', 'Choco', 'Nariño', 'Cundinamarca'], n_samples),  # Add study_area column
    }

    # Create gold presence labels (correlated with features)
    # Higher probability near faults, with certain geochemical signatures
    gold_prob = (
        0.1 +  # Base probability
        0.3 * (1 / (1 + sample_data['dist_to_faults'] / 1000)) +  # Closer to faults
        0.2 * (sample_data['Au_ppm'] > 0.1).astype(float) +  # High Au
        0.2 * (sample_data['As_ppm'] > 5).astype(float) +  # High As (pathfinder)
        0.1 * (sample_data['Cu_ppm'] > 50).astype(float) +  # High Cu
        0.1 * np.random.random(n_samples)  # Random component
    )

    # Handle potential division by zero in probability calculation
    gold_prob = np.clip(gold_prob, 0, 1)

    # Ensure no infinite values in probability
    gold_prob = np.where(np.isfinite(gold_prob), gold_prob, 0.5)

    sample_data['label_gold'] = (gold_prob > 0.4).astype(int)

    # Create DataFrame and save
    df = pd.DataFrame(sample_data)

    # Add some correlated features
    df['Au_As_ratio'] = df['Au_ppm'] / (df['As_ppm'] + 0.1)
    df['alteration_index'] = df['clay_index'] * df['iron_index']

    # Handle potential infinite values in derived features
    for col in ['Au_As_ratio', 'alteration_index']:
        df[col] = df[col].replace([np.inf, -np.inf], df[col].median())
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Save to CSV
    output_file = output_path / "sample_training_data.csv"
    df.to_csv(output_file, index=False)

    logger.info(f"Created sample training data: {output_file}")

    return {
        'training_data': str(output_file)
    }
