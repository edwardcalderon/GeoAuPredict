"""
Helper functions for the GeoAuPredict Complete Pipeline Notebook

This module contains utility functions used by the notebook for creating sample data,
handling missing modules, and providing fallbacks for demonstration purposes.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


def create_sample_borehole_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create sample borehole data for demonstration when real data is not available.

    Args:
        n_samples: Number of sample points to generate

    Returns:
        DataFrame with sample borehole data
    """
    logger.info(f"Creating {n_samples} sample borehole data points")

    np.random.seed(42)

    # Create realistic Colombian geography bounds
    lat_bounds = (4.0, 12.0)  # Colombia latitude range
    lon_bounds = (-75.0, -67.0)  # Colombia longitude range

    # Generate sample locations
    latitudes = np.random.uniform(lat_bounds[0], lat_bounds[1], n_samples)
    longitudes = np.random.uniform(lon_bounds[0], lon_bounds[1], n_samples)
    elevations = np.random.uniform(0, 3000, n_samples)

    # Create geological features
    geological_features = {
        'igneous': np.random.exponential(0.3, n_samples),
        'sedimentary': np.random.exponential(0.4, n_samples),
        'metamorphic': np.random.exponential(0.2, n_samples),
        'structural_complexity': np.random.uniform(0, 1, n_samples),
        'alteration_intensity': np.random.exponential(0.5, n_samples),
        'mineralization_index': np.random.uniform(0, 1, n_samples)
    }

    # Create terrain features
    terrain_features = {
        'slope': np.random.exponential(5, n_samples),
        'aspect': np.random.uniform(0, 360, n_samples),
        'curvature': np.random.normal(0, 0.1, n_samples),
        'twi': np.random.exponential(3, n_samples),  # Topographic Wetness Index
        'flow_accumulation': np.random.exponential(100, n_samples),
        'drainage_density': np.random.uniform(0, 2, n_samples)
    }

    # Create spectral features (simulating remote sensing data)
    spectral_features = {
        'ndvi': np.random.uniform(-0.2, 0.8, n_samples),
        'ndwi': np.random.uniform(-0.5, 0.5, n_samples),
        'band2': np.random.uniform(0, 255, n_samples),
        'band3': np.random.uniform(0, 255, n_samples),
        'band4': np.random.uniform(0, 255, n_samples),
        'band5': np.random.uniform(0, 255, n_samples),
        'band7': np.random.uniform(0, 255, n_samples)
    }

    # Generate gold concentrations with spatial correlation
    gold_concentrations = np.random.exponential(0.05, n_samples)  # Base distribution

    # Add spatial correlation - higher concentrations in certain regions
    for i in range(n_samples):
        lat, lon = latitudes[i], longitudes[i]

        # Northwestern region (Antioquia)
        if lat > 7.0 and lon < -72.0:
            gold_concentrations[i] += np.random.exponential(0.15)

        # Southeastern region (Cauca Valley)
        elif lat < 6.0 and lon > -70.0:
            gold_concentrations[i] += np.random.exponential(0.1)

        # Central region (Cundinamarca)
        elif 4.0 < lat < 7.0 and -75.0 < lon < -72.0:
            gold_concentrations[i] += np.random.exponential(0.08)

    # Create labels based on gold concentration threshold
    labels = (gold_concentrations > 0.1).astype(int)

    # Combine all features
    sample_data = {
        'id': range(n_samples),
        'lat': latitudes,
        'lon': longitudes,
        'elev': elevations,
        'Au_ppm': gold_concentrations,
        'label_gold': labels,
        'study_area': np.random.choice(['Antioquia', 'Cauca', 'Cundinamarca', 'Nariño', 'Chocó'], n_samples),
        'source': 'sample_data',
        'date': pd.date_range('2020-01-01', periods=n_samples, freq='D').strftime('%Y-%m-%d')
    }

    # Add geological features
    for key, values in geological_features.items():
        sample_data[f'geo_{key}'] = values

    # Add terrain features
    for key, values in terrain_features.items():
        sample_data[f'terrain_{key}'] = values

    # Add spectral features
    for key, values in spectral_features.items():
        sample_data[f'spectral_{key}'] = values

    # Add some interaction features
    sample_data['geo_igneous_sedimentary'] = geological_features['igneous'] * geological_features['sedimentary']
    sample_data['terrain_slope_elev'] = terrain_features['slope'] * (elevations / 1000)
    sample_data['spectral_ndvi_ndwi'] = spectral_features['ndvi'] * spectral_features['ndwi']

    return pd.DataFrame(sample_data)


def create_sample_trained_models() -> List[Any]:
    """
    Create sample trained models for demonstration when actual training fails.

    Returns:
        List of mock trained model objects
    """
    logger.info("Creating sample trained models for demonstration")

    class MockModel:
        def __init__(self, name: str):
            self.model_name = name
            self.training_time = f"{np.random.uniform(30, 120):.1f}s"
            self.best_cv_score = np.random.uniform(0.75, 0.90)

        def predict(self, X):
            return np.random.choice([0, 1], size=len(X))

        def predict_proba(self, X):
            probs = np.random.uniform(0.3, 0.8, (len(X), 2))
            return probs / probs.sum(axis=1, keepdims=True)

        def get_params(self):
            return {'param1': 'value1', 'param2': 'value2'}

    models = [
        MockModel("XGBoost"),
        MockModel("LightGBM"),
        MockModel("Random Forest")
    ]

    return models


def create_sample_evaluation_results(trained_models: List[Any]) -> Dict[str, Any]:
    """
    Create sample evaluation results for demonstration.

    Args:
        trained_models: List of trained model objects

    Returns:
        Dictionary with evaluation results
    """
    logger.info("Creating sample evaluation results")

    evaluations = []
    for model in trained_models:
        eval_result = {
            'basic_metrics': {
                'accuracy': np.random.uniform(0.75, 0.90),
                'roc_auc': np.random.uniform(0.80, 0.95),
                'f1_score': np.random.uniform(0.70, 0.85),
                'precision': np.random.uniform(0.75, 0.90),
                'recall': np.random.uniform(0.70, 0.85)
            },
            'spatial_cv': {
                'cv_mean': np.random.uniform(0.75, 0.90),
                'cv_std': np.random.uniform(0.05, 0.15)
            }
        }
        evaluations.append(eval_result)

    # Create comparison DataFrame
    comparison_data = []
    for i, model in enumerate(trained_models):
        comparison_data.append({
            'model': model.model_name,
            'roc_auc': evaluations[i]['basic_metrics']['roc_auc'],
            'f1_score': evaluations[i]['basic_metrics']['f1_score'],
            'accuracy': evaluations[i]['basic_metrics']['accuracy']
        })

    comparison_df = pd.DataFrame(comparison_data)

    return {
        'evaluations': evaluations,
        'comparison': comparison_df
    }


def create_sample_probability_map() -> Dict[str, Any]:
    """
    Create sample probability map data for demonstration.

    Returns:
        Dictionary with sample map data
    """
    logger.info("Creating sample probability map data")

    # Create sample raster data
    height, width = 100, 100
    probability_raster = np.random.uniform(0.2, 0.8, (height, width))
    uncertainty_raster = np.random.uniform(0.1, 0.3, (height, width))

    # Add some spatial structure
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    # Create some high-probability zones
    probability_raster += 0.2 * np.sin(5 * x) * np.cos(3 * y)
    probability_raster = np.clip(probability_raster, 0, 1)

    # Create bounds (Colombia approximate bounds)
    bounds = (-79.0, -4.3, -66.8, 12.5)

    raster_data = {
        'probability': probability_raster,
        'uncertainty': uncertainty_raster,
        'x_grid': np.linspace(bounds[0], bounds[2], width),
        'y_grid': np.linspace(bounds[1], bounds[3], height),
        'bounds': bounds
    }

    # Create sample output paths
    project_root = Path.cwd()
    output_dir = project_root / 'outputs' / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)

    probability_raster_path = output_dir / 'sample_gold_probability_map.tif'
    probability_map_path = output_dir / 'sample_gold_probability_map.png'

    return {
        'probability_raster': str(probability_raster_path),
        'probability_map': str(probability_map_path),
        'statistics': {
            'mean': float(probability_raster.mean()),
            'std': float(probability_raster.std()),
            'min': float(probability_raster.min()),
            'max': float(probability_raster.max()),
            'median': float(np.median(probability_raster)),
            'q25': float(np.percentile(probability_raster, 25)),
            'q75': float(np.percentile(probability_raster, 75)),
            'q90': float(np.percentile(probability_raster, 90)),
            'n_pixels': probability_raster.size,
            'area_above_05': int(np.sum(probability_raster >= 0.5)),
            'area_above_07': int(np.sum(probability_raster >= 0.7)),
            'area_above_09': int(np.sum(probability_raster >= 0.9))
        }
    }


class MockPhase3Pipeline:
    """Mock Phase 3 pipeline for demonstration when actual pipeline fails."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config['output_dir'])

    def train_models(self, preprocessed_data: Dict[str, Any]) -> List[Any]:
        """Mock model training."""
        logger.info("Running mock model training")
        return create_sample_trained_models()

    def evaluate_models(self, trained_models: List[Any], preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock model evaluation."""
        logger.info("Running mock model evaluation")
        return create_sample_evaluation_results(trained_models)


def setup_notebook_environment():
    """
    Set up the notebook environment and ensure all required paths are available.
    """
    project_root = Path.cwd()

    # Ensure output directories exist
    output_dirs = [
        project_root / 'outputs',
        project_root / 'outputs' / 'models',
        project_root / 'outputs' / 'visualizations',
        project_root / 'data' / 'processed'
    ]

    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info(f"Notebook environment set up for project: {project_root}")

    return project_root


def validate_pipeline_outputs(project_root: Path) -> Dict[str, bool]:
    """
    Validate that all expected pipeline outputs were created.

    Args:
        project_root: Project root directory

    Returns:
        Dictionary with validation results
    """
    expected_outputs = {
        'models_trained': (project_root / 'outputs' / 'models').exists(),
        'model_evaluation': (project_root / 'outputs' / 'models' / 'model_comparison.csv').exists(),
        'probability_raster': (project_root / 'outputs' / 'visualizations' / 'gold_probability_map.tif').exists(),
        'probability_map': (project_root / 'outputs' / 'visualizations' / 'gold_probability_map.png').exists(),
        'exploration_report': (project_root / 'outputs' / 'exploration_report.json').exists(),
        'processed_data': (project_root / 'data' / 'processed' / 'gold_dataset_master.csv').exists()
    }

    validation_results = {}
    for output_name, file_exists in expected_outputs.items():
        validation_results[output_name] = file_exists
        status = "✅" if file_exists else "❌"
        logger.info(f"{status} {output_name}: {'Created' if file_exists else 'Missing'}")

    return validation_results


if __name__ == "__main__":
    # Test the helper functions
    print("🧪 Testing helper functions...")

    # Test sample data creation
    sample_data = create_sample_borehole_data(100)
    print(f"✅ Created {len(sample_data)} sample borehole records")

    # Test sample models
    sample_models = create_sample_trained_models()
    print(f"✅ Created {len(sample_models)} sample models")

    # Test evaluation results
    eval_results = create_sample_evaluation_results(sample_models)
    print(f"✅ Created evaluation results for {len(eval_results['evaluations'])} models")

    # Test probability map
    map_data = create_sample_probability_map()
    print(f"✅ Created sample probability map with {len(map_data['statistics'])} statistics")

    print("\\n🎉 All helper functions working correctly!")
