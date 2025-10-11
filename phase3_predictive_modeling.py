#!/usr/bin/env python3
"""
GeoAuPredict Phase 3: Predictive Modeling Pipeline

This script orchestrates the complete Phase 3 pipeline for gold presence prediction:

1. Data Loading & Preprocessing
2. Model Training with multiple algorithms
3. Spatial Cross-Validation
4. Model Evaluation with geographic metrics
5. Prediction and Probability Mapping

Usage:
    python phase3_predictive_modeling.py --config phase3_config.yaml
    python phase3_predictive_modeling.py --demo  # Run with sample data
"""

import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys
import os
import json
import pandas as pd
import numpy as np

from src.data.data_preprocessing import GeospatialDataPreprocessor, create_sample_training_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.models.gold_prediction_models import (
    ModelFactory, RandomForestModel, XGBoostModel, LightGBMModel,
    EnsembleModel, compare_models
)
from src.models.spatial_cross_validation import SpatialCrossValidator
from src.models.probability_mapping import ProbabilityMapper, create_sample_predictions
from src.models.model_evaluation import GeospatialModelEvaluator, compare_model_evaluations

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase3_predictive_modeling.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Phase3Pipeline:
    """Main pipeline for Phase 3 predictive modeling."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Phase 3 pipeline.

        Args:
            config: Configuration dictionary with pipeline parameters
        """
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'phase3_output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.data_preprocessor = GeospatialDataPreprocessor(
            test_size=config.get('test_size', 0.2),
            random_state=config.get('random_state', 42),
            spatial_split=config.get('spatial_split', True),
            scaler_type=config.get('scaler_type', 'standard')
        )

        # Safe scaling fallback
        self.use_safe_scaling = config.get('use_safe_scaling', True)

        self.spatial_cv = SpatialCrossValidator(
            method=config.get('cv_method', 'geographic_blocks'),
            n_splits=config.get('n_splits', 5),
            block_size=config.get('block_size', 1.0),
            random_state=config.get('random_state', 42)
        )

        self.probability_mapper = ProbabilityMapper(
            interpolation_method=config.get('interpolation_method', 'kriging'),
            pixel_size=config.get('pixel_size', 1000),
            uncertainty_estimation=config.get('uncertainty_estimation', True)
        )

    def safe_scale_features(self, X_train, X_test, target_column='label_gold'):
        """
        Safely scale features, handling categorical columns automatically.

        Args:
            X_train: Training features DataFrame
            X_test: Testing features DataFrame
            target_column: Name of target column to exclude from scaling

        Returns:
            Tuple of (X_train_scaled, X_test_scaled, scaler)
        """
        logger.info("🔍 Analyzing features for preprocessing...")

        # Work on copies to avoid mutating inputs
        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()

        # Identify column types
        numeric_cols = X_train_processed.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = X_train_processed.select_dtypes(include=['object', 'category']).columns.tolist()

        # Ensure target is not considered a feature
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        if target_column in categorical_cols:
            categorical_cols.remove(target_column)

        logger.info("📊 Feature Analysis:")
        logger.info(f"   Total features: {len(X_train_processed.columns)}")
        logger.info(f"   Numeric features: {len(numeric_cols)}")
        logger.info(f"   Categorical features: {len(categorical_cols)}")

        # Label-encode categorical features
        if categorical_cols:
            logger.info(f"   Label encoding {len(categorical_cols)} categorical features: {sorted(categorical_cols)}")
            from sklearn.preprocessing import LabelEncoder
            for col in categorical_cols:
                le = LabelEncoder()
                # Fit on combined train+test to capture all categories
                combined = pd.concat([X_train_processed[col], X_test_processed[col]], axis=0).astype(str)
                le.fit(combined)
                X_train_processed[col] = le.transform(X_train_processed[col].astype(str))
                X_test_processed[col] = le.transform(X_test_processed[col].astype(str))

        # Features to scale (numeric + encoded categorical)
        feature_cols = numeric_cols + categorical_cols
        if not feature_cols:
            logger.warning("⚠️ No features to scale!")
            return X_train_processed, X_test_processed, None

        logger.info(f"✅ Scaling {len(feature_cols)} features")
        try:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_processed[feature_cols] = scaler.fit_transform(X_train_processed[feature_cols])
            X_test_processed[feature_cols] = scaler.transform(X_test_processed[feature_cols])

            logger.info("✅ Feature preprocessing completed successfully")
            return X_train_processed, X_test_processed, scaler

        except Exception as e:
            logger.error(f"❌ Feature preprocessing failed: {e}")
            logger.info("🔄 Returning encoded but unscaled features")
            return X_train_processed, X_test_processed, None

    def load_and_preprocess_data(self) -> Dict[str, Any]:
        """
        Load and preprocess training data.

        Returns:
            Dictionary with preprocessed datasets
        """
        logger.info("🚀 Starting data loading and preprocessing...")

        data_config = self.config.get('data', {})
        training_data_path = data_config.get('training_data_path')

        if not training_data_path:
            logger.warning("No training data path provided, using sample data")
            sample_files = create_sample_training_data(str(self.output_dir / "sample_data"))
            training_data_path = sample_files['training_data']

        try:
            # Preprocess data
            preprocessed_data = self.data_preprocessor.preprocess_complete_pipeline(training_data_path)

            # Save preprocessed data info
            info_path = self.output_dir / "preprocessed_data_info.json"
            info_data = {
                'training_samples': len(preprocessed_data['X_train']),
                'test_samples': len(preprocessed_data['X_test']),
                'feature_columns': preprocessed_data['feature_columns'],
                'class_distribution': {
                    'train': preprocessed_data['y_train'].value_counts().to_dict(),
                    'test': preprocessed_data['y_test'].value_counts().to_dict()
                }
            }

            with open(info_path, 'w') as f:
                json.dump(info_data, f, indent=2)

            logger.info("✅ Data preprocessing completed")
            return preprocessed_data

        except Exception as e:
            logger.error(f"❌ Fallback preprocessing also failed: {e}")
            raise

    def train_models(self, preprocessed_data: Dict[str, Any]) -> List[Any]:
        """
        Train multiple models on preprocessed data.
        """
        logger.info("🏋️ Starting model training...")

        X_train = preprocessed_data['X_train']
        y_train = preprocessed_data['y_train']

        # Get model configurations
        models_config = self.config.get('models', {})

        # Create models
        models_to_train = []

        if models_config.get('random_forest', True):
            rf_model = RandomForestModel(
                n_estimators=models_config.get('rf_n_estimators', 100),
                max_depth=models_config.get('rf_max_depth', 10),
                random_state=self.config.get('random_state', 42)
            )
            models_to_train.append(rf_model)

        if models_config.get('xgboost', True):
            xgb_model = XGBoostModel(
                n_estimators=models_config.get('xgb_n_estimators', 100),
                max_depth=models_config.get('xgb_max_depth', 6),
                learning_rate=models_config.get('xgb_learning_rate', 0.1),
                random_state=self.config.get('random_state', 42)
            )
            models_to_train.append(xgb_model)

        if models_config.get('lightgbm', True):
            lgb_model = LightGBMModel(
                n_estimators=models_config.get('lgb_n_estimators', 100),
                max_depth=models_config.get('lgb_max_depth', 6),
                learning_rate=models_config.get('lgb_learning_rate', 0.1),
                random_state=self.config.get('random_state', 42)
            )
            models_to_train.append(lgb_model)

        # Train models
        trained_models = []
        for model in models_to_train:
            logger.info(f"Training {model.model_name}...")

            try:
                training_metrics = model.train(X_train, y_train)
                logger.info(f"✅ {model.model_name} trained: CV AUC = {training_metrics.get('cv_mean_roc_auc', 0):.3f}")
                trained_models.append(model)

            except Exception as e:
                logger.error(f"❌ {model.model_name} training failed: {e}")
                continue

        logger.info(f"✅ Successfully trained {len(trained_models)} models")
        return trained_models


    def evaluate_models(self, trained_models: List[Any], preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate trained models using spatial cross-validation with enhanced error handling."""
        logger.info("Starting model evaluation...")

        if not trained_models:
            logger.warning("⚠️ No trained models provided for evaluation")
            return {
                'evaluations': [],
                'comparison': pd.DataFrame(columns=['model','roc_auc','f1_score','accuracy','precision','recall']),
                'error': "No models to evaluate"
        }
        
        try:
            X_test = preprocessed_data.get('X_test')
            y_test = preprocessed_data.get('y_test')
            test_df = preprocessed_data.get('test_df')
            
            if X_test is None or y_test is None:
                raise ValueError("Missing required test data (X_test or y_test)")
                
            if test_df is None:
                logger.warning("⚠️ No test_df provided, spatial analysis will be limited")
                coordinates = None
            else:
                try:
                    coordinates = test_df[['lat', 'lon']].values
                except KeyError:
                    logger.warning("⚠️ Missing 'lat' or 'lon' in test_df, spatial analysis disabled")
                    coordinates = None

            evaluations = []
            for model in trained_models:
                model_name = getattr(model, 'model_name', 'Unknown')
                logger.info(f"🔍 Evaluating {model_name}...")

                try:
                    # 1. Basic evaluation
                    logger.debug(f"Running basic evaluation for {model_name}")
                    basic_metrics = {}
                    try:
                        if hasattr(model, 'evaluate'):
                            basic_metrics = model.evaluate(X_test, y_test)
                        else:
                            y_pred = model.predict(X_test)
                            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                            
                            basic_metrics = {
                                    'accuracy': float(accuracy_score(y_test, y_pred)),
                                    'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                                    'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                                    'f1': float(f1_score(y_test, y_pred, zero_division=0)),
                                    'roc_auc': float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None
                            }
                    except Exception as e:
                        logger.error(f"❌ Basic evaluation failed for {model_name}: {str(e)}")
                        basic_metrics = {'error': str(e)}

                    # 2. Spatial cross-validation
                    spatial_cv_results = None
                    if coordinates is not None:
                        try:
                            logger.debug(f"Running spatial CV for {model_name}")
                            spatial_cv_results = self.spatial_cv.cross_validate_spatial(
                                model, X_test, y_test, coordinates
                            )
                        except Exception as e:
                            logger.warning(f"⚠️ Spatial CV failed for {model_name}: {str(e)}")

                    # 3. Comprehensive evaluation
                    evaluation_results = {
                        'model_name': model_name, 
                        'basic_metrics': basic_metrics,
                        'spatial_cv': spatial_cv_results
                    }
                    
                    try:
                        evaluator = GeospatialModelEvaluator(model_name)
                        comprehensive_eval = evaluator.create_comprehensive_evaluation(
                            model, X_test, y_test, coordinates
                        )
                        evaluation_results.update(comprehensive_eval)
                    except Exception as e:
                        logger.warning(f"⚠️ Comprehensive evaluation failed for {model_name}: {str(e)}")

                    evaluations.append(evaluation_results)
                    logger.info(f"✅ Completed evaluation for {model_name}")

                except Exception as e:
                    logger.error(f"❌ Evaluation failed for {model_name}: {str(e)}")
                    continue

            # Create model comparison
            comparison = []
            for eval_result in evaluations:
                bm = eval_result.get('basic_metrics', {})
                if 'error' not in bm:
                    comparison.append({
                        'model': eval_result['model_name'],
                        'accuracy': bm.get('accuracy'),
                        'roc_auc': bm.get('roc_auc'),
                        'f1_score': bm.get('f1'),
                        'precision': bm.get('precision', 0),
                        'recall': bm.get('recall', 0)
                    })

            # Always a DataFrame (empty if no rows)
            comparison_df = pd.DataFrame(comparison, columns=['model','roc_auc','f1_score','accuracy','precision','recall'])

            # Optional: print comparison if available
            if comparison_df is not None and not comparison_df.empty:
                print("\n🏆 Model Comparison:")
                print(comparison_df[['model', 'roc_auc', 'f1_score', 'accuracy']].to_string(index=False))
                best_model_row = comparison_df.loc[comparison_df['roc_auc'].idxmax()]
                print(f"\n✨ Best model: {best_model_row['model']} (ROC-AUC: {best_model_row['roc_auc']:.4f})")

            return {
                'evaluations': evaluations,
                'comparison': comparison_df
            }

        except Exception as e:
            logger.error(f"❌ Model evaluation failed: {str(e)}")
            return {
                'evaluations': [],
                'comparison': None,
                'error': str(e)
            }
    

    def create_probability_maps(self, best_model: Any,
                              preprocessed_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Create probability maps using the best performing model.

        Args:
            best_model: Best performing trained model
            preprocessed_data: Preprocessed data

        Returns:
            Dictionary with paths to created maps
        """
        logger.info("🗺️ Creating probability maps...")

        # Generate predictions for mapping
        test_df = preprocessed_data['test_df']
        X_test = preprocessed_data['X_test']

        # Get model predictions
        predictions = best_model.predict(X_test)
        probabilities = best_model.predict_proba(X_test)

        # Create predictions DataFrame
        predictions_df = test_df.copy()
        predictions_df['predicted_class'] = predictions
        predictions_df['probability'] = probabilities

        # Define mapping bounds
        mapping_config = self.config.get('mapping', {})
        bounds = mapping_config.get('bounds')

        if bounds is None:
            # Use data bounds with buffer
            bounds = (
                test_df['lon'].min() - 0.1,
                test_df['lat'].min() - 0.1,
                test_df['lon'].max() + 0.1,
                test_df['lat'].max() + 0.1
            )

        try:
            # Create probability raster
            raster_data = self.probability_mapper.create_probability_raster(
                predictions_df[['lat', 'lon', 'probability']],
                bounds=bounds
            )

            # Save probability raster
            probability_raster_path = self.output_dir / "gold_probability_map.tif"
            saved_probability_path = self.probability_mapper.save_probability_raster(
                raster_data, str(probability_raster_path)
            )

            # Create and save probability map visualization
            map_plot_path = self.output_dir / "gold_probability_map.png"
            self.probability_mapper.plot_probability_map(
                raster_data,
                title=f"Gold Presence Probability - {best_model.model_name}",
                save_path=str(map_plot_path)
            )

            # Get raster statistics
            stats = self.probability_mapper.extract_raster_statistics(raster_data['probability'])

            # Save statistics
            stats_path = self.output_dir / "probability_map_statistics.json"
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)

            logger.info("✅ Probability maps created successfully")

            return {
                'probability_raster': saved_probability_path,
                'probability_map': str(map_plot_path),
                'statistics': str(stats_path)
            }

        except Exception as e:
            logger.error(f"❌ Probability mapping failed: {e}")
            raise

    def run_demo(self) -> Dict[str, str]:
        """
        Run complete pipeline with sample data.

        Returns:
            Dictionary with paths to generated outputs
        """
        logger.info("🎯 Running Phase 3 pipeline in demo mode...")

        try:
            # Create sample data
            sample_files = create_sample_training_data(str(self.output_dir / "sample_data"))

            # Preprocess data
            preprocessed_data = self.data_preprocessor.preprocess_complete_pipeline(
                sample_files['training_data']
            )

            # Train models
            trained_models = self.train_models(preprocessed_data)

            if not trained_models:
                raise ValueError("No models were successfully trained")

            # Evaluate models
            evaluation_results = self.evaluate_models(trained_models, preprocessed_data)

            # Get best model (highest ROC-AUC)
            evaluations = evaluation_results['evaluations']
            best_model_idx = np.argmax([eval['basic_metrics']['roc_auc'] for eval in evaluations])
            best_model = trained_models[best_model_idx]

            logger.info(f"🏆 Best model: {best_model.model_name} (AUC = {evaluations[best_model_idx]['basic_metrics']['roc_auc']:.3f}")

            # Create probability maps
            map_results = self.create_probability_maps(best_model, preprocessed_data)

            # Save summary
            summary = {
                'best_model': best_model.model_name,
                'best_model_auc': evaluations[best_model_idx]['basic_metrics']['roc_auc'],
                'models_trained': len(trained_models),
                'training_samples': len(preprocessed_data['X_train']),
                'test_samples': len(preprocessed_data['X_test']),
                **map_results
            }

            summary_path = self.output_dir / "phase3_demo_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info("✅ Demo completed successfully!")
            return summary

        except Exception as e:
            logger.error(f"❌ Demo execution failed: {e}")
            raise

    def run_pipeline(self) -> Dict[str, str]:
        """
        Run complete Phase 3 pipeline.

        Returns:
            Dictionary with paths to generated outputs
        """
        logger.info("🚀 Starting complete Phase 3 predictive modeling pipeline...")

        try:
            # Load and preprocess data
            preprocessed_data = self.load_and_preprocess_data()

            # Train models
            trained_models = self.train_models(preprocessed_data)

            if not trained_models:
                raise ValueError("No models were successfully trained")

            # Evaluate models
            evaluation_results = self.evaluate_models(trained_models, preprocessed_data)

            # Get best model
            evaluations = evaluation_results['evaluations']
            best_model_idx = np.argmax([eval['basic_metrics']['roc_auc'] for eval in evaluations])
            best_model = trained_models[best_model_idx]

            logger.info(f"🏆 Best model: {best_model.model_name} (AUC = {evaluations[best_model_idx]['basic_metrics']['roc_auc']:.3f}")

            # Create probability maps
            map_results = self.create_probability_maps(best_model, preprocessed_data)

            # Save final summary
            summary = {
                'best_model': best_model.model_name,
                'best_model_auc': evaluations[best_model_idx]['basic_metrics']['roc_auc'],
                'models_trained': len(trained_models),
                'training_samples': len(preprocessed_data['X_train']),
                'test_samples': len(preprocessed_data['X_test']),
                'spatial_cv_mean': evaluations[best_model_idx]['spatial_cv']['cv_mean'],
                **map_results
            }

            summary_path = self.output_dir / "phase3_results_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info("🎉 Phase 3 pipeline completed successfully!")
            return summary

        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {e}")
            raise


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


def create_default_config(output_dir: str = "phase3_output") -> Dict[str, Any]:
    """
    Create default configuration for Phase 3 pipeline.

    Args:
        output_dir: Output directory path

    Returns:
        Default configuration dictionary
    """
    return {
        'output_dir': output_dir,
        'random_state': 42,
        'test_size': 0.2,
        'spatial_split': True,
        'scaler_type': 'standard',
        'use_safe_scaling': True,  # Enable safe scaling that handles categorical columns
        'cv_method': 'geographic_blocks',
        'n_splits': 5,
        'block_size': 1.0,
        'interpolation_method': 'kriging',
        'pixel_size': 1000,
        'uncertainty_estimation': True,
        'data': {
            'training_data_path': None  # Will use sample data if None
        },
        'models': {
            'random_forest': True,
            'xgboost': True,
            'lightgbm': True,
            'rf_n_estimators': 100,
            'rf_max_depth': 10,
            'xgb_n_estimators': 100,
            'xgb_max_depth': 6,
            'xgb_learning_rate': 0.1,
            'lgb_n_estimators': 100,
            'lgb_max_depth': 6,
            'lgb_learning_rate': 0.1
        },
        'mapping': {
            'bounds': None,  # Will use data bounds if None
            'create_uncertainty_map': True
        }
    }


def main():
    """Main function to run the Phase 3 pipeline."""
    parser = argparse.ArgumentParser(description="GeoAuPredict Phase 3: Predictive Modeling")
    parser.add_argument('--config', '-c', type=str, help='Path to configuration YAML file')
    parser.add_argument('--demo', '-d', action='store_true', help='Run with sample data')
    parser.add_argument('--output', '-o', type=str, default='phase3_output', help='Output directory')

    args = parser.parse_args()

    try:
        if args.demo:
            # Run demo with sample data
            logger.info("🎯 Running Phase 3 pipeline in demo mode...")

            config = create_default_config(args.output)
            pipeline = Phase3Pipeline(config)
            results = pipeline.run_demo()

        else:
            # Run with provided configuration
            if args.config:
                config = load_config(args.config)
            else:
                logger.info("No config provided, using default configuration")
                config = create_default_config(args.output)

            pipeline = Phase3Pipeline(config)
            results = pipeline.run_pipeline()

        # Print summary
        logger.info("✅ Phase 3 pipeline completed successfully!")
        logger.info("📊 Results Summary:")
        for key, value in results.items():
            if key.endswith('_path') or key.endswith('_file'):
                logger.info(f"  {key}: {value}")
            else:
                logger.info(f"  {key}: {value}")

        logger.info(f"📁 All outputs saved to: {args.output}")

    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
