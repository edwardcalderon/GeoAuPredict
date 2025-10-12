#!/usr/bin/env python3
"""
GeoAuPredict Phase 3: Predictive Modeling - Model Training Framework

This module implements multiple machine learning algorithms for gold presence prediction:
- Traditional ML: Random Forest, XGBoost, LightGBM
- Deep Learning: CNN for raster data, LSTM for time series, GNN for spatial graphs
- Ensemble methods and stacking

Author: GeoAuPredict Team
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import xgboost as xgb
import lightgbm as lgb
import logging
logger = logging.getLogger(__name__)

# Deep Learning imports (optional)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Deep learning models will not be available.")


class GoldPredictionModel:
    """
    Base class for gold presence prediction models.
    """

    def __init__(self, model_name: str, random_state: int = 42):
        """
        Initialize the model.

        Args:
            model_name: Name of the model
            random_state: Random seed for reproducibility
        """
        self.model_name = model_name
        self.random_state = random_state
        self.model = None
        self.is_trained = False
        self.feature_importance_ = None

        logger.info(f"Initialized {model_name} model")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              **kwargs) -> Dict[str, float]:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional training parameters

        Returns:
            Dictionary with training metrics
        """
        raise NotImplementedError("Subclasses must implement train method")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predicted probabilities or classes
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features

        Returns:
            Predicted probabilities for each class
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]  # Return positive class probability
        else:
            # For models without predict_proba, use decision function
            if hasattr(self.model, 'decision_function'):
                decision_scores = self.model.decision_function(X)
                # Convert to probabilities using sigmoid
                return 1 / (1 + np.exp(-decision_scores))
            else:
                raise ValueError("Model does not support probability prediction")

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        # Get predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

        logger.info(f"{self.model_name} evaluation metrics: {metrics}")

        return metrics

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.

        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained or self.feature_importance_ is None:
            raise ValueError("Model must be trained and feature importance must be available")

        return self.feature_importance_


class RandomForestModel(GoldPredictionModel):
    """Random Forest model for gold prediction."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 10, **kwargs):
        super().__init__("RandomForest", **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              **kwargs) -> Dict[str, float]:
        """Train Random Forest model."""
        from sklearn.ensemble import RandomForestClassifier

        # Initialize model
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            **kwargs
        )

        # Train model
        self.model.fit(X_train, y_train)

        # Get feature importance
        importance = self.model.feature_importances_
        self.feature_importance_ = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)

        self.is_trained = True

        # Return training metrics using cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train,
                                  cv=5, scoring='roc_auc')

        return {
            'cv_mean_roc_auc': cv_scores.mean(),
            'cv_std_roc_auc': cv_scores.std(),
            'n_features': len(X_train.columns)
        }


class XGBoostModel(GoldPredictionModel):
    """XGBoost model for gold prediction."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 6,
                 learning_rate: float = 0.1, **kwargs):
        super().__init__("XGBoost", **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              **kwargs) -> Dict[str, float]:
        """Train XGBoost model."""
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=-1,
            enable_categorical=True,  # Enable categorical data support
            **kwargs
        )

        # Train model
        self.model.fit(X_train, y_train)

        # Get feature importance
        importance = self.model.feature_importances_
        self.feature_importance_ = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)

        self.is_trained = True

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train,
                                  cv=5, scoring='roc_auc')

        return {
            'cv_mean_roc_auc': cv_scores.mean(),
            'cv_std_roc_auc': cv_scores.std(),
            'n_features': len(X_train.columns)
        }


class LightGBMModel(GoldPredictionModel):
    """LightGBM model for gold prediction."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 6,
                 learning_rate: float = 0.1, **kwargs):
        super().__init__("LightGBM", **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              **kwargs) -> Dict[str, float]:
        """Train LightGBM model."""
        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=-1,
            **kwargs
        )

        # Train model
        self.model.fit(X_train, y_train)

        # Get feature importance
        importance = self.model.feature_importances_
        self.feature_importance_ = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)

        self.is_trained = True

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train,
                                  cv=5, scoring='roc_auc')

        return {
            'cv_mean_roc_auc': cv_scores.mean(),
            'cv_std_roc_auc': cv_scores.std(),
            'n_features': len(X_train.columns)
        }


if TORCH_AVAILABLE:
    class CNNModel(GoldPredictionModel):
        """CNN model for geospatial raster data."""

        def __init__(self, input_channels: int = 10, hidden_dim: int = 64,
                     num_classes: int = 2, **kwargs):
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch required for CNN model")

            super().__init__("CNN", **kwargs)
            self.input_channels = input_channels
            self.hidden_dim = hidden_dim
            self.num_classes = num_classes

        def _create_model(self) -> nn.Module:
            """Create CNN architecture."""
            class GeospatialCNN(nn.Module):
                def __init__(self, input_channels, hidden_dim, num_classes):
                    super().__init__()
                    self.conv1 = nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1)
                    self.conv2 = nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1)
                    self.conv3 = nn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=3, padding=1)
                    self.pool = nn.MaxPool2d(2, 2)
                    self.dropout = nn.Dropout(0.3)
                    self.fc1 = nn.Linear(hidden_dim*4 * 1 * 1, hidden_dim)
                    self.fc2 = nn.Linear(hidden_dim, num_classes)

                def forward(self, x):
                    x = self.pool(torch.relu(self.conv1(x)))
                    x = self.pool(torch.relu(self.conv2(x)))
                    x = self.pool(torch.relu(self.conv3(x)))
                    x = self.dropout(x)
                    x = x.view(-1, self.hidden_dim*4 * 1 * 1)
                    x = torch.relu(self.fc1(x))
                    x = self.dropout(x)
                    x = self.fc2(x)
                    return x

            return GeospatialCNN(self.input_channels, self.hidden_dim, self.num_classes)

        def train(self, X_train: pd.DataFrame, y_train: pd.Series,
                  **kwargs) -> Dict[str, float]:
            """Train CNN model."""
            # For CNN, we need to reshape data to image format
            # This is a simplified version - in practice, you'd need to create image patches

            logger.warning("CNN training not fully implemented - requires raster data preprocessing")
            return {'status': 'not_implemented'}


class EnsembleModel(GoldPredictionModel):
    """Ensemble model combining multiple base models."""

    def __init__(self, base_models: List[GoldPredictionModel], **kwargs):
        super().__init__("Ensemble", **kwargs)
        self.base_models = base_models

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              **kwargs) -> Dict[str, float]:
        """Train ensemble model."""
        # Train base models
        trained_models = []
        for model in self.base_models:
            model.train(X_train, y_train, **kwargs)
            trained_models.append((model.model_name, model.model))

        # Create voting classifier
        self.model = VotingClassifier(
            estimators=trained_models,
            voting='soft'  # Use probabilities for better performance
        )

        # Train ensemble
        self.model.fit(X_train, y_train)

        # Combine feature importance from base models
        importance_dfs = []
        for model in self.base_models:
            if model.feature_importance_ is not None:
                importance_dfs.append(model.feature_importance_)

        if importance_dfs:
            combined_importance = pd.concat(importance_dfs)
            self.feature_importance_ = combined_importance.groupby('feature')['importance'].mean().reset_index()
            self.feature_importance_ = self.feature_importance_.sort_values('importance', ascending=False)

        self.is_trained = True

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train,
                                  cv=5, scoring='roc_auc')

        return {
            'cv_mean_roc_auc': cv_scores.mean(),
            'cv_std_roc_auc': cv_scores.std(),
            'n_base_models': len(self.base_models)
        }


class ModelFactory:
    """Factory class for creating different model types."""

    @staticmethod
    def create_model(model_type: str, **kwargs) -> GoldPredictionModel:
        """
        Create a model instance.

        Args:
            model_type: Type of model to create
            **kwargs: Model-specific parameters

        Returns:
            Model instance
        """
        if model_type.lower() == 'randomforest':
            return RandomForestModel(**kwargs)
        elif model_type.lower() == 'xgboost':
            return XGBoostModel(**kwargs)
        elif model_type.lower() == 'lightgbm':
            return LightGBMModel(**kwargs)
        elif model_type.lower() == 'cnn':
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch required for CNN model")
            return CNNModel(**kwargs)
        elif model_type.lower() == 'ensemble':
            base_models = kwargs.get('base_models', [])
            return EnsembleModel(base_models, **{k: v for k, v in kwargs.items() if k != 'base_models'})
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def create_ensemble(models: List[GoldPredictionModel]) -> EnsembleModel:
        """Create ensemble from list of models."""
        return EnsembleModel(models)

    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available model types."""
        models = ['randomforest', 'xgboost', 'lightgbm', 'ensemble']
        if TORCH_AVAILABLE:
            models.append('cnn')
        return models


def hyperparameter_tuning(model: GoldPredictionModel,
                         X_train: pd.DataFrame,
                         y_train: pd.Series,
                         param_grid: Dict[str, List],
                         cv: int = 5,
                         scoring: str = 'roc_auc') -> Dict[str, Any]:
    """
    Perform hyperparameter tuning for a model.

    Args:
        model: Model instance
        X_train: Training features
        y_train: Training labels
        param_grid: Parameter grid for search
        cv: Number of cross-validation folds
        scoring: Scoring metric

    Returns:
        Dictionary with best parameters and scores
    """
    logger.info(f"Starting hyperparameter tuning for {model.model_name}")

    # Use RandomizedSearchCV for efficiency
    search = RandomizedSearchCV(
        model.model,
        param_distributions=param_grid,
        n_iter=50,
        cv=cv,
        scoring=scoring,
        random_state=model.random_state,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    logger.info(f"Best parameters: {search.best_params_}")
    logger.info(f"Best CV score: {search.best_score_:.4f}")

    # Update model with best parameters
    model.model = search.best_estimator_

    return {
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'cv_results': search.cv_results_
    }


def compare_models(models: List[GoldPredictionModel],
                  X_train: pd.DataFrame,
                  X_test: pd.DataFrame,
                  y_train: pd.Series,
                  y_test: pd.Series) -> pd.DataFrame:
    """
    Compare multiple models and return performance comparison.

    Args:
        models: List of trained models
        X_train, X_test: Train/test features
        y_train, y_test: Train/test labels

    Returns:
        DataFrame with model comparison results
    """
    logger.info("Comparing model performances")

    results = []

    for model in models:
        # Train model if not already trained
        if not model.is_trained:
            model.train(X_train, y_train)

        # Evaluate on test set
        metrics = model.evaluate(X_test, y_test)
        metrics['model_name'] = model.model_name

        results.append(metrics)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results).set_index('model_name')

    # Sort by ROC-AUC
    comparison_df = comparison_df.sort_values('roc_auc', ascending=False)

    logger.info("Model comparison completed")
    logger.info(f"\n{comparison_df.round(4)}")

    return comparison_df


# Example usage and testing functions
def create_sample_models() -> List[GoldPredictionModel]:
    """Create sample models for testing."""
    models = [
        RandomForestModel(n_estimators=50, max_depth=8),
        XGBoostModel(n_estimators=50, max_depth=4, learning_rate=0.1),
        LightGBMModel(n_estimators=50, max_depth=4, learning_rate=0.1)
    ]

    return models


if __name__ == "__main__":
    # Example usage
    logger.info("Testing model training framework...")

    # Create sample models
    models = create_sample_models()

    # This would be used in the main pipeline
    logger.info(f"Created {len(models)} sample models: {[m.model_name for m in models]}")
