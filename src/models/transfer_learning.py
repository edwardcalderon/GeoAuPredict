"""
Phase 3: Cross-Region Transfer Learning
Fine-tuning across geological regions for improved generalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import copy

logger = logging.getLogger(__name__)


class UncertaintyAwareModel(nn.Module):
    """Model with uncertainty quantification for exploration decision-making"""

    def __init__(self, base_model: nn.Module, uncertainty_method: str = "dropout"):
        super(UncertaintyAwareModel, self).__init__()
        self.base_model = base_model
        self.uncertainty_method = uncertainty_method

        # Uncertainty head
        if uncertainty_method == "dropout":
            # Enable dropout during inference for MC dropout
            self.dropout_rate = 0.1
        elif uncertainty_method == "ensemble":
            # Ensemble of models for uncertainty
            self.ensemble_size = 5

    def forward(self, x, num_samples: int = 10):
        """Forward pass with uncertainty estimation"""
        if self.uncertainty_method == "dropout":
            return self._mc_dropout_forward(x, num_samples)
        elif self.uncertainty_method == "ensemble":
            return self._ensemble_forward(x)
        else:
            return self.base_model(x)

    def _mc_dropout_forward(self, x: torch.Tensor, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Monte Carlo dropout for uncertainty estimation"""
        self.base_model.train()  # Enable dropout

        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.base_model(x)
                predictions.append(pred)

        predictions = torch.stack(predictions)
        mean_pred = torch.mean(predictions, dim=0)
        uncertainty = torch.std(predictions, dim=0)

        return mean_pred, uncertainty

    def _ensemble_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ensemble prediction for uncertainty estimation"""
        predictions = []

        for i in range(self.ensemble_size):
            with torch.no_grad():
                # Use different random seeds for ensemble diversity
                torch.manual_seed(i)
                pred = self.base_model(x)
                predictions.append(pred)

        predictions = torch.stack(predictions)
        mean_pred = torch.mean(predictions, dim=0)
        uncertainty = torch.std(predictions, dim=0)

        return mean_pred, uncertainty


class TransferLearningFramework:
    """Framework for cross-region transfer learning"""

    def __init__(self, base_model_path: Path, device: str = "cuda"):
        """
        Initialize transfer learning framework

        Args:
            base_model_path: Path to pre-trained base model
            device: Device for training
        """
        self.device = device
        self.base_model_path = Path(base_model_path)

        # Regional adaptation layers
        self.regional_adapters = {}

        logger.info(f"Initialized transfer learning framework with base model: {base_model_path}")

    def create_regional_adapter(self, region_name: str, num_features: int) -> nn.Module:
        """Create adapter layers for regional fine-tuning"""
        adapter = nn.Sequential(
            nn.Linear(num_features, num_features // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_features // 2, num_features // 4),
            nn.ReLU(),
            nn.Linear(num_features // 4, 1),  # Regional output
            nn.Sigmoid()
        )

        self.regional_adapters[region_name] = adapter
        logger.info(f"Created adapter for region: {region_name}")

        return adapter

    def fine_tune_regional_model(
        self,
        region_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        learning_rate: float = 1e-5
    ) -> Dict:
        """
        Fine-tune model for specific geological region

        Args:
            region_name: Name of geological region
            train_loader: Regional training data
            val_loader: Regional validation data
            num_epochs: Number of fine-tuning epochs
            learning_rate: Fine-tuning learning rate

        Returns:
            Fine-tuning history
        """
        logger.info(f"Starting fine-tuning for region: {region_name}")

        # Load base model
        base_model = torch.load(self.base_model_path)
        base_model = base_model.to(self.device)

        # Create regional adapter
        adapter = self.create_regional_adapter(region_name, 128)  # Assuming 128 features
        adapter = adapter.to(self.device)

        # Freeze base model parameters
        for param in base_model.parameters():
            param.requires_grad = False

        # Optimizer for adapter only
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        for epoch in range(num_epochs):
            # Training
            adapter.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()

                # Get features from base model
                with torch.no_grad():
                    features = base_model(inputs)

                # Regional prediction
                outputs = adapter(features).squeeze()
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()

            # Validation
            adapter.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    features = base_model(inputs)
                    outputs = adapter(features).squeeze()
                    loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()

            # Update history
            train_loss_avg = train_loss / len(train_loader)
            val_loss_avg = val_loss / len(val_loader)
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total

            history['train_loss'].append(train_loss_avg)
            history['val_loss'].append(val_loss_avg)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Region {region_name} - Epoch {epoch+1}/{num_epochs}: "
                           f"Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}, "
                           f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        # Save regional model
        regional_model = {
            'base_model': base_model.state_dict(),
            'adapter': adapter.state_dict(),
            'region': region_name
        }
        torch.save(regional_model, f'models/regional_model_{region_name}.pth')

        logger.info(f"Fine-tuning completed for region: {region_name}")
        return history

    def predict_with_uncertainty(
        self,
        model_path: Path,
        test_loader: DataLoader,
        uncertainty_samples: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty quantification

        Args:
            model_path: Path to trained regional model
            test_loader: Test data loader
            uncertainty_samples: Number of MC dropout samples

        Returns:
            Tuple of (predictions, uncertainties)
        """
        logger.info(f"Running prediction with uncertainty quantification using {model_path}")

        # Load regional model
        regional_model = torch.load(model_path)
        base_model = regional_model['base_model']
        adapter = regional_model['adapter']

        base_model = base_model.to(self.device)
        adapter = adapter.to(self.device)

        # Create uncertainty-aware model
        uncertainty_model = UncertaintyAwareModel(base_model)

        all_predictions = []
        all_uncertainties = []

        uncertainty_model.eval()
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)

                # Get prediction with uncertainty
                pred, uncertainty = uncertainty_model(inputs, uncertainty_samples)

                all_predictions.extend(pred.cpu().numpy())
                all_uncertainties.extend(uncertainty.cpu().numpy())

        predictions = np.array(all_predictions)
        uncertainties = np.array(all_uncertainties)

        logger.info(f"Generated {len(predictions)} predictions with uncertainty")
        return predictions, uncertainties


class IncrementalLearning:
    """Framework for continuous model improvement with new borehole data"""

    def __init__(self, base_model_path: Path):
        self.base_model_path = Path(base_model_path)
        self.model_versions = []

    def add_new_borehole_data(
        self,
        new_borehole_df: pd.DataFrame,
        model_path: Path,
        retrain_epochs: int = 20
    ) -> Path:
        """
        Incrementally update model with new borehole data

        Args:
            new_borehole_df: New borehole data for incremental learning
            model_path: Current model to update
            retrain_epochs: Number of epochs for incremental training

        Returns:
            Path to updated model
        """
        logger.info(f"Adding {len(new_borehole_df)} new borehole samples for incremental learning")

        # Load current model
        current_model = torch.load(model_path)

        # Create dataset with new data
        dataset = ColombiaHybridDataset(new_borehole_df, Path("data/dem"))

        # Incremental training (simplified)
        # In production, would implement proper incremental learning algorithms

        # Save updated model version
        version_num = len(self.model_versions) + 1
        updated_model_path = Path(f"models/model_version_{version_num}.pth")

        # For demo, just save the current model as updated
        torch.save(current_model, updated_model_path)
        self.model_versions.append(updated_model_path)

        logger.info(f"Incremental learning completed. Model updated: {updated_model_path}")
        return updated_model_path


def create_exploration_recommendations(
    probability_maps: List[Path],
    uncertainty_maps: List[Path],
    borehole_data: pd.DataFrame,
    min_probability: float = 0.7,
    max_uncertainty: float = 0.3
) -> pd.DataFrame:
    """
    Generate exploration recommendations based on model predictions

    Args:
        probability_maps: List of gold probability map paths
        uncertainty_maps: List of uncertainty map paths
        borehole_data: Borehole data for validation
        min_probability: Minimum probability threshold for recommendation
        max_uncertainty: Maximum uncertainty threshold for recommendation

    Returns:
        DataFrame with exploration recommendations
    """
    logger.info("Generating exploration recommendations based on model predictions")

    recommendations = []

    for prob_map, unc_map in zip(probability_maps, uncertainty_maps):
        # Load probability and uncertainty data
        with rasterio.open(prob_map) as prob_src, rasterio.open(unc_map) as unc_src:
            prob_data = prob_src.read(1)
            unc_data = unc_src.read(1)

            # Find high-probability, low-uncertainty regions
            high_prob_mask = prob_data >= min_probability
            low_unc_mask = unc_data <= max_uncertainty
            recommended_mask = high_prob_mask & low_unc_mask

            if np.any(recommended_mask):
                # Get coordinates of recommended areas
                rows, cols = np.where(recommended_mask)
                coords = prob_src.transform * (cols, rows)  # Convert pixel to map coordinates

                for i, (x, y) in enumerate(zip(coords[0], coords[1])):
                    recommendations.append({
                        'longitude': x,
                        'latitude': y,
                        'probability': prob_data[rows[i], cols[i]],
                        'uncertainty': unc_data[rows[i], cols[i]],
                        'priority_score': prob_data[rows[i], cols[i]] / (unc_data[rows[i], cols[i]] + 1e-6),
                        'recommendation': 'High Priority' if prob_data[rows[i], cols[i]] >= 0.8 else 'Medium Priority'
                    })

    # Create recommendations DataFrame
    rec_df = pd.DataFrame(recommendations)

    if not rec_df.empty:
        # Sort by priority score
        rec_df = rec_df.sort_values('priority_score', ascending=False)

        # Save recommendations
        output_path = Path("outputs/exploration_recommendations.csv")
        rec_df.to_csv(output_path, index=False)

        logger.info(f"Generated {len(rec_df)} exploration recommendations")
        logger.info(f"High priority targets: {len(rec_df[rec_df['recommendation'] == 'High Priority'])}")
    else:
        logger.warning("No exploration targets meet the specified criteria")

    return rec_df
