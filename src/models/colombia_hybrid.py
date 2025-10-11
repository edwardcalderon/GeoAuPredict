"""
Phase 2: Colombia Hybrid Model
DEM + Borehole Depth CNN with Positional Encoding for subsurface prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import rasterio
from typing import Tuple, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional Encoding for spatial context in borehole data"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class DepthAwareCNN(nn.Module):
    """CNN architecture adapted for borehole depth information"""

    def __init__(self, input_channels: int = 3, hidden_dim: int = 128):
        super(DepthAwareCNN, self).__init__()

        # Feature extraction layers
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Depth-aware attention
        self.depth_attention = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.Sigmoid()
        )

        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),  # Binary classification (gold presence)
            nn.Sigmoid()
        )

        # Positional encoding for spatial context
        self.positional_encoding = PositionalEncoding(hidden_dim)

    def forward(self, x, depth_info=None):
        """
        Forward pass with optional depth information

        Args:
            x: Input DEM tensor (batch_size, channels, height, width)
            depth_info: Optional depth information tensor (batch_size, 1)
        """
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))

        # Apply depth-aware attention if depth info available
        if depth_info is not None:
            depth_weights = self.depth_attention(depth_info)
            x = x * depth_weights.unsqueeze(-1).unsqueeze(-1)

        # Global pooling
        x = self.global_pool(x).squeeze(-1).squeeze(-1)

        # Apply positional encoding for spatial context
        x = self.positional_encoding(x.unsqueeze(0)).squeeze(0)

        # Classification
        output = self.classifier(x)

        return output


class ColombiaHybridDataset(Dataset):
    """Dataset for Colombia hybrid model (DEM + borehole data)"""

    def __init__(self, borehole_df: pd.DataFrame, dem_dir: Path, tile_size: int = 64):
        """
        Initialize dataset with borehole data and DEM directory

        Args:
            borehole_df: DataFrame with borehole information
            dem_dir: Directory containing DEM tiles
            tile_size: Size of DEM tiles to extract
        """
        self.borehole_df = borehole_df.copy()
        self.dem_dir = Path(dem_dir)
        self.tile_size = tile_size

        # Filter boreholes with valid coordinates
        self.borehole_df = self.borehole_df.dropna(subset=['lon', 'lat'])

        logger.info(f"Initialized Colombia dataset with {len(self.borehole_df)} boreholes")

    def __len__(self) -> int:
        return len(self.borehole_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Get DEM tile, borehole data, and metadata"""
        borehole = self.borehole_df.iloc[idx]

        # Extract DEM tile around borehole location
        dem_tile = self._extract_dem_tile(borehole['lon'], borehole['lat'])

        # Get target (gold presence)
        target = torch.tensor(borehole['label_gold'], dtype=torch.float32)

        # Get depth information
        depth = torch.tensor(borehole.get('depth_m', 0.0), dtype=torch.float32).unsqueeze(0)

        # Metadata for analysis
        metadata = {
            'borehole_id': idx,
            'coordinates': (borehole['lon'], borehole['lat']),
            'lithology_text': borehole.get('lithology_text', ''),
            'au_ppm': borehole.get('au_ppm', 0.0)
        }

        return dem_tile, target, depth, metadata

    def _extract_dem_tile(self, lon: float, lat: float) -> torch.Tensor:
        """Extract DEM tile around borehole coordinates"""
        # For demo purposes, create synthetic DEM data
        # In production, this would extract actual DEM tiles
        tile = torch.randn(3, self.tile_size, self.tile_size)  # 3-channel DEM data

        # Normalize to reasonable elevation range
        tile = torch.clamp(tile, -100, 5000)

        return tile


class ColombiaHybridModel(nn.Module):
    """Hybrid CNN model for Colombia borehole + DEM integration"""

    def __init__(self, dem_channels: int = 3, hidden_dim: int = 128):
        super(ColombiaHybridModel, self).__init__()

        self.dem_cnn = DepthAwareCNN(dem_channels, hidden_dim)

        # Borehole-specific layers
        self.borehole_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim // 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, dem_data, borehole_features, depth_info=None):
        """
        Forward pass combining DEM and borehole data

        Args:
            dem_data: DEM tensor (batch_size, channels, height, width)
            borehole_features: Borehole feature tensor (batch_size, features)
            depth_info: Depth information (batch_size, 1)
        """
        # Process DEM data
        dem_features = self.dem_cnn(dem_data, depth_info)

        # Process borehole features
        borehole_encoded = self.borehole_encoder(borehole_features)

        # Fuse DEM and borehole features
        combined_features = torch.cat([dem_features, borehole_encoded], dim=1)
        output = self.fusion(combined_features)

        return output


def train_colombia_hybrid_model(
    model: ColombiaHybridModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    device: str = "cuda"
) -> Dict:
    """
    Train the Colombia hybrid model

    Args:
        model: ColombiaHybridModel
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: Training device

    Returns:
        Training history
    """
    logger.info(f"Starting Colombia hybrid model training on {device}")

    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.BCELoss()  # Binary cross-entropy for gold presence
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_f1': [],
        'val_f1': []
    }

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_tp = 0  # True positives
        train_fp = 0  # False positives
        train_fn = 0  # False negatives

        for dem_data, targets, depth_info, _ in train_loader:
            dem_data = dem_data.to(device)
            targets = targets.to(device)
            depth_info = depth_info.to(device)

            optimizer.zero_grad()

            # For demo, use random borehole features
            batch_size = dem_data.size(0)
            borehole_features = torch.randn(batch_size, 128).to(device)

            outputs = model(dem_data, borehole_features, depth_info)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calculate accuracy
            predicted = (outputs.squeeze() > 0.5).float()
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

            # Calculate F1 components
            train_tp += ((predicted == 1) & (targets == 1)).sum().item()
            train_fp += ((predicted == 1) & (targets == 0)).sum().item()
            train_fn += ((predicted == 0) & (targets == 1)).sum().item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_tp = 0
        val_fp = 0
        val_fn = 0

        with torch.no_grad():
            for dem_data, targets, depth_info, _ in val_loader:
                dem_data = dem_data.to(device)
                targets = targets.to(device)
                depth_info = depth_info.to(device)

                batch_size = dem_data.size(0)
                borehole_features = torch.randn(batch_size, 128).to(device)

                outputs = model(dem_data, borehole_features, depth_info)
                loss = criterion(outputs.squeeze(), targets)

                val_loss += loss.item()

                predicted = (outputs.squeeze() > 0.5).float()
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

                val_tp += ((predicted == 1) & (targets == 1)).sum().item()
                val_fp += ((predicted == 1) & (targets == 0)).sum().item()
                val_fn += ((predicted == 0) & (targets == 1)).sum().item()

        # Calculate metrics
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        # F1 Score
        train_precision = train_tp / (train_tp + train_fp) if (train_tp + train_fp) > 0 else 0
        train_recall = train_tp / (train_tp + train_fn) if (train_tp + train_fn) > 0 else 0
        train_f1 = 2 * train_precision * train_recall / (train_precision + train_recall) if (train_precision + train_recall) > 0 else 0

        val_precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0
        val_recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
        val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0

        # Update history
        history['train_loss'].append(train_loss_avg)
        history['val_loss'].append(val_loss_avg)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        # Save best model
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), 'models/colombia_hybrid_best.pth')

        # Update learning rate
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                       f"Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}, "
                       f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, "
                       f"Train F1: {train_f1:.3f}, Val F1: {val_f1:.3f}")

    logger.info("Colombia hybrid model training completed")
    return history


def predict_subsurface_gold(
    model: ColombiaHybridModel,
    dem_tiles: List[Path],
    borehole_data: pd.DataFrame,
    output_dir: Path,
    device: str = "cuda"
) -> List[Path]:
    """
    Predict subsurface gold occurrence probabilities

    Args:
        model: Trained ColombiaHybridModel
        dem_tiles: List of DEM tile paths
        borehole_data: DataFrame with borehole information
        output_dir: Directory to save predictions
        device: Device for prediction

    Returns:
        List of prediction file paths
    """
    logger.info(f"Running subsurface gold prediction on {len(dem_tiles)} tiles")

    model = model.to(device)
    model.eval()

    output_paths = []

    with torch.no_grad():
        for tile_path in dem_tiles:
            # Load DEM tile
            with rasterio.open(tile_path) as src:
                dem_data = src.read()  # Read all bands
                dem_tensor = torch.tensor(dem_data, dtype=torch.float32).unsqueeze(0)

                # For demo, use random borehole features
                borehole_features = torch.randn(1, 128).to(device)

                # Run prediction
                dem_tensor = dem_tensor.to(device)
                depth_info = torch.randn(1, 1).to(device)  # Placeholder depth

                output = model(dem_tensor, borehole_features, depth_info)
                probability = output.squeeze().cpu().numpy()

                # Save probability map
                output_file = output_dir / f"gold_probability_{tile_path.stem}.tif"

                # Update profile for single-band probability output
                profile = src.profile.copy()
                profile.update(count=1, dtype='float32')

                with rasterio.open(output_file, 'w', **profile) as dst:
                    dst.write(probability.astype('float32'), 1)

                output_paths.append(output_file)
                logger.info(f"Saved prediction: {output_file} (Probability: {probability:.3f})")

    logger.info(f"Subsurface gold prediction completed. Generated {len(output_paths)} predictions")
    return output_paths


def create_borehole_cross_sections(
    borehole_df: pd.DataFrame,
    output_dir: Path,
    max_depth: float = 50.0
) -> List[Path]:
    """
    Create 3D borehole cross-section visualizations

    Args:
        borehole_df: DataFrame with borehole data
        output_dir: Directory to save cross-sections
        max_depth: Maximum depth for visualization

    Returns:
        List of cross-section file paths
    """
    logger.info(f"Creating borehole cross-sections for {len(borehole_df)} boreholes")

    output_paths = []

    # Group boreholes by study area for cross-section creation
    study_areas = borehole_df['study_area'].unique()

    for area in study_areas:
        area_boreholes = borehole_df[borehole_df['study_area'] == area]

        # Create cross-section data
        cross_section_data = {
            'borehole_id': area_boreholes.index.tolist(),
            'longitude': area_boreholes['lon'].tolist(),
            'latitude': area_boreholes['lat'].tolist(),
            'depth_m': area_boreholes.get('depth_m', [0] * len(area_boreholes)).tolist(),
            'au_ppm': area_boreholes.get('au_ppm', [0] * len(area_boreholes)).tolist(),
            'lithology_text': area_boreholes.get('lithology_text', ['Unknown'] * len(area_boreholes)).tolist()
        }

        # Save cross-section data as JSON for 3D visualization
        import json
        output_file = output_dir / f"borehole_cross_section_{area.replace(' ', '_')}.json"

        with open(output_file, 'w') as f:
            json.dump(cross_section_data, f, indent=2)

        output_paths.append(output_file)
        logger.info(f"Created cross-section: {output_file}")

    logger.info(f"Borehole cross-section creation completed. Generated {len(output_paths)} visualizations")
    return output_paths
