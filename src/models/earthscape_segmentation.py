"""
Phase 1: EarthScape-Style Segmentation Model
ResNeXt/UNet with multimodal inputs for geological feature segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import rasterio
from typing import Tuple, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class AttentionBlock(nn.Module):
    """Spatial and Channel Attention Block"""

    def __init__(self, in_channels: int, reduction: int = 16):
        super(AttentionBlock, self).__init__()

        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca

        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        sa_input = torch.cat([avg_pool, max_pool], dim=1)
        sa = self.spatial_attention(sa_input)
        x = x * sa

        return x


class ResNeXtBlock(nn.Module):
    """ResNeXt Block for feature extraction"""

    def __init__(self, in_channels: int, out_channels: int, cardinality: int = 32, width: int = 4):
        super(ResNeXtBlock, self).__init__()

        self.cardinality = cardinality
        self.width = width
        bottleneck_channels = cardinality * width

        # Bottleneck convolution
        self.conv_bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True)
        )

        # Grouped convolution (simplified ResNeXt)
        self.conv_grouped = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3,
                     padding=1, groups=cardinality, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True)
        )

        # Projection convolution
        self.conv_projection = nn.Sequential(
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Skip connection
        self.skip_connection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip_connection(x)

        x = self.conv_bottleneck(x)
        x = self.conv_grouped(x)
        x = self.conv_projection(x)

        return F.relu(x + residual)


class EarthScapeUNet(nn.Module):
    """
    UNet architecture with ResNeXt backbone for EarthScape-style segmentation
    """

    def __init__(self, in_channels: int = 10, num_classes: int = 6):
        super(EarthScapeUNet, self).__init__()

        # Encoder (ResNeXt-based)
        self.encoder1 = ResNeXtBlock(in_channels, 64)
        self.encoder2 = ResNeXtBlock(64, 128)
        self.encoder3 = ResNeXtBlock(128, 256)
        self.encoder4 = ResNeXtBlock(256, 512)

        # Bottleneck
        self.bottleneck = ResNeXtBlock(512, 1024)

        # Decoder
        self.decoder4 = ResNeXtBlock(1024 + 512, 512)
        self.decoder3 = ResNeXtBlock(512 + 256, 256)
        self.decoder2 = ResNeXtBlock(256 + 128, 128)
        self.decoder1 = ResNeXtBlock(128 + 64, 64)

        # Final classification
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        # Attention mechanisms
        self.attention3 = AttentionBlock(256)
        self.attention2 = AttentionBlock(128)
        self.attention1 = AttentionBlock(64)

        # Pooling for downsampling
        self.pool = nn.MaxPool2d(2, 2)

        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc1_pooled = self.pool(enc1)

        enc2 = self.encoder2(enc1_pooled)
        enc2_pooled = self.pool(enc2)

        enc3 = self.encoder3(enc2_pooled)
        enc3_pooled = self.pool(enc3)

        enc4 = self.encoder4(enc3_pooled)
        enc4_pooled = self.pool(enc4)

        # Bottleneck
        bottleneck = self.bottleneck(enc4_pooled)

        # Decoder with skip connections
        dec4 = self.upsample(bottleneck)
        dec4 = torch.cat([dec4, self.attention3(enc4)], dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upsample(dec4)
        dec3 = torch.cat([dec3, self.attention2(enc3)], dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upsample(dec3)
        dec2 = torch.cat([dec2, self.attention1(enc2)], dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upsample(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)

        # Final classification
        output = self.final_conv(dec1)

        return output


class EarthScapeDataset(Dataset):
    """Dataset class for EarthScape-style multimodal geological segmentation"""

    def __init__(self, data_dir: Path, tile_size: int = 256, transform=None):
        self.data_dir = Path(data_dir)
        self.tile_size = tile_size
        self.transform = transform

        # Multimodal input channels (following EarthScape)
        self.input_channels = [
            'rgb',           # RGB imagery (3 channels)
            'nir',           # Near-infrared (1 channel)
            'dem',           # Digital elevation model (1 channel)
            'slope',         # Slope (1 channel)
            'aspect',        # Aspect (1 channel)
            'curvature',     # Curvature (1 channel)
            'twi',           # Topographic wetness index (1 channel)
            'flow_accum',    # Flow accumulation (1 channel)
            'geology',       # Geological map (1 channel)
            'hydrology'      # Hydrology (1 channel)
        ]

        # Geological classes for segmentation
        self.geological_classes = [
            'igneous',      # Igneous rocks
            'sedimentary',  # Sedimentary rocks
            'metamorphic',  # Metamorphic rocks
            'unconsolidated', # Unconsolidated sediments
            'water',        # Water bodies
            'anthropogenic' # Human-modified areas
        ]

        self.samples = self._scan_data_directory()

    def _scan_data_directory(self) -> List[Dict]:
        """Scan data directory for available multimodal tiles"""
        samples = []

        # Look for tile directories
        tile_dirs = list(self.data_dir.glob("tile_*"))
        logger.info(f"Found {len(tile_dirs)} tile directories")

        for tile_dir in tile_dirs:
            sample = {'tile_dir': tile_dir}

            # Check for each required input
            for channel in self.input_channels:
                channel_file = tile_dir / f"{channel}.tif"
                if channel_file.exists():
                    sample[channel] = channel_file
                else:
                    logger.warning(f"Missing {channel} file in {tile_dir}")
                    break
            else:
                # Check for ground truth
                gt_file = tile_dir / "ground_truth.tif"
                if gt_file.exists():
                    sample['ground_truth'] = gt_file
                    samples.append(sample)
                else:
                    logger.warning(f"Missing ground truth in {tile_dir}")

        logger.info(f"Valid samples found: {len(samples)}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load multimodal tile and ground truth"""
        sample = self.samples[idx]

        # Load multimodal inputs
        inputs = []
        for channel in self.input_channels:
            if channel in sample:
                with rasterio.open(sample[channel]) as src:
                    # Read first band only for single-channel inputs
                    if channel in ['dem', 'slope', 'aspect', 'curvature', 'twi', 'flow_accum', 'geology', 'hydrology']:
                        data = src.read(1)
                    else:
                        # RGB and NIR (use first 4 bands)
                        data = src.read(1)  # Simplified for demo

                    # Normalize and convert to tensor
                    data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
                    inputs.append(data)

        # Concatenate all inputs
        input_tensor = torch.cat(inputs, dim=0)

        # Load ground truth
        with rasterio.open(sample['ground_truth']) as src:
            gt_data = src.read(1)
            gt_tensor = torch.tensor(gt_data, dtype=torch.long)

        if self.transform:
            input_tensor = self.transform(input_tensor)
            gt_tensor = self.transform(gt_tensor)

        return input_tensor, gt_tensor


def train_earthscape_model(
    model: EarthScapeUNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    device: str = "cuda"
) -> Dict:
    """
    Train the EarthScape-style segmentation model

    Args:
        model: EarthScapeUNet model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on

    Returns:
        Training history dictionary
    """
    logger.info(f"Starting EarthScape model training on {device}")

    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.nelement()
            train_correct += (predicted == targets).sum().item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.nelement()
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

        # Save best model
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), 'models/earthscape_segmentation_best.pth')

        # Update learning rate
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                       f"Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}, "
                       f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    logger.info("EarthScape model training completed")
    return history


def predict_geological_features(
    model: EarthScapeUNet,
    input_tiles: List[Path],
    output_dir: Path,
    device: str = "cuda"
) -> List[Path]:
    """
    Predict geological features using trained EarthScape model

    Args:
        model: Trained EarthScapeUNet model
        input_tiles: List of input tile paths
        output_dir: Directory to save predictions
        device: Device to run prediction on

    Returns:
        List of prediction file paths
    """
    logger.info(f"Running geological feature prediction on {len(input_tiles)} tiles")

    model = model.to(device)
    model.eval()

    output_paths = []

    with torch.no_grad():
        for tile_path in input_tiles:
            # Load input tile
            tile_dir = tile_path.parent
            sample = {'tile_dir': tile_dir}

            # Load all input channels
            inputs = []
            for channel in ['rgb', 'nir', 'dem', 'slope', 'aspect', 'curvature', 'twi', 'flow_accum', 'geology', 'hydrology']:
                channel_file = tile_dir / f"{channel}.tif"
                if channel_file.exists():
                    with rasterio.open(channel_file) as src:
                        data = src.read(1)
                        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
                        inputs.append(data)

            if len(inputs) == 10:  # All channels available
                input_tensor = torch.cat(inputs, dim=0).unsqueeze(0).to(device)

                # Run prediction
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)

                # Save prediction
                prediction = predicted.cpu().numpy()[0]

                # Save as GeoTIFF (maintaining input georeferencing)
                output_file = output_dir / f"geological_prediction_{tile_path.stem}.tif"

                # Copy georeferencing from input
                with rasterio.open(tile_dir / "rgb.tif") as template_src:
                    profile = template_src.profile.copy()
                    profile.update(count=1, dtype='int16')

                    with rasterio.open(output_file, 'w', **profile) as dst:
                        dst.write(prediction.astype('int16'), 1)

                output_paths.append(output_file)
                logger.info(f"Saved prediction: {output_file}")

    logger.info(f"Geological feature prediction completed. Generated {len(output_paths)} predictions")
    return output_paths
