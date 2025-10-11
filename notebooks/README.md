# Notebooks

Exploratory Data Analysis (EDA) and reproducible experiments.

## Structure
- `01_eda_*.ipynb` - Exploratory data analysis notebooks
- `02_preprocessing_*.ipynb` - Data preprocessing experiments
- `03_modeling_*.ipynb` - Model training and evaluation
- `04_visualization_*.ipynb` - Visualization experiments

All notebooks are Colab-ready.

## Requirements and Setup

Before running any notebook, ensure you have the required dependencies installed:

### Quick Start
1. **Set up the conda environment:**
   ```bash
   conda env create -f ../environment.yml
   conda activate geoau
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r ../requirements.txt
   ```

3. **Verify your environment:**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import geopandas; print(f'GeoPandas: {geopandas.__version__}')"
   ```

### Detailed Setup Guide

See [NOTEBOOK_REQUIREMENTS.md](NOTEBOOK_REQUIREMENTS.md) for comprehensive setup instructions, troubleshooting, and environment verification.

### Hardware Requirements
- **Minimum**: 8GB RAM, 4-core CPU
- **Recommended**: 16GB+ RAM, GPU with CUDA support for model training
- **Storage**: 10GB+ for datasets and model outputs

### Environment Verification

Run the verification script to ensure all dependencies are properly installed:

```python
# Copy and run this in a Python cell
import sys
import torch
import numpy as np
import pandas as pd
import geopandas as gpd

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"GeoPandas: {gpd.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
```
