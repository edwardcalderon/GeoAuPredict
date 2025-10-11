# GeoAuPredict Notebook Requirements and Setup Guide
# This file provides detailed instructions for setting up the environment to run the GeoAuPredict notebook

## Quick Start (Conda Environment)
```bash
# 1. Create and activate the geoau environment
conda env create -f ../environment.yml
conda activate geoau

# 2. Install additional pip dependencies
pip install -r requirements.txt

# 3. Verify installation
python -c "import geopandas; print(f'GeoPandas: {geopandas.__version__}')"

# 4. Launch Jupyter
jupyter notebook
```

## Manual Installation (Alternative)
If you prefer manual installation or don't have conda:

### Core Dependencies
```bash
# Python 3.8-3.11 required
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn xgboost
pip install geopandas rasterio shapely pyproj matplotlib seaborn plotly
pip install jupyter notebook ipykernel fastapi uvicorn pydeck
```

### Geospatial Libraries (Conda Recommended)
```bash
# Install GDAL and geospatial libraries via conda (recommended)
conda install -c conda-forge gdal rasterio geopandas

# Alternative: Install individual packages
conda install -c conda-forge gdal
conda install -c conda-forge rasterio
conda install -c conda-forge geopandas
```

## Hardware Requirements
- **Minimum**: 8GB RAM, 4-core CPU
- **Recommended**: 16GB+ RAM, NVIDIA GPU with CUDA support
- **Storage**: 10GB+ for datasets and model outputs

## Troubleshooting

### Common Issues

**ModuleNotFoundError: No module named 'geopandas'**
```bash
conda install -c conda-forge geopandas
# or
pip install geopandas
```

**ModuleNotFoundError: No module named 'rasterio'**
```bash
conda install -c conda-forge rasterio
# or
pip install rasterio
```

**GDAL Installation Issues**
```bash
# Use conda instead of pip for GDAL (recommended)
conda install -c conda-forge gdal

# If you must use pip and get gdal-config errors:
sudo apt install gdal-bin libgdal-dev  # Ubuntu/Debian
# or
sudo yum install gdal-devel  # CentOS/RHEL

# Then try pip install again
pip install gdal
```

**CUDA Issues**
- Check CUDA version: `nvidia-smi`
- Install PyTorch with matching CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### Environment Verification
Run this script to verify your environment:

```python
import sys
import torch
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

def verify_environment():
    print("🔍 Environment Verification")
    print("=" * 40)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")

    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"GeoPandas version: {gpd.__version__}")

    # Test plotting
    try:
        plt.figure(figsize=(6, 4))
        plt.plot([1, 2, 3], [1, 4, 2])
        plt.title("Matplotlib Test")
        plt.close()
        print("✅ Matplotlib: Working")
    except Exception as e:
        print(f"❌ Matplotlib: {e}")

    print("\n✅ Environment verification complete!")

if __name__ == "__main__":
    verify_environment()
```

## Jupyter Kernel Setup
If using a custom conda environment:

```bash
# Install the kernel for Jupyter
conda activate geoau
python -m ipykernel install --user --name geoau --display-name "geoau"

# Verify kernel is available
jupyter kernelspec list
```

## Notebook Dependencies Summary

| Category | Libraries | Purpose |
|----------|-----------|---------|
| **Deep Learning** | PyTorch, TorchVision | Neural network models, computer vision |
| **Data Processing** | NumPy, Pandas, Scikit-learn | Data manipulation, ML algorithms |
| **Geospatial** | GeoPandas, Rasterio, GDAL* | Geographic data, DEM processing |
| **Visualization** | Matplotlib, Seaborn, Plotly | Plots, interactive maps |
| **Web Framework** | FastAPI, Uvicorn, PyDeck | 3D visualization dashboard |
| **Jupyter** | Notebook, IPyKernel | Interactive computing environment |

*GDAL installed via conda-forge

## Getting Help

If you encounter issues:
1. Check that you're in the correct conda environment (`conda env list`)
2. Verify PyTorch installation (`python -c "import torch"`)
3. Check CUDA compatibility if using GPU (`nvidia-smi`)
4. Review the error messages carefully - they usually indicate missing dependencies

For additional support, refer to:
- PyTorch documentation: https://pytorch.org/docs/
- GeoPandas documentation: https://geopandas.org/
- Rasterio documentation: https://rasterio.readthedocs.io/
