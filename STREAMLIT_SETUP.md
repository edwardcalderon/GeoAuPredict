# Streamlit Cloud Deployment Setup

## Files Configuration

### `requirements.txt` (ACTIVE for Streamlit)
- **Purpose**: Minimal dependencies for Streamlit Cloud deployment
- **Contents**: 15 essential packages with pinned geospatial versions
- **Installation**: Uses pip with pre-built wheels (fast!)

### `requirements_full.txt` (For Local Development)
- **Purpose**: Complete development environment with all optional packages
- **Contents**: 40+ packages including ML, deep learning, testing, etc.
- **Usage**: `pip install -r requirements_full.txt`

### `environment_dev.yml` (For Local Development)
- **Purpose**: Conda environment for local development
- **Original name**: `environment.yml` (renamed to avoid Streamlit using conda)
- **Usage**: `conda env create -f environment_dev.yml`
- **Why renamed**: Streamlit Cloud prioritizes conda over pip, causing slow/stuck deployments

### `packages.txt` (REMOVED)
- **Status**: Deleted to avoid ODBC dependency conflicts
- **Alternative**: Using Python wheels with bundled libraries instead

## Deployment Strategy

1. **Streamlit Cloud** uses:
   - ✅ `requirements.txt` (pip-based, fast)
   - ❌ No `packages.txt` (avoids system dependency conflicts)
   - ❌ No `environment.yml` (renamed to prevent conda usage)

2. **Local Development** can use:
   - `requirements_full.txt` for pip installation
   - `environment_dev.yml` for conda environment

## Geospatial Packages

Pinned to specific versions with pre-built wheels:
- `shapely==2.0.2`
- `pyproj==3.6.1`
- `Fiona==1.9.5`
- `geopandas==0.14.1`
- `rasterio==1.3.9`

These versions have binary wheels that don't require system libraries (GDAL, GEOS, etc.).

## Troubleshooting

### If deployment fails with import errors:
1. Check if a newer wheel version is available
2. May need to add back minimal `packages.txt` with just system essentials

### If deployment is slow/stuck:
1. Ensure no `environment.yml` exists in root
2. Keep `requirements.txt` minimal
3. Use pinned versions to leverage pip cache

