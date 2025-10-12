# Streamlit Cloud Deployment Guide

## Files Required for Streamlit Cloud

### 1. requirements.txt (Dashboard-specific)

For Streamlit Cloud deployment, you should use a **lighter** requirements.txt. The main `requirements.txt` includes PyTorch which is too heavy for Streamlit Cloud.

**Option A: Use web_requirements.txt (Recommended)**

Rename or copy `web_requirements.txt` to `requirements.txt` for deployment:

```bash
# For deployment only
cp web_requirements.txt requirements.txt
```

**web_requirements.txt** contains:
```
streamlit>=1.28.0
dash>=2.14.0
plotly>=5.17.0
pandas>=1.5.0
numpy>=1.21.0
folium>=0.14.0
rasterio>=1.3.0
geopandas>=0.12.0
```

**Option B: Keep main requirements.txt and specify in Streamlit settings**

In your Streamlit Cloud app settings:
1. Go to Advanced settings
2. Set Python version to 3.10 or 3.11
3. Specify requirements file as `web_requirements.txt`

### 2. packages.txt (System Dependencies)

Already created! This file contains system packages needed for geospatial libraries:

```
gdal-bin
libgdal-dev
libspatialindex-dev
libproj-dev
libgeos-dev
```

### 3. .streamlit/config.toml

Already exists with proper configuration for embedding and dark theme.

## Deployment Steps

### Method 1: Streamlit Cloud (streamlit.io)

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Add Streamlit Cloud deployment files"
   git push
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to https://share.streamlit.io/
   - Click "New app"
   - Select your repository
   - Set main file path: `src/app/spatial_validation_dashboard.py`
   - Deploy!

3. **Advanced Settings (if needed):**
   - Python version: 3.10 or 3.11
   - Requirements file: `web_requirements.txt` (if not using default)

### Method 2: Docker Deployment

Create a `Dockerfile` for the dashboards:

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    libproj-dev \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY web_requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r web_requirements.txt

# Copy application code
COPY src/app/ ./src/app/
COPY data/ ./data/
COPY outputs/ ./outputs/

# Expose ports
EXPOSE 8501 8050

# Run Streamlit
CMD ["streamlit", "run", "src/app/spatial_validation_dashboard.py", "--server.port=8501", "--server.headless=true"]
```

### Method 3: Other Cloud Platforms

#### Render.com
1. Create a `render.yaml`:
```yaml
services:
  - type: web
    name: geoaupredict-streamlit
    env: python
    buildCommand: pip install -r web_requirements.txt
    startCommand: streamlit run src/app/spatial_validation_dashboard.py --server.port=$PORT --server.headless=true
```

#### Heroku
1. Create a `Procfile`:
```
web: streamlit run src/app/spatial_validation_dashboard.py --server.port=$PORT --server.headless=true
```

2. Create a `runtime.txt`:
```
python-3.10.12
```

## Troubleshooting

### Error: ModuleNotFoundError: No module named 'plotly'

**Solution:** Make sure your requirements file includes all dependencies:
- Check that `plotly>=5.17.0` is in requirements.txt
- Use `web_requirements.txt` instead of main `requirements.txt`
- Clear Streamlit Cloud cache and redeploy

### Error: GDAL/Geopandas installation fails

**Solution:** Ensure `packages.txt` is present with system dependencies:
```
gdal-bin
libgdal-dev
libspatialindex-dev
libproj-dev
libgeos-dev
```

### Error: App too large for Streamlit Cloud

**Solution:** 
- Use `web_requirements.txt` (without PyTorch)
- Remove heavy data files from deployment
- Use `.slugignore` (similar to `.gitignore`) to exclude:
  ```
  *.ipynb
  notebooks/
  test/
  venv/
  node_modules/
  ```

## Environment Variables

If you need to set environment variables on Streamlit Cloud:
1. Go to App settings
2. Secrets section
3. Add your secrets in TOML format

Example `.streamlit/secrets.toml` (don't commit this!):
```toml
# API keys or other secrets
API_KEY = "your-api-key"
```

## Production Configuration

The dashboards automatically detect production environment and skip dynamic package installation. They check for:
- `STREAMLIT_SHARING_MODE` env variable
- `STREAMLIT_CLOUD` env variable  
- Docker containers
- Kubernetes environments

## Data Files for Production

Make sure required data files are accessible:
- `outputs/spatial_validation_results.json` (sample data used as fallback)
- `data/processed/gold_dataset_master.geojson` (optional, sample data used if missing)

The dashboards will automatically fall back to sample data if these files don't exist.

## Next.js Integration

For full integration with the Next.js frontend at `localhost:3000/dashboards`:
1. Deploy the Streamlit and Dash apps separately
2. Update `NEXT_PUBLIC_STREAMLIT_URL` and `NEXT_PUBLIC_DASH_URL` in your Next.js environment
3. Deploy Next.js separately (Vercel, Netlify, etc.)

## Support

If you encounter issues:
1. Check Streamlit Cloud logs
2. Verify all dependencies are in requirements file
3. Ensure system packages are in packages.txt
4. Check that main app file path is correct: `src/app/spatial_validation_dashboard.py`

