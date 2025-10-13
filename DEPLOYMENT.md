# 🚀 GeoAuPredict Deployment Guide

Complete guide for deploying GeoAuPredict models and dashboards.

## 📋 Overview

The project provides multiple deployment options:

1. **REST API** - FastAPI service for batch predictions
2. **Streamlit Dashboards** - Interactive visualization dashboards
3. **MLflow** - Model tracking and versioning
4. **Docker** - Containerized deployment
5. **GitHub Pages** - Static website with documentation

---

## 🐳 Docker Deployment (Recommended)

### Quick Start

```bash
# Build and start all services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f
```

### Available Services

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| **REST API** | 8000 | http://localhost:8000 | Prediction API |
| **Spatial Dashboard** | 8501 | http://localhost:8501 | Spatial validation |
| **3D Viz Dashboard** | 8502 | http://localhost:8502 | 3D visualization |
| **MLflow UI** | 5000 | http://localhost:5000 | Model tracking |

### Stop Services

```bash
docker compose down
```

---

## 🔌 REST API

### Start API

```bash
# With uvicorn directly
uvicorn src.api.prediction_api:app --host 0.0.0.0 --port 8000

# Or with python
python src/api/prediction_api.py
```

### API Documentation

- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Example Usage

```bash
# Health check
curl http://localhost:8000/health

# Make predictions
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      [0.5, -0.2, 1.3, 0.1, -0.5, 0.8, 0.3, -1.1, 0.6, 0.2]
    ]
  }'
```

### Python Client Example

```python
import requests

# API endpoint
url = "http://localhost:8000/predict"

# Sample features
data = {
    "features": [
        [0.5, -0.2, 1.3, 0.1, -0.5, 0.8, 0.3, -1.1, 0.6, 0.2],
        [-0.1, 0.8, 0.5, -0.3, 1.2, -0.6, 0.9, 0.4, -0.7, 0.3]
    ]
}

# Make request
response = requests.post(url, json=data)
predictions = response.json()

print(predictions)
# {
#   "predictions": [1, 0],
#   "probabilities": [0.75, 0.32],
#   "confidence": ["High", "Medium"]
# }
```

---

## 📊 Streamlit Dashboards

### Start Dashboards

```bash
# Spatial validation dashboard
streamlit run src/app/spatial_validation_dashboard.py

# 3D visualization dashboard
streamlit run src/app/3d_visualization_dashboard.py
```

### Features

- **Interactive Maps**: Explore predictions geographically
- **Model Performance**: View evaluation metrics
- **Feature Importance**: Understand model decisions
- **3D Visualization**: View terrain and probability surfaces

---

## 📈 MLflow Tracking

### Start MLflow UI

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Visit: http://localhost:5000

### Track Model Training

```python
from mlflow_config import track_model_training

# After training a model
track_model_training(
    model_name="random_forest",
    model=rf_model,
    params={"n_estimators": 100, "max_depth": 10},
    metrics={"accuracy": 0.85, "auc": 0.87},
    model_path="outputs/models/random_forest_model.pkl"
)
```

### Features

- **Experiment Tracking**: Compare model runs
- **Parameter Logging**: Track hyperparameters
- **Metric Visualization**: Plot performance over time
- **Model Registry**: Version and deploy models
- **Artifact Storage**: Save models, plots, data

---

## 🌐 GitHub Pages Deployment

The static website is automatically deployed to GitHub Pages on push to `main`.

### Manual Deployment

```bash
# Build Next.js static export
npm run build
npm run export

# Deploy to GitHub Pages
npm run deploy
```

### Configuration

Update `.github/workflows/nextjs.yml` with your repository details.

---

## 📦 Model Files

### Location

All trained models are stored in:
```
outputs/models/
├── ensemble_gold_v1.pkl          # Main ensemble model
├── random_forest_model.pkl       # Random Forest
├── xgboost_model.pkl            # XGBoost
├── lightgbm_model.pkl           # LightGBM
└── model_registry_metadata.json # Metadata
```

### (Re)Train Models

```bash
python src/models/model_training.py
```

This creates/updates all model files with:
- Trained model objects (`.pkl`)
- Performance metrics
- Training metadata

---

## 🔧 Environment Setup

### Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_full.txt
```

### Required Packages

- `fastapi` - REST API framework
- `uvicorn` - ASGI server
- `streamlit` - Dashboard framework
- `mlflow` - Model tracking
- `scikit-learn` - ML models
- `xgboost` - Gradient boosting
- `lightgbm` - Gradient boosting
- `geopandas` - Geospatial data
- `plotly` - Interactive plots

---

## 🧪 Testing

### Test API

```bash
# Run API tests
pytest tests/test_api.py -v

# Test with curl
curl http://localhost:8000/health
```

### Test Models

```bash
# Run model tests
pytest tests/test_models.py -v
```

---

## 📝 Deployment Checklist

- [ ] Trained models exist in `outputs/models/`
- [ ] REST API starts successfully
- [ ] Streamlit dashboards load
- [ ] MLflow UI accessible
- [ ] Docker containers build
- [ ] GitHub Pages deployed
- [ ] Environment variables configured
- [ ] Logs directory writable

---

## 🔒 Security Considerations

### Production Deployment

1. **API Authentication**: Add API keys or OAuth
2. **HTTPS**: Use SSL certificates
3. **Rate Limiting**: Prevent abuse
4. **Input Validation**: Sanitize user inputs
5. **Model Versioning**: Use MLflow model registry
6. **Monitoring**: Set up logging and alerts

### Example: Add API Key Authentication

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY = "your-secret-api-key"
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

@app.post("/predict")
async def predict(
    input_data: PredictionInput,
    api_key: str = Security(verify_api_key)
):
    # ...
```

---

## 📞 Support

For deployment issues:
- Check logs: `docker-compose logs` or `streamlit run --logger.level=debug`
- Review [GitHub Issues](https://github.com/edwardcalderon/GeoAuPredict/issues)
- Consult [documentation](https://edwardcalderon.github.io/GeoAuPredict/)

---

**Ready for production!** 🎉

