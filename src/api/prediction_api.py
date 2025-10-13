#!/usr/bin/env python3
"""
REST API for Gold Prediction
Serves trained models for batch predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pickle
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="GeoAuPredict API",
    description="REST API for gold deposit prediction using ensemble ML models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models on startup
MODELS_DIR = Path("outputs/models")
ENSEMBLE_MODEL = None


class PredictionInput(BaseModel):
    """Input schema for prediction requests"""
    features: List[List[float]] = Field(
        ...,
        description="List of feature vectors, each with 10 numeric values",
        example=[[0.5, -0.2, 1.3, 0.1, -0.5, 0.8, 0.3, -1.1, 0.6, 0.2]]
    )


class PredictionOutput(BaseModel):
    """Output schema for prediction responses"""
    predictions: List[int] = Field(..., description="Binary predictions (0 or 1)")
    probabilities: List[float] = Field(..., description="Probability of gold presence (0-1)")
    confidence: List[str] = Field(..., description="Confidence level (Low/Medium/High)")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    models_available: List[str]


@app.on_event("startup")
async def load_models():
    """Load trained models on application startup"""
    global ENSEMBLE_MODEL
    
    try:
        # Load Voting Ensemble (production model - winner of comparison)
        ensemble_path = MODELS_DIR / "ensemble_gold_v1.pkl"
        if ensemble_path.exists():
            with open(ensemble_path, 'rb') as f:
                ENSEMBLE_MODEL = pickle.load(f)
            
            ensemble_type = ENSEMBLE_MODEL.get('ensemble_type', 'voting')
            n_models = ENSEMBLE_MODEL.get('n_models', len(ENSEMBLE_MODEL.get('models', {})))
            
            logger.info(f"✓ Loaded {ensemble_type.upper()} ensemble with {n_models} base models")
            logger.info(f"  Base models: {list(ENSEMBLE_MODEL.get('models', {}).keys())}")
        else:
            logger.warning(f"Ensemble model not found at {ensemble_path}")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        ENSEMBLE_MODEL = None


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "GeoAuPredict API - Gold Deposit Prediction",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Make predictions (POST)",
            "/ensemble-info": "Ensemble model information",
            "/docs": "Interactive API documentation"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    models_available = []
    
    if MODELS_DIR.exists():
        models_available = [f.stem for f in MODELS_DIR.glob("*.pkl")]
    
    return HealthResponse(
        status="healthy" if ENSEMBLE_MODEL is not None else "degraded",
        model_loaded=ENSEMBLE_MODEL is not None,
        models_available=models_available
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Make predictions for gold deposit presence
    
    Args:
        input_data: List of feature vectors
    
    Returns:
        Predictions with probabilities and confidence levels
    """
    if ENSEMBLE_MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to numpy array
        X = np.array(input_data.features)
        
        # Validate input shape
        if X.shape[1] != 10:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 10 features per sample, got {X.shape[1]}"
            )
        
        # Make predictions with ensemble
        models = ENSEMBLE_MODEL['models']
        weights = ENSEMBLE_MODEL['weights']
        
        # Average predictions from all models
        ensemble_proba = np.zeros(len(X))
        for model_name, model in models.items():
            weight = weights[model_name]
            proba = model.predict_proba(X)[:, 1]
            ensemble_proba += proba * weight
        
        # Binary predictions (threshold = 0.5)
        predictions = (ensemble_proba > 0.5).astype(int).tolist()
        probabilities = ensemble_proba.tolist()
        
        # Confidence levels
        confidence = []
        for prob in probabilities:
            if prob < 0.3 or prob > 0.7:
                confidence.append("High")
            elif prob < 0.4 or prob > 0.6:
                confidence.append("Medium")
            else:
                confidence.append("Low")
        
        return PredictionOutput(
            predictions=predictions,
            probabilities=probabilities,
            confidence=confidence
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/info")
async def models_info():
    """Get information about loaded models"""
    if ENSEMBLE_MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "ensemble": {
            "n_models": ENSEMBLE_MODEL['n_models'],
            "models": list(ENSEMBLE_MODEL['models'].keys()),
            "weights": ENSEMBLE_MODEL['weights'],
            "trained_at": ENSEMBLE_MODEL.get('trained_at', 'Unknown')
        }
    }


@app.get("/ensemble-info")
async def ensemble_info():
    """Get detailed information about the ensemble model"""
    if ENSEMBLE_MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    ensemble_type = ENSEMBLE_MODEL.get('ensemble_type', 'voting')
    
    info = {
        "ensemble_type": ensemble_type,
        "production_model": "Voting Ensemble (Winner)",
        "n_base_models": ENSEMBLE_MODEL.get('n_models', len(ENSEMBLE_MODEL.get('models', {}))),
        "base_models": list(ENSEMBLE_MODEL.get('models', {}).keys()),
        "trained_at": ENSEMBLE_MODEL.get('trained_at', 'Unknown'),
        "description": {
            "voting": "Simple averaging of predictions from all base models with equal weights (33.3% each)",
            "stacking": "Logistic Regression meta-model learns optimal combination of base model predictions"
        }.get(ensemble_type, "Unknown ensemble type"),
        "comparison_results": {
            "tested_against": "Stacking Ensemble",
            "winner": "Voting Ensemble",
            "auc_improvement": "0.02% better than stacking",
            "reason": "Simpler, more robust, better generalization"
        }
    }
    
    if ensemble_type == 'voting':
        info["weights"] = ENSEMBLE_MODEL.get('weights', {})
        info["weight_type"] = "Equal (no learning required)"
    
    return info


if __name__ == "__main__":
    import uvicorn
    import os
    
    # Use PORT environment variable (Render provides this)
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting API server on port {port}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)

