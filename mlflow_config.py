#!/usr/bin/env python3
"""
MLflow Configuration and Tracking Setup
Tracks model training experiments, metrics, and model versions
"""

import mlflow
import mlflow.sklearn
from pathlib import Path
import os


class MLflowConfig:
    """MLflow tracking configuration"""
    
    def __init__(self, experiment_name="geoaupredict", tracking_uri="file:./mlruns"):
        """
        Initialize MLflow tracking
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: URI for MLflow tracking server
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create experiment if it doesn't exist
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
        mlflow.set_experiment(experiment_name)
    
    def start_run(self, run_name=None):
        """Start an MLflow run"""
        return mlflow.start_run(run_name=run_name)
    
    def log_params(self, params):
        """Log model parameters"""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics):
        """Log evaluation metrics"""
        mlflow.log_metrics(metrics)
    
    def log_model(self, model, artifact_path="model"):
        """Log trained model"""
        mlflow.sklearn.log_model(model, artifact_path)
    
    def log_artifact(self, file_path):
        """Log arbitrary file"""
        mlflow.log_artifact(file_path)


def track_model_training(model_name, model, params, metrics, model_path=None):
    """
    Track a model training run with MLflow
    
    Args:
        model_name: Name of the model (e.g., 'random_forest')
        model: Trained model object
        params: Dictionary of hyperparameters
        metrics: Dictionary of evaluation metrics
        model_path: Optional path to saved model file
    """
    mlflow_config = MLflowConfig()
    
    with mlflow_config.start_run(run_name=model_name):
        # Log parameters
        mlflow_config.log_params(params)
        
        # Log metrics
        mlflow_config.log_metrics(metrics)
        
        # Log model
        mlflow_config.log_model(model, artifact_path=model_name)
        
        # Log model file if provided
        if model_path and Path(model_path).exists():
            mlflow_config.log_artifact(model_path)
        
        # Log tags
        mlflow.set_tags({
            "model_type": model_name,
            "framework": "scikit-learn",
            "project": "GeoAuPredict"
        })
        
        print(f"✓ Logged {model_name} to MLflow")
        print(f"  Run ID: {mlflow.active_run().info.run_id}")
    
    return mlflow.active_run().info.run_id


# Example usage
if __name__ == "__main__":
    print("MLflow Configuration for GeoAuPredict")
    print("=" * 60)
    print(f"Tracking URI: file:./mlruns")
    print(f"Experiment: geoaupredict")
    print("\nTo start MLflow UI, run:")
    print("  mlflow ui")
    print("\nThen visit: http://localhost:5000")

