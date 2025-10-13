#!/usr/bin/env python3
"""
Model Training and Persistence Script
Trains ensemble models and saves them to the model registry
"""

import pickle
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available")


def create_sample_data(n_samples=1000):
    """Create sample training data for demonstration"""
    np.random.seed(42)
    
    # Generate synthetic features
    X = np.random.randn(n_samples, 10)
    # Add some signal
    y = (X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
    
    return X, y


def train_and_save_models(X_train, X_test, y_train, y_test, output_dir='outputs/models'):
    """Train models and save to disk"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    models = {}
    results = {}
    
    # Train Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_rf = rf_model.predict(X_test)
    y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    
    results['random_forest'] = {
        'accuracy': float(accuracy_score(y_test, y_pred_rf)),
        'precision': float(precision_score(y_test, y_pred_rf)),
        'recall': float(recall_score(y_test, y_pred_rf)),
        'f1': float(f1_score(y_test, y_pred_rf)),
        'auc': float(roc_auc_score(y_test, y_pred_proba_rf))
    }
    
    # Save model
    rf_path = output_path / 'random_forest_model.pkl'
    with open(rf_path, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"✓ Saved Random Forest to {rf_path}")
    models['random_forest'] = rf_model
    
    # Train XGBoost if available
    if XGBOOST_AVAILABLE:
        print("Training XGBoost...")
        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        
        y_pred_xgb = xgb_model.predict(X_test)
        y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
        
        results['xgboost'] = {
            'accuracy': float(accuracy_score(y_test, y_pred_xgb)),
            'precision': float(precision_score(y_test, y_pred_xgb)),
            'recall': float(recall_score(y_test, y_pred_xgb)),
            'f1': float(f1_score(y_test, y_pred_xgb)),
            'auc': float(roc_auc_score(y_test, y_pred_proba_xgb))
        }
        
        xgb_path = output_path / 'xgboost_model.pkl'
        with open(xgb_path, 'wb') as f:
            pickle.dump(xgb_model, f)
        print(f"✓ Saved XGBoost to {xgb_path}")
        models['xgboost'] = xgb_model
    
    # Train LightGBM if available
    if LIGHTGBM_AVAILABLE:
        print("Training LightGBM...")
        lgbm_model = LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        lgbm_model.fit(X_train, y_train)
        
        y_pred_lgbm = lgbm_model.predict(X_test)
        y_pred_proba_lgbm = lgbm_model.predict_proba(X_test)[:, 1]
        
        results['lightgbm'] = {
            'accuracy': float(accuracy_score(y_test, y_pred_lgbm)),
            'precision': float(precision_score(y_test, y_pred_lgbm)),
            'recall': float(recall_score(y_test, y_pred_lgbm)),
            'f1': float(f1_score(y_test, y_pred_lgbm)),
            'auc': float(roc_auc_score(y_test, y_pred_proba_lgbm))
        }
        
        lgbm_path = output_path / 'lightgbm_model.pkl'
        with open(lgbm_path, 'wb') as f:
            pickle.dump(lgbm_model, f)
        print(f"✓ Saved LightGBM to {lgbm_path}")
        models['lightgbm'] = lgbm_model
    
    # Create ensemble (average predictions)
    if len(models) > 1:
        print("Creating ensemble model...")
        ensemble_path = output_path / 'ensemble_gold_v1.pkl'
        ensemble_data = {
            'models': models,
            'weights': {name: 1.0 / len(models) for name in models.keys()},
            'trained_at': datetime.now().isoformat(),
            'n_models': len(models)
        }
        with open(ensemble_path, 'wb') as f:
            pickle.dump(ensemble_data, f)
        print(f"✓ Saved Ensemble to {ensemble_path}")
        
        # Calculate ensemble metrics
        ensemble_proba = np.zeros_like(y_test, dtype=float)
        for model_name, model in models.items():
            ensemble_proba += model.predict_proba(X_test)[:, 1] / len(models)
        
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        results['ensemble'] = {
            'accuracy': float(accuracy_score(y_test, ensemble_pred)),
            'precision': float(precision_score(y_test, ensemble_pred)),
            'recall': float(recall_score(y_test, ensemble_pred)),
            'f1': float(f1_score(y_test, ensemble_pred)),
            'auc': float(roc_auc_score(y_test, ensemble_proba))
        }
    
    # Save results metadata
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'n_features': X_train.shape[1],
        'models': list(models.keys()),
        'results': results
    }
    
    metadata_path = output_path / 'model_registry_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {metadata_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL TRAINING SUMMARY")
    print("="*60)
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name:12s}: {value:.4f}")
    
    return models, results


def main():
    """Main training script"""
    print("🚀 GeoAuPredict Model Training Pipeline\n")
    
    # Create or load data
    print("📦 Preparing data...")
    X, y = create_sample_data(n_samples=1000)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test:  {len(X_test)} samples")
    print(f"   Features: {X.shape[1]}\n")
    
    # Train and save models
    models, results = train_and_save_models(X_train, X_test, y_train, y_test)
    
    print("\n✅ Model training and persistence complete!")
    print(f"   Models saved to: outputs/models/")
    
    return models, results


if __name__ == '__main__':
    main()

