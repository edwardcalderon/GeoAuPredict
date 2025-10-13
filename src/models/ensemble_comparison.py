#!/usr/bin/env python3
"""
Ensemble Model Comparison: Voting vs Stacking
Trains both ensemble types and provides detailed performance comparison
"""

import pickle
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, classification_report, 
    confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    # Generate synthetic features with geological patterns
    X = np.random.randn(n_samples, 10)
    # Add some signal mimicking geological features
    y = (X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + 
         0.2 * X[:, 3] * X[:, 4] + 
         np.random.randn(n_samples) * 0.5 > 0).astype(int)
    
    return X, y


def train_base_models(X_train, y_train):
    """Train individual base models"""
    print("\n" + "="*60)
    print("TRAINING BASE MODELS")
    print("="*60)
    
    models = {}
    
    # Train Random Forest
    print("\n1. Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    models['random_forest'] = rf_model
    print("   ✓ Random Forest trained")
    
    # Train XGBoost if available
    if XGBOOST_AVAILABLE:
        print("\n2. Training XGBoost...")
        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        models['xgboost'] = xgb_model
        print("   ✓ XGBoost trained")
    
    # Train LightGBM if available
    if LIGHTGBM_AVAILABLE:
        print("\n3. Training LightGBM...")
        lgbm_model = LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        lgbm_model.fit(X_train, y_train)
        models['lightgbm'] = lgbm_model
        print("   ✓ LightGBM trained")
    
    return models


def create_voting_ensemble(models):
    """Create voting ensemble (simple averaging)"""
    print("\n" + "="*60)
    print("CREATING VOTING ENSEMBLE")
    print("="*60)
    
    ensemble_data = {
        'models': models,
        'weights': {name: 1.0 / len(models) for name in models.keys()},
        'ensemble_type': 'voting',
        'trained_at': datetime.now().isoformat(),
        'n_models': len(models)
    }
    
    print(f"\n✓ Voting Ensemble created with {len(models)} models")
    print(f"  Equal weights: {1.0/len(models):.3f} each")
    
    return ensemble_data


def create_stacking_ensemble(models, X_train, y_train):
    """Create stacking ensemble with meta-learner"""
    print("\n" + "="*60)
    print("CREATING STACKING ENSEMBLE")
    print("="*60)
    
    # Prepare estimators for stacking
    estimators = [(name, model) for name, model in models.items()]
    
    # Create meta-learner (Logistic Regression)
    meta_learner = LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='lbfgs'
    )
    
    print("\n📚 Base models:", list(models.keys()))
    print("🧠 Meta-learner: Logistic Regression")
    
    # Create stacking classifier
    print("\n⚙️  Training stacking ensemble...")
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,  # Use 5-fold CV for generating meta-features
        stack_method='predict_proba',  # Use probabilities
        n_jobs=-1
    )
    
    stacking_model.fit(X_train, y_train)
    
    print("   ✓ Stacking Ensemble trained with cross-validation")
    print(f"   ✓ Meta-model learned optimal combination weights")
    
    ensemble_data = {
        'model': stacking_model,
        'base_models': list(models.keys()),
        'meta_learner': 'LogisticRegression',
        'ensemble_type': 'stacking',
        'trained_at': datetime.now().isoformat(),
        'n_models': len(models),
        'cv_folds': 5
    }
    
    return ensemble_data


def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    """Evaluate a model and return metrics"""
    metrics = {
        'model': model_name,
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'auc': float(roc_auc_score(y_true, y_pred_proba))
    }
    return metrics


def compare_ensembles(voting_ensemble, stacking_ensemble, X_test, y_test):
    """Compare voting and stacking ensemble performance"""
    print("\n" + "="*60)
    print("ENSEMBLE COMPARISON")
    print("="*60)
    
    results = []
    
    # Evaluate base models
    print("\n📊 Base Models Performance:")
    for model_name, model in voting_ensemble['models'].items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        metrics = evaluate_model(y_test, y_pred, y_pred_proba, model_name)
        results.append(metrics)
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  AUC-ROC:   {metrics['auc']:.4f}")
    
    # Evaluate Voting Ensemble
    print("\n" + "-"*60)
    print("📊 VOTING ENSEMBLE (Simple Averaging):")
    print("-"*60)
    
    voting_proba = np.zeros_like(y_test, dtype=float)
    for model_name, model in voting_ensemble['models'].items():
        voting_proba += model.predict_proba(X_test)[:, 1] / len(voting_ensemble['models'])
    
    voting_pred = (voting_proba > 0.5).astype(int)
    voting_metrics = evaluate_model(y_test, voting_pred, voting_proba, 'voting_ensemble')
    results.append(voting_metrics)
    
    print(f"  Accuracy:  {voting_metrics['accuracy']:.4f}")
    print(f"  Precision: {voting_metrics['precision']:.4f}")
    print(f"  Recall:    {voting_metrics['recall']:.4f}")
    print(f"  F1 Score:  {voting_metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {voting_metrics['auc']:.4f}")
    print(f"  Weights:   Equal ({1.0/len(voting_ensemble['models']):.3f} each)")
    
    # Evaluate Stacking Ensemble
    print("\n" + "-"*60)
    print("📊 STACKING ENSEMBLE (Learned Meta-Model):")
    print("-"*60)
    
    stacking_pred = stacking_ensemble['model'].predict(X_test)
    stacking_proba = stacking_ensemble['model'].predict_proba(X_test)[:, 1]
    stacking_metrics = evaluate_model(y_test, stacking_pred, stacking_proba, 'stacking_ensemble')
    results.append(stacking_metrics)
    
    print(f"  Accuracy:  {stacking_metrics['accuracy']:.4f}")
    print(f"  Precision: {stacking_metrics['precision']:.4f}")
    print(f"  Recall:    {stacking_metrics['recall']:.4f}")
    print(f"  F1 Score:  {stacking_metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {stacking_metrics['auc']:.4f}")
    print(f"  Meta-model: {stacking_ensemble['meta_learner']}")
    
    # Extract meta-learner coefficients
    meta_coef = stacking_ensemble['model'].final_estimator_.coef_[0]
    print(f"\n  Learned weights (meta-model coefficients):")
    for i, model_name in enumerate(stacking_ensemble['base_models']):
        print(f"    {model_name}: {meta_coef[i]:.4f}")
    
    # Winner
    print("\n" + "="*60)
    print("🏆 WINNER DETERMINATION")
    print("="*60)
    
    if stacking_metrics['auc'] > voting_metrics['auc']:
        improvement = (stacking_metrics['auc'] - voting_metrics['auc']) * 100
        print(f"\n🥇 STACKING ENSEMBLE wins!")
        print(f"   AUC improvement: +{improvement:.2f}%")
        winner = 'stacking'
    elif voting_metrics['auc'] > stacking_metrics['auc']:
        improvement = (voting_metrics['auc'] - stacking_metrics['auc']) * 100
        print(f"\n🥇 VOTING ENSEMBLE wins!")
        print(f"   AUC improvement: +{improvement:.2f}%")
        winner = 'voting'
    else:
        print(f"\n🤝 TIE! Both ensembles perform equally well.")
        winner = 'tie'
    
    comparison = {
        'results': results,
        'voting_metrics': voting_metrics,
        'stacking_metrics': stacking_metrics,
        'winner': winner,
        'compared_at': datetime.now().isoformat()
    }
    
    return comparison


def save_comparison_report(comparison, output_dir='outputs/models'):
    """Save detailed comparison report"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save JSON report
    report_path = output_path / 'ensemble_comparison_report.json'
    with open(report_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\n✓ Saved comparison report to {report_path}")
    
    # Create comparison DataFrame
    df = pd.DataFrame(comparison['results'])
    
    # Save CSV
    csv_path = output_path / 'ensemble_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved comparison CSV to {csv_path}")
    
    return df


def save_models(voting_ensemble, stacking_ensemble, output_dir='outputs/models'):
    """Save both ensemble models"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    # Save voting ensemble (keep as ensemble_gold_v1.pkl for compatibility)
    voting_path = output_path / 'ensemble_gold_v1.pkl'
    with open(voting_path, 'wb') as f:
        pickle.dump(voting_ensemble, f)
    print(f"\n✓ Saved Voting Ensemble to {voting_path}")
    
    # Save stacking ensemble
    stacking_path = output_path / 'stacking_ensemble_v1.pkl'
    with open(stacking_path, 'wb') as f:
        pickle.dump(stacking_ensemble, f)
    print(f"✓ Saved Stacking Ensemble to {stacking_path}")
    
    # Save base models individually
    for model_name, model in voting_ensemble['models'].items():
        model_path = output_path / f'{model_name}_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Saved {model_name} to {model_path}")


def main():
    """Main comparison script"""
    print("🚀 GeoAuPredict Ensemble Comparison: Voting vs Stacking\n")
    print("="*60)
    
    # Create or load data
    print("\n📦 Preparing data...")
    X, y = create_sample_data(n_samples=1000)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test:  {len(X_test)} samples")
    print(f"   Features: {X.shape[1]}")
    print(f"   Class balance: {np.sum(y_train)/len(y_train)*100:.1f}% positive")
    
    # Train base models
    models = train_base_models(X_train, y_train)
    
    # Create voting ensemble
    voting_ensemble = create_voting_ensemble(models)
    
    # Create stacking ensemble
    stacking_ensemble = create_stacking_ensemble(models, X_train, y_train)
    
    # Compare performance
    comparison = compare_ensembles(voting_ensemble, stacking_ensemble, X_test, y_test)
    
    # Save comparison report
    df_results = save_comparison_report(comparison)
    
    # Save models
    save_models(voting_ensemble, stacking_ensemble)
    
    # Final summary
    print("\n" + "="*60)
    print("✅ ENSEMBLE COMPARISON COMPLETE!")
    print("="*60)
    print(f"\n📁 All models saved to: outputs/models/")
    print(f"📊 Comparison report: outputs/models/ensemble_comparison_report.json")
    print(f"📈 Performance CSV: outputs/models/ensemble_comparison.csv")
    
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(df_results.to_string(index=False))
    
    return voting_ensemble, stacking_ensemble, comparison


if __name__ == '__main__':
    main()

