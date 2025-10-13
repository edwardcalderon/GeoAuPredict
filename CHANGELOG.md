# Changelog

All notable changes to GeoAuPredict will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-10-13

### Added
- **Voting Ensemble Model**: Simple averaging of RF, XGBoost, LightGBM predictions (AUC: 0.9208) ⭐ PRODUCTION
- **Stacking Ensemble Model**: Meta-learning with Logistic Regression (AUC: 0.9206)
- **Ensemble Comparison Framework**: Comprehensive comparison of ensemble methods
- **Versioning System**: Automated version bumping with VERSION file, __version__.py, and package.json sync
- **REST API Enhancement**: New `/ensemble-info` endpoint with detailed model information
- **Model Registry**: Production-ready model storage with metadata tracking
- **Comprehensive Documentation**: Ensemble comparison report (Markdown, JSON, CSV)

### Changed
- Updated API to use Voting Ensemble as production model (winner of comparison)
- Enhanced model loading with ensemble type detection
- Improved logging for model startup and ensemble information

### Performance
- Voting Ensemble achieves 0.02% better AUC than Stacking Ensemble
- Best individual model: LightGBM (AUC: 0.9243)
- Voting Ensemble selected for production due to simplicity and robustness

### Infrastructure
- Fixed Render.com deployment (removed persistent disk mount conflict)
- All model files now tracked in git repository for deployment
- Docker configuration updated and tested
- Streamlit dashboards ready for deployment

### Files
- `VERSION`: 1.0.0
- `CHANGELOG.md`: Initial changelog
- `src/__version__.py`: Version tracking module
- `scripts/bump_version.py`: Automated version bumping
- `src/models/ensemble_comparison.py`: Ensemble comparison framework
- `outputs/models/ENSEMBLE_COMPARISON_REPORT.md`: Detailed analysis

### Models Released
- `ensemble_gold_v1.pkl` (1.6 MB) - Voting Ensemble - PRODUCTION ✅
- `stacking_ensemble_v1.pkl` (3.2 MB) - Stacking Ensemble - Available
- `random_forest_model.pkl` (1.2 MB) - Base model
- `xgboost_model.pkl` (202 KB) - Base model
- `lightgbm_model.pkl` (220 KB) - Base model

---

## [Unreleased]

### Planned
- Field validation campaign with high-probability targets
- Deep learning model integration (CNN for raster data)
- 3D subsurface visualization
- Real-time prediction API with streaming data
- Expansion to other minerals (Cu, Ag, emeralds)
- International deployment to Andean countries

---

**Note**: For detailed performance metrics and technical specifications, see `outputs/models/ENSEMBLE_COMPARISON_REPORT.md`

