---
title: Discussion
---

# Discussion

## Ensemble Architecture Selection

Our finding that Voting Ensemble outperforms Stacking Ensemble contradicts conventional wisdom suggesting meta-learning should always improve performance. We attribute this to:

**1. Dataset Size:** With 1000 training samples, stacking meta-model may overfit on cross-validated predictions, especially when base models already achieve high AUC (\>0.91).

**2. Model Diversity:** RF, XGBoost, and LightGBM share similar decision boundary structures (all tree-based), limiting complementarity gains from learned weighting.

**3. Simplicity Bias:** Equal weighting provides implicit regularization preventing meta-model from exploiting spurious patterns in validation folds.

This suggests production systems should rigorously evaluate both voting and stacking approaches rather than assuming meta-learning superiority.

## Industrial Adoption Implications

GAP demonstrates several requirements for industry adoption:

**Economic Viability:** 59% cost reduction per discovery (\$295k savings) provides clear ROI. For a company conducting 10 annual campaigns, GAP yields \$3M annual savings.

**Risk Mitigation:** 71% success rate vs 30% baseline reduces exploration failure risk by 2.4$\times$, enabling smaller companies to compete with resource-rich competitors.

**Environmental Responsibility:** Targeting high-probability areas reduces unnecessary drilling by 75%, minimizing ecological disruption while maintaining discovery rates.

**Scalability:** REST API architecture enables integration with existing GIS workflows, enterprise resource planning (ERP) systems, and mobile field applications.

## Limitations and Future Work

**Geographic Generalization:** Current model trained exclusively on Colombian data. Transfer learning to other Andean regions (Peru, Ecuador) could validate cross-border applicability.

**Temporal Dynamics:** Static model doesn't incorporate temporal changes in land use, vegetation, or environmental conditions. Time-series integration with continuous Sentinel-2 could improve predictions.

**Deep Learning Integration:** Future work should explore convolutional neural networks (CNNs) for raw satellite imagery processing, potentially extracting features tree-based models cannot capture.

**Multi-Mineral Extension:** Architecture readily extends to other minerals (copper, silver, zinc) by retraining with appropriate geochemical pathfinders.

**Uncertainty Quantification:** While ensemble variance provides uncertainty estimates, formal calibration (e.g., conformal prediction) would enable probabilistic guarantees for risk-averse exploration decisions.

