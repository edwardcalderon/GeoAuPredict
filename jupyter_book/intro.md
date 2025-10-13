## Introduction

### Industrial Context and Motivation

The global mineral exploration industry faces unprecedented challenges balancing economic viability with environmental sustainability. For example, traditional gold exploration relies on expensive drilling campaigns averaging \$150,000 per borehole, with typical discovery rates below 30% [@colombian_gold]. A standard 100-borehole campaign costs \$15 million with only 30 confirmed deposits, yielding \$500,000 per discovery. This economic burden particularly impacts developing nations like Colombia, where rich mineral resources remain underexplored due to capital constraints.

Beyond financial considerations, conventional exploration generates substantial environmental footprints through invasive drilling, vegetation clearing, and soil disruption across vast territories. The mining industry's contribution to Colombia's GDP (2.2% in 2024) necessitates balancing economic development with ecological preservation—a challenge requiring data-driven, targeted exploration strategies.

Recent advances in artificial intelligence and remote sensing present transformative opportunities for mineral prospectivity mapping. However, existing approaches suffer from: (1) limited integration of heterogeneous data sources, (2) lack of spatial validation leading to overly optimistic performance estimates, (3) insufficient ensemble methodologies for robust predictions, and (4) absence of production-ready deployments for industry adoption.

### Research Objectives

GeoAuPredict (GAP) addresses these limitations through a comprehensive AI system integrating six heterogeneous geospatial data sources with novel ensemble machine learning architectures. Our specific contributions include rigorous evaluation of Voting Ensemble versus Stacking Ensemble approaches, demonstrating that simpler averaging methods yield superior generalization; comprehensive spatial cross-validation using geographic blocks to prevent autocorrelation leakage and ensure honest performance estimates; complete production deployment with REST API, model versioning, and real-time prediction capabilities on cloud infrastructure; and validation showing 2.4× improvement in success rates with 59% cost reduction per discovery.

The remainder of this paper is organized as follows: Section 2 presents the integrated data sources and feature engineering methodology; Section 3 details the ensemble machine learning architecture with implementation specifics; Section 4 reports comprehensive results including ensemble comparison; Section 5 discusses implications for industrial adoption; Section 6 concludes with future research directions.

---

```{admonition} Version 1.0.9
:class: tip
Built: October 13, 2025
```

