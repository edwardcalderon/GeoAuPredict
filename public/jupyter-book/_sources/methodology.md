# Materials and Methods

## Multi-Source Data Integration

GAP integrates six heterogeneous data sources spanning satellite imagery, geochemistry, geophysics, and ground-truth validation:

**Integrated Data Sources**

| **Source** | **Resolution** | **Variables** |
|---|---|---|
| USGS MRDS | Point data | Au occurrences |
| SGC Geochem | 1:100,000 | 35 elements |
| Sentinel-2 | 10-60m | 13 bands |
| SRTM DEM | 30m | Elevation |
| Geophysics | Variable | Mag/Grav |
| Boreholes | Point (147) | Ground truth |

**USGS Mineral Resources (MRDS):** Global mineral occurrence database providing gold-specific locations across Colombia with deposit type classifications.

**Servicio Geológico Colombiano (SGC):** National geochemical surveys at 1:100,000 scale with 35 element concentrations including pathfinder elements (Au, As, Sb, Cu) critical for gold exploration.

**Sentinel-2 Multispectral Imagery:** European Space Agency optical data (10m visible, 20m NIR, 60m atmospheric) enabling spectral indices for alteration mapping (iron oxides, clay minerals, vegetation).

**SRTM Digital Elevation Model:** NASA 30m resolution DEM for terrain analysis including slope, aspect, curvature, topographic wetness index, and flow accumulation—critical for structural geology interpretation.

**Geophysical Surveys:** Magnetic and gravimetric anomaly data revealing subsurface structures, intrusions, and fault systems associated with gold mineralization.

**Colombian Borehole Dataset:** 147 boreholes from Cauca River basin (Caucasia, Antioquia) with 8,642 samples providing spatially-distributed ground truth for model validation. For machine learning experiments, a stratified subset of 1,000 samples (800 training, 200 testing) was used to ensure computational efficiency while maintaining geographic representativeness.

## Geospatial Feature Engineering

We engineered 10 geologically-meaningful features across six categories (with feature selection reducing from 35 candidate features to the top 10 most predictive):

**1. Terrain Morphology (5 features):** Elevation, slope, aspect, plan curvature, profile curvature derived from SRTM DEM using standard geomorphometric methods.

**2. Spectral Indices (3 features):** NDVI calculated as $(B08 - B04)/(B08 + B04)$ for vegetation mapping, Clay Index using $B11/B12$ for alteration detection, and Iron Oxide ratio $B04/B03$ for oxidation zones.

**3. Geochemical Ratios (8 features):** Au concentration, Au/Ag ratio, Cu/As ratio, As/Sb ratio, and normalized concentrations leveraging pathfinder element relationships.

**4. Geological Proximity (2 features):** Euclidean distance to nearest fault (km) and distance to nearest intrusive body (km) using Colombian geological maps.

**5. Geophysical Signatures (2 features):** Magnetic anomaly (nT) and Bouguer gravity anomaly (mGal) indicating subsurface density/magnetic contrasts.

**6. Lithological Encoding (4+ features):** One-hot encoding of rock types (volcanic, sedimentary, metamorphic, intrusive) from SGC geological maps.

## Ensemble Machine Learning Architecture

### Base Model Selection

GAP employs three complementary base models leveraging different inductive biases:

**Random Forest (RF):** Ensemble of 100 decision trees with max depth 10, providing interpretable feature importance and robustness to outliers. Trees use GINI impurity with bootstrap aggregation.

**Algorithm:** For each of $n=100$ trees: (1) Sample bootstrap dataset $\mathcal{D}_t$, (2) Train tree $T_t$ with max depth 10, (3) Average: $RF(x) = \frac{1}{n}\sum_{t}^{n} T_t(x)$

**XGBoost:** Gradient boosting with regularization (L1/L2), learning rate 0.1, max depth 6. Employs histogram-based splitting and column sampling for efficiency.

**LightGBM:** Gradient-based One-Side Sampling (GOSS) with Exclusive Feature Bundling (EFB), learning rate 0.1, max depth 6. Achieves fastest training with competitive accuracy.

### Voting Ensemble (Production Model)

The Voting Ensemble combines base model predictions through simple averaging:

\begin{equation}
| P_{vot}(y|x) = \frac{1}{3}\sum_{i=1}^{3} P_i(y|x) |
\end{equation}

where $P_i$ represents predictions from RF, XGBoost, and LightGBM.

**Implementation Details:** Soft voting using probability estimates with equal weights of 33.3% for each model, requiring no additional training, and resulting in a compact 1.6 MB model file (ensemble_gold_v1.pkl).

**Advantages:** The approach offers simplicity, robustness to overfitting, transparent decision-making, lower computational cost, and better generalization performance on test data.

### Stacking Ensemble (Alternative Model)

The Stacking Ensemble employs meta-learning where a Logistic Regression model learns optimal combination weights:

\begin{equation}
| P_{stack}(y|x) = \sigma\left(\sum_{k=1}^{3} w_k P_k(y|x)\right) |
\end{equation}

where $\sigma$ is sigmoid, $w_k$ are learned weights.

**Implementation Details:** 5-fold cross-validation for meta-feature generation using a Logistic Regression meta-learner with max_iter=1000, learning optimal weights of RF=3.60, LGBM=1.86, and XGB=0.40, resulting in a 3.2 MB model file (stacking_ensemble_v1.pkl).

**Learned Behavior:** The meta-model heavily favors Random Forest despite LightGBM having the best individual AUC of 0.9243, suggesting that Random Forest predictions offer superior complementarity for ensemble performance.

## Spatial Cross-Validation

Standard K-Fold cross-validation overestimates performance for geospatial data due to spatial autocorrelation (Tobler's First Law of Geography). We implement Geographic Block Cross-Validation:

**Methodology:**
Geographic blocks are created by dividing the study area into $k$ regions, where for each fold $i$, the model trains on blocks ${1, \ldots, k} \setminus {i}$ and tests on block $i$, ensuring minimum 50km separation between train/test blocks.

This approach prevents spatial leakage where training samples artificially boost test performance through geographic proximity.

## Production Deployment

**REST API Architecture:** FastAPI framework with asynchronous request handling deployed on Render.com cloud infrastructure.

Live API (<https://geoaupredict.onrender.com>):

**Production API Endpoints**

| **Method** | **Endpoint** | **Description** |
|---|---|---|
| `GET` | [`/health`](https://geoaupredict.onrender.com/health) | System health check returning uptime, memory usage, |
|  |  | and model loading status |
| `POST` | [`/predict`](https://geoaupredict.onrender.com/predict) | Real-time gold probability prediction accepting |
|  |  | geospatial coordinates and returning ensemble |
|  |  | model predictions with confidence scores |
| `GET` | [`/ensemble-info`](https://geoaupredict.onrender.com/ensemble-info) | Model registry metadata including version history, |
|  |  | performance metrics, and ensemble composition |
| `GET` | [`/docs`](https://geoaupredict.onrender.com/docs) | Interactive Swagger/OpenAPI documentation with |
|  |  | request/response schemas and testing interface |

**Model Registry:** The system maintains complete versioning tracking including model artifacts in .pkl files, performance metrics per version, training data provenance, and deployment timestamps.
