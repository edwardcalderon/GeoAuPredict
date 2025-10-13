# GeoAuPredict - Gold Prediction Dataset

A comprehensive geospatial dataset and ingestion pipeline for gold deposit prediction in Colombia, integrating multiple data sources including geochemical sampling, geological features, satellite imagery, and known mineral deposits.

## 📓 Interactive Notebooks

**Try the complete project demo online - no installation required!**

Choose your preferred platform:

| 🔷 Google Colab (with GPU) | 🟠 Binder (no login) |
|---|---|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edwardcalderon/GeoAuPredict/blob/main/notebooks/GeoAuPredict_Project_Presentation.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/edwardcalderon/GeoAuPredict/main?filepath=notebooks/GeoAuPredict_Project_Presentation.ipynb) |
| Requires Google account | No account needed |
| Faster, with GPU support | May take 2-3 min to start |

**Available Notebooks:**
- 📊 **[Project Presentation](notebooks/GeoAuPredict_Project_Presentation.ipynb)** - Complete project demo with visualizations
- 🔧 **[Complete Pipeline](notebooks/GeoAuPredict_Complete_Pipeline_Base.ipynb)** - Full data processing workflow
- 🗺️ **[Spatial Validation](notebooks/GeoAuPredict_Spatial_Validation.ipynb)** - Advanced validation techniques

## 🚀 Deployment

This project is configured for automatic deployment to GitHub Pages.

### Live Demo
[View Live Site](https://edwardcalderon.github.io/GeoAuPredict/) (original repository)

> **Note for Forked Repositories**: When you fork this repository, your deployment URL will be `https://yourusername.github.io/GeoAuPredict/` 

### Deployment Setup

1. **Enable GitHub Pages**:
   - Go to repository settings → Pages
   - Set source to "GitHub Actions"

2. **Automatic Deployment**:
   - Pushes to `main` branch trigger automatic builds
   - Static export is generated and deployed

3. **Manual Deployment**:
   ```bash
   npm run export    # Build static export
   npm run deploy    # Deploy to GitHub Pages
   ```

## 📋 Overview

This project creates a machine learning-ready dataset for gold prediction by combining:

- **USGS MRDS**: Global gold deposit locations
- **SGC Geochemistry**: Colombian geochemical sampling data
- **SRTM DEM**: Elevation data
- **Sentinel-2**: Spectral indices (NDVI, Clay, Iron)
- **Geological features**: Magnetic anomalies, gravity data, fault distances, lithology

## 🏗️ Project Structure

```
GeoAuPredict/
├── data/
│   ├── raw/              # Raw input data
│   ├── processed/        # Processed and cleaned data
│   └── external/         # External datasets (SRTM, etc.)
├── src/
│   └── ingest/
│       └── data_ingest.py    # Main ingestion pipeline
├── config_template.yaml      # Configuration template
├── requirements.txt          # Python dependencies
├── create_sample_data.py     # Sample data generator
└── README.md                # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd GeoAuPredict

# Install dependencies
pip install -r requirements.txt

# Optional: Install Sentinel-2 support
pip install sentinelhub
```

### 2. Configuration

```bash
# Copy and edit configuration template
cp config_template.yaml config.yaml
# Edit config.yaml with your settings
```

### 3. Data Acquisition

#### USGS MRDS Data
- Automatically downloaded by the ingestion script

#### SGC Geochemical Data
- Download from [SGC Portal](https://www.sgc.gov.co/)
- Place in `data/raw/geoquimica_sgc.csv`

#### SRTM Elevation Data
- Download from [USGS EarthExplorer](https://earthexplorer.usgs.gov/)
- Place in `data/external/srtm_colombia.tif`

#### Sentinel-2 (Optional)
- Requires [SentinelHub](https://sentinelhub.com/) account
- Configure credentials in `config.yaml`

### 4. Run Data Ingestion

```bash
# Basic usage
python src/ingest/data_ingest.py

# Skip Sentinel-2 processing (faster)
python src/ingest/data_ingest.py --skip-sentinel

# Use custom config file
python src/ingest/data_ingest.py --config-file config.yaml

# Verbose logging
python src/ingest/data_ingest.py --log-level DEBUG
```

### 5. Test with Sample Data

```bash
# Generate sample datasets for testing
python create_sample_data.py

# Test ingestion with sample data
python src/ingest/data_ingest.py --skip-sentinel
```

## 📊 Dataset Structure

### Core Columns

| Column | Type | Unit | Description | Source |
|--------|------|------|-------------|---------|
| `id` | int | - | Unique identifier | Internal |
| `lat` | float | degrees | Latitude WGS84 | SGC/USGS |
| `lon` | float | degrees | Longitude WGS84 | SGC/USGS |
| `elev` | float | meters | Elevation | SRTM |
| `Au_ppm` | float | ppm | Gold concentration | Geochemistry |
| `As_ppm` | float | ppm | Arsenic | Geochemistry |
| `Cu_ppm` | float | ppm | Copper | Geochemistry |
| `Fe_ppm` | float | ppm | Iron | Geochemistry |
| `Sb_ppm` | float | ppm | Antimony | Geochemistry |
| `NDVI` | float | index | Vegetation index | Sentinel-2 |
| `Clay_Index` | float | index | Clay minerals | Sentinel-2 |
| `Iron_Index` | float | index | Iron oxides | Sentinel-2 |
| `Mag_Anomaly` | float | nT | Magnetic anomaly | USGS/SGC |
| `Grav_Anomaly` | float | mGal | Gravity anomaly | USGS |
| `Dist_Fault` | float | km | Distance to fault | Geology |
| `Lithology` | string | - | Rock type | IGAC/SGC |
| `label_gold` | int | 0-1 | Gold presence | USGS/SGC |
| `source` | string | - | Data source | Internal |
| `date` | string | ISO8601 | Sampling date | Geochemistry |
| `region` | string | - | Administrative region | SGC/IGAC |

### Output Formats

- **CSV**: Machine learning ready format
- **GeoJSON**: GIS visualization and spatial analysis
- **CRS**: EPSG:4326 (WGS84)

## ⚙️ Configuration

### Data Sources

```yaml
sources:
  sgc:
    csv_path: "data/raw/geoquimica_sgc.csv"
    column_mapping:
      latitude: ["LATITUD", "LAT"]
      gold_ppm: ["AU_PPM", "Au_ppm"]
    gold_threshold: 0.1

  sentinel2:
    credentials:
      instance_id: "your-instance-id"
      client_id: "your-client-id"
      client_secret: "your-client-secret"
```

### Processing Options

```yaml
processing:
  batch_size: 100
  workers: 4
  request_timeout: 60

quality_filters:
  spatial_filter: true
  remove_duplicates: true
  min_elevation: -100
```

## 🛠️ Development

### Adding New Data Sources

1. Extend the `GoldDataIngester` class
2. Add configuration options to `config_template.yaml`
3. Update the `unify_datasets()` method
4. Add new columns to the dataset structure

### Testing

```bash
# Run with sample data
python create_sample_data.py
python src/ingest/data_ingest.py --skip-sentinel

# Check output
ls -la data/processed/
```

## 📈 Usage Examples

### Basic Analysis

```python
import pandas as pd
import geopandas as gpd

# Load the dataset
df = pd.read_csv('data/processed/gold_dataset_master.csv')
gdf = gpd.read_file('data/processed/gold_dataset_master.geojson')

# Basic statistics
print(df['label_gold'].value_counts())
print(df.describe())

# Spatial analysis
gdf.plot(column='Au_ppm', legend=True)
```

### Machine Learning

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Prepare features
features = ['lat', 'lon', 'elev', 'Au_ppm', 'As_ppm', 'Cu_ppm', 'NDVI']
X = df[features].fillna(0)
y = df['label_gold']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
print(f"Accuracy: {model.score(X_test, y_test):.3f}")
```

## 🔧 Troubleshooting

### Common Issues

1. **Missing SGC data**: Ensure the geochemical file is properly formatted and placed in the correct location
2. **Sentinel-2 credentials**: Configure SentinelHub credentials for satellite data
3. **Memory issues**: Use `--skip-sentinel` for faster processing with less memory usage
4. **Coordinate issues**: Verify that coordinates are in EPSG:4326 format

### Logs

Check `data/processed/data_ingestion.log` for detailed error messages and processing information.

## 📚 References

- [USGS MRDS](https://mrdata.usgs.gov/mrds/)
- [Servicio Geológico Colombiano](https://www.sgc.gov.co/)
- [SentinelHub](https://sentinelhub.com/)
- [SRTM Data](https://earthexplorer.usgs.gov/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the logs in `data/processed/data_ingestion.log`