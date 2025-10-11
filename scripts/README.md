# 📊 GeoAuPredict Data Lake

## Overview
A comprehensive geospatial data lake for gold prediction in Colombia, organizing multiple data sources into a standardized structure for AI/ML applications.

## 🏗️ Data Lake Structure

```
data/
├── raw/                          # Original, unprocessed data
│   ├── sgc/                     # SGC geochemical data
│   ├── usgs/                    # USGS MRDS mineral deposits
│   ├── sentinel/                # Sentinel-2 satellite imagery
│   ├── srtm/                    # SRTM elevation data
│   ├── perforaciones/           # Drilling data
│   ├── magnetic/                # Magnetic anomaly data
│   ├── gravity/                 # Gravity data
│   └── lithology/               # Geological maps
├── processed/                   # Cleaned and processed data
│   ├── datasets/               # Final ML-ready datasets
│   ├── interim/                # Intermediate processing results
│   └── cache/                  # Cached data and temp files
└── external/                   # External datasets
    ├── dem/                    # Digital elevation models
    ├── satellite/              # Satellite imagery
    └── geological/             # Geological data
```

## 🚀 Quick Start

### 1. Setup Data Lake Structure
```bash
python scripts/data_ingestion/ingest_raw_data.py --create-structure
```

### 2. Ingest Raw Data
```bash
# Ingest all sources
python scripts/data_ingestion/ingest_raw_data.py

# Ingest specific sources
python scripts/data_ingestion/ingest_raw_data.py --ingest-sgc
python scripts/data_ingestion/ingest_raw_data.py --ingest-usgs
```

### 3. Validate Data Quality
```bash
python scripts/data_ingestion/validate_data.py
```

## 📋 Data Sources

### SGC Geochemistry
- **Source**: Servicio Geológico Colombiano
- **Format**: CSV with geochemical sampling data
- **Location**: `data/raw/sgc/`
- **Elements**: Au, As, Cu, Fe, Sb, etc.

### USGS MRDS
- **Source**: USGS Mineral Resource Data System
- **Format**: CSV with global mineral deposits
- **Location**: `data/raw/usgs/`
- **Focus**: Gold deposits worldwide

### SRTM Elevation
- **Source**: Shuttle Radar Topography Mission
- **Format**: GeoTIFF DEM files
- **Location**: `data/external/dem/`
- **Coverage**: Global elevation data

### Sentinel-2
- **Source**: ESA Copernicus program
- **Format**: Processed spectral indices
- **Location**: `data/raw/sentinel/`
- **Requires**: SentinelHub API credentials

## 🛠️ Data Processing Pipeline

### Standardization
1. **Coordinate Normalization**: Convert all coordinates to WGS84 (EPSG:4326)
2. **Element Standardization**: Map element names to standard nomenclature
3. **Units Normalization**: Ensure consistent units (ppm, meters, etc.)
4. **Format Conversion**: Convert to ML-friendly formats

### Quality Control
1. **Spatial Filtering**: Filter to Colombia bounds
2. **Duplicate Removal**: Remove duplicate coordinates
3. **Outlier Detection**: Identify and handle outliers
4. **Missing Data Handling**: Impute or remove missing values

### AI Compatibility
1. **Feature Engineering**: Create derived features
2. **Label Creation**: Generate gold presence/absence labels
3. **Scaling**: Normalize features for ML algorithms
4. **Splitting**: Create train/test/validation splits

## 📊 Data Formats

### Input Formats Supported
- **CSV**: Comma/tab/semicolon separated values
- **GeoJSON**: Geographic JSON format
- **GeoTIFF**: Geospatial raster data
- **Shapefile**: ESRI shapefile format

### Output Formats
- **CSV**: Machine learning ready tabular data
- **GeoJSON**: GIS-ready geospatial data
- **Parquet**: Efficient columnar storage

## 🔍 Validation & Quality Checks

### Automated Validation
- Coordinate validity (lat/lon ranges)
- Data type consistency
- Missing value assessment
- Duplicate detection
- Spatial bounds checking

### Quality Reports
- Data completeness scores
- Outlier analysis
- Distribution statistics
- Spatial coverage maps

## 📚 Usage Examples

### Basic Data Ingestion
```python
from scripts.data_ingestion.ingest_raw_data import DataLakeManager

manager = DataLakeManager("config.yaml")
manager.run_full_ingestion()
```

### Data Standardization
```python
from scripts.data_ingestion.data_utils import DataStandardizer

standardizer = DataStandardizer()
df = standardizer.standardize_coordinates(df)
df = standardizer.standardize_elements(df)
df = standardizer.create_gold_labels(df, threshold=0.1)
```

### Data Validation
```python
validation_results = standardizer.validate_dataset(df)
print(f"Dataset quality: {validation_results}")
```

## 🗂️ Metadata & Documentation

### Data Dictionary
Each dataset includes comprehensive metadata:
- Source information
- Collection methods
- Processing history
- Quality metrics
- Usage recommendations

### Data Catalog
- Automated catalog generation
- Dataset discovery
- Version tracking
- Access permissions

## 🔧 Configuration

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure data sources
cp config_template.yaml config.yaml
# Edit config.yaml with your settings
```

### Configuration Options
- Data source URLs and paths
- Processing parameters
- Quality thresholds
- Output formats
- Spatial bounds

## 🚨 Best Practices

### Data Management
- Always backup raw data before processing
- Maintain clear data lineage
- Use consistent naming conventions
- Document data transformations

### Quality Assurance
- Validate data after each processing step
- Use automated quality checks
- Maintain processing logs
- Regular data audits

### Performance
- Use appropriate batch sizes
- Cache expensive operations
- Monitor memory usage
- Parallel processing where possible

## 📈 Monitoring & Maintenance

### Data Health Checks
- Regular validation runs
- Storage usage monitoring
- Processing time tracking
- Error rate monitoring

### Updates & Maintenance
- Regular data source updates
- Schema evolution handling
- Performance optimization
- Documentation updates

## 🆘 Troubleshooting

### Common Issues
1. **Missing Dependencies**: Install with `pip install -r requirements.txt`
2. **Permission Errors**: Check file/folder permissions
3. **Memory Issues**: Reduce batch size in config
4. **Network Errors**: Check internet connection for downloads

### Getting Help
- Check logs in `data/processed/`
- Review configuration in `config.yaml`
- Validate data manually with provided scripts
- Check documentation in `docs/`

---

## 🎯 Next Steps

1. **Complete FASE 1**: Finish data lake organization
2. **FASE 2**: Implement ETL pipeline
3. **FASE 3**: Build ML feature engineering
4. **FASE 4**: Deploy prediction models
