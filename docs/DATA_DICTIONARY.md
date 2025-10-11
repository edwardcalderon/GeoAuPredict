# 📚 GeoAuPredict Data Dictionary

## Overview
Comprehensive documentation of all datasets, variables, and metadata in the GeoAuPredict data lake.

## 🏗️ Data Lake Structure

### Raw Data (`data/raw/`)
Contains original, unprocessed data from various sources.

#### SGC Geochemistry (`sgc/`)
**Source**: Servicio Geológico Colombiano geochemical sampling database
**Format**: CSV files with geochemical analysis results
**Geographic Coverage**: Colombia

| Column | Type | Unit | Description | Source |
|--------|------|------|-------------|---------|
| ID_MUESTRA | string | - | Unique sample identifier | SGC |
| LATITUD | float | degrees | Latitude (WGS84) | SGC |
| LONGITUD | float | degrees | Longitude (WGS84) | SGC |
| AU_PPM | float | ppm | Gold concentration | SGC |
| AS_PPM | float | ppm | Arsenic concentration | SGC |
| CU_PPM | float | ppm | Copper concentration | SGC |
| FE_PPM | float | ppm | Iron concentration | SGC |
| SB_PPM | float | ppm | Antimony concentration | SGC |
| FECHA_MUESTREO | string | YYYY-MM-DD | Sampling date | SGC |
| PROYECTO | string | - | Project name/code | SGC |
| METODO | string | - | Analytical method | SGC |

#### USGS MRDS (`usgs/`)
**Source**: USGS Mineral Resource Data System
**Format**: CSV files with global mineral deposit information
**Geographic Coverage**: Global, filtered to gold deposits

| Column | Type | Unit | Description | Source |
|--------|------|------|-------------|---------|
| site_name | string | - | Deposit/site name | USGS |
| latitude | float | degrees | Latitude (WGS84) | USGS |
| longitude | float | degrees | Longitude (WGS84) | USGS |
| country | string | - | Country name | USGS |
| state | string | - | State/province | USGS |
| commodity | string | - | Primary commodity | USGS |
| orebody_type | string | - | Ore body type | USGS |
| development_status | string | - | Development status | USGS |

#### SRTM Elevation (`srtm/`)
**Source**: Shuttle Radar Topography Mission
**Format**: GeoTIFF raster files
**Geographic Coverage**: Global

| Dataset | Resolution | Unit | Description |
|---------|------------|------|-------------|
| SRTM_COLOMBIA | 30m | meters | Digital Elevation Model for Colombia |

#### Sentinel-2 Imagery (`sentinel/`)
**Source**: ESA Sentinel-2 satellite program
**Format**: Processed raster indices
**Geographic Coverage**: Colombia

| Index | Formula | Description |
|-------|---------|-------------|
| NDVI | (B08 - B04) / (B08 + B04) | Normalized Difference Vegetation Index |
| Clay_Index | B11 / B8A | Clay mineral indicator |
| Iron_Index | B04 / B02 | Iron oxide indicator |

### External Data (`data/external/`)
Externally sourced datasets and reference data.

#### Digital Elevation Models (`dem/`)
| Dataset | Source | Resolution | Coverage |
|---------|--------|------------|-----------|
| SRTM | USGS | 30m | Colombia |

#### Geological Data (`geological/`)
| Dataset | Source | Format | Description |
|---------|--------|--------|-------------|
| Geological Maps | IGAC | Shapefile | 1:100,000 geological maps |
| Fault Database | SGC | GeoJSON | Major fault systems |

## 🧮 Standardized Variables

### Coordinates
| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| latitude | float | -90 to 90 | Latitude in WGS84 |
| longitude | float | -180 to 180 | Longitude in WGS84 |

### Geochemical Elements
| Element | Symbol | Unit | Typical Range | Detection Limit |
|---------|--------|------|---------------|-----------------|
| Gold | Au | ppm | 0.001 - 100 | 0.001 |
| Arsenic | As | ppm | 0.1 - 1000 | 0.1 |
| Copper | Cu | ppm | 1 - 10000 | 1 |
| Iron | Fe | ppm | 1000 - 100000 | 100 |
| Antimony | Sb | ppm | 0.1 - 100 | 0.1 |

### Spectral Indices
| Index | Range | Description |
|-------|-------|-------------|
| NDVI | -1 to 1 | Vegetation health indicator |
| Clay_Index | 0 to 10 | Clay mineral abundance |
| Iron_Index | 0 to 5 | Iron oxide abundance |

### Geophysical Data
| Variable | Unit | Description |
|----------|------|-------------|
| magnetic_anomaly | nT | Magnetic field anomaly |
| gravity_anomaly | mGal | Bouguer gravity anomaly |

### Geological Features
| Variable | Type | Description |
|----------|------|-------------|
| lithology | string | Rock type classification |
| fault_distance | km | Distance to nearest fault |
| formation_age | Ma | Geological formation age |

## 🏷️ Target Variables

### Gold Labels
| Variable | Type | Values | Description |
|----------|------|--------|-------------|
| gold_label | integer | 0, 1 | Binary gold presence indicator |
| gold_threshold | float | ppm | Threshold used for labeling |

**Labeling Logic:**
- `gold_label = 1` if `AU_PPM >= gold_threshold`
- `gold_label = 0` if `AU_PPM < gold_threshold`

### Confidence Scores
| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| confidence_score | float | 0-1 | Reliability of gold label |

## 📊 Processed Datasets

### Master Dataset (`gold_dataset_master.*`)
Final integrated dataset for machine learning.

| Column Group | Count | Description |
|--------------|-------|-------------|
| Coordinates | 2 | latitude, longitude |
| Geochemistry | 8 | Au, As, Cu, Fe, Sb, etc. |
| Spectral | 3 | NDVI, Clay_Index, Iron_Index |
| Geophysical | 2 | magnetic_anomaly, gravity_anomaly |
| Geological | 5 | lithology, fault_distance, etc. |
| Topography | 3 | elevation, slope, aspect |
| Targets | 3 | gold_label, confidence_score, gold_threshold |
| Metadata | 5 | sample_id, source, date, project |

### Dataset Statistics
- **Total Features**: ~25 variables
- **Target Classes**: Binary (gold presence/absence)
- **Spatial Coverage**: Colombia mainland
- **Temporal Range**: 1980-2023

## 🔍 Data Quality Standards

### Completeness
- **Minimum Threshold**: 80% complete records
- **Critical Variables**: Coordinates, gold measurements
- **Handling**: Records below threshold are flagged

### Accuracy
- **Coordinate Precision**: 6 decimal places (~10cm)
- **Element Detection**: Laboratory-certified values
- **Geological Mapping**: 1:100,000 scale minimum

### Consistency
- **CRS**: All data in EPSG:4326 (WGS84)
- **Units**: Standardized units (ppm, meters, degrees)
- **Formats**: Consistent column naming and data types

## 🚨 Data Issues & Limitations

### Known Issues
1. **SGC Data Gaps**: Some departments have limited sampling coverage
2. **USGS Coordinate Accuracy**: Some legacy deposits have approximate coordinates
3. **Temporal Inconsistency**: Different sampling periods across datasets

### Data Limitations
1. **Sampling Bias**: Geochemical sampling focused on known mineralized areas
2. **Detection Limits**: Values below detection limit reported as "<0.1"
3. **Scale Differences**: Point data vs. regional geophysical surveys

## 📋 Data Sources & Access

### Primary Sources
- **SGC**: https://www.sgc.gov.co/
- **USGS**: https://mrdata.usgs.gov/
- **SRTM**: https://earthexplorer.usgs.gov/

### Data Access Policies
- **SGC Data**: Public domain, requires registration
- **USGS Data**: Public domain, open access
- **Sentinel-2**: ESA open data policy

## 🔄 Data Update Schedule

### Regular Updates
- **SGC Data**: Quarterly (new sampling campaigns)
- **USGS Data**: Annual (MRDS updates)
- **Sentinel-2**: Monthly (new imagery)

### Update Procedures
1. Download latest data from sources
2. Run validation pipeline
3. Update master dataset
4. Archive previous versions
5. Update documentation

## 📞 Support & Documentation

### Getting Help
- **Issues**: Create GitHub issue in repository
- **Documentation**: Check `docs/` folder
- **Data Questions**: Review this data dictionary

### Version History
- **v1.0.0**: Initial data lake structure (2024)
- **v1.1.0**: Added Sentinel-2 processing (2024)
- **v2.0.0**: Enhanced validation and standardization (2024)

---

*This data dictionary is automatically generated and maintained as part of the data lake validation process.*
