#!/usr/bin/env python3
"""
Sample Data Generator for Gold Prediction Dataset
Creates sample datasets for testing the ingestion pipeline
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from shapely.geometry import Point
import random


def create_sample_sgc_data(output_path: Path, n_samples: int = 1000):
    """Create sample SGC geochemical data"""

    # Colombia bounds approximately
    lat_range = (1.0, 12.0)
    lon_range = (-79.0, -67.0)

    # Generate random points within Colombia
    lats = np.random.uniform(lat_range[0], lat_range[1], n_samples)
    lons = np.random.uniform(lon_range[0], lon_range[1], n_samples)

    # Create sample geochemical data
    data = {
        'ID_MUESTRA': range(1, n_samples + 1),
        'LATITUD': lats,
        'LONGITUD': lons,
        'AU_PPM': np.random.exponential(0.05, n_samples),  # Most values low, few high
        'AS_PPM': np.random.exponential(1.0, n_samples),
        'CU_PPM': np.random.exponential(5.0, n_samples),
        'FE_PPM': np.random.exponential(1000.0, n_samples),
        'SB_PPM': np.random.exponential(0.5, n_samples),
        'PROYECTO': np.random.choice(['Proyecto_1', 'Proyecto_2', 'Proyecto_3'], n_samples),
        'FECHA_MUESTREO': pd.date_range('2020-01-01', periods=n_samples, freq='D').strftime('%Y-%m-%d')
    }

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, sep=';')
    print(f"Created sample SGC data: {output_path}")


def create_sample_usgs_data(output_path: Path, n_samples: int = 100):
    """Create sample USGS MRDS gold deposits data"""

    # Generate gold deposit locations (concentrated in known gold regions)
    gold_regions = [
        (6.2, -74.5),  # Antioquia region
        (2.5, -76.5),  # Cauca region
        (7.0, -73.0),  # Santander region
        (4.5, -75.5),  # Valle del Cauca region
    ]

    lats, lons = [], []
    for _ in range(n_samples):
        region = random.choice(gold_regions)
        lat_noise = np.random.normal(0, 0.5)
        lon_noise = np.random.normal(0, 0.5)
        lats.append(region[0] + lat_noise)
        lons.append(region[1] + lon_noise)

    data = {
        'site_name': [f'Gold_Deposit_{i}' for i in range(1, n_samples + 1)],
        'latitude': lats,
        'longitude': lons,
        'country': 'Colombia',
        'state': np.random.choice(['Antioquia', 'Cauca', 'Santander', 'Chocó', 'Nariño'], n_samples),
        'commodity': 'Gold',
        'orebody_type': np.random.choice(['Vein', 'Disseminated', 'Placer', 'Skarn'], n_samples),
        'development_status': np.random.choice(['Past Producer', 'Producer', 'Prospect'], n_samples)
    }

    df = pd.DataFrame(data)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs='EPSG:4326'
    )
    gdf.to_file(output_path, driver='GeoJSON')
    print(f"Created sample USGS data: {output_path}")


def create_sample_srtm_data(output_path: Path):
    """Create a minimal SRTM-like raster file for testing"""

    # This is a simplified example - in practice you'd download real SRTM data
    # For now, we'll create a placeholder file
    print(f"SRTM data would be downloaded to: {output_path}")
    print("In practice, download SRTM data from: https://earthexplorer.usgs.gov/")


def main():
    """Generate sample datasets"""

    data_dir = Path('data')
    raw_dir = data_dir / 'raw'
    external_dir = data_dir / 'external'
    processed_dir = data_dir / 'processed'

    # Create directories
    for dir_path in [raw_dir, external_dir, processed_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    print("Creating sample datasets for testing...")

    # Create sample SGC geochemical data
    sgc_path = raw_dir / 'geoquimica_sgc.csv'
    create_sample_sgc_data(sgc_path, n_samples=1000)

    # Create sample USGS gold deposits
    usgs_path = processed_dir / 'usgs_gold.geojson'
    create_sample_usgs_data(usgs_path, n_samples=100)

    # SRTM placeholder
    srtm_path = external_dir / 'srtm_colombia.tif'
    create_sample_srtm_data(srtm_path)

    print("\nSample datasets created!")
    print(f"📊 SGC Geochemical: {sgc_path}")
    print(f"🏆 USGS Gold Deposits: {usgs_path}")
    print(f"🗺️  SRTM DEM: {srtm_path}")
    print("\nYou can now test the ingestion pipeline with:")
    print("python src/ingest/data_ingest.py --skip-sentinel")


if __name__ == "__main__":
    main()
