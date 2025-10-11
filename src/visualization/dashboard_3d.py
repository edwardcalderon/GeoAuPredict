"""
3D Geospatial Visualization Framework
Interactive 3D dashboard using CesiumJS, Leaflet 3D, and Kepler.gl
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import rasterio
import geopandas as gpd

logger = logging.getLogger(__name__)


class CesiumJSVisualizer:
    """3D visualization using CesiumJS"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def create_cesium_visualization(
        self,
        probability_maps: List[Path],
        borehole_data: pd.DataFrame,
        geological_features: List[Path],
        output_file: str = "cesium_visualization.html"
    ) -> Path:
        """
        Create CesiumJS 3D visualization

        Args:
            probability_maps: List of gold probability map paths
            borehole_data: DataFrame with borehole information
            geological_features: List of geological feature map paths
            output_file: Output HTML file name

        Returns:
            Path to generated visualization file
        """
        logger.info("Creating CesiumJS 3D visualization")

        # Generate CesiumJS HTML template
        html_content = self._generate_cesium_html(
            probability_maps,
            borehole_data,
            geological_features
        )

        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            f.write(html_content)

        logger.info(f"CesiumJS visualization saved: {output_path}")
        return output_path

    def _generate_cesium_html(
        self,
        probability_maps: List[Path],
        borehole_data: pd.DataFrame,
        geological_features: List[Path]
    ) -> str:
        """Generate CesiumJS HTML content"""

        # Convert borehole data to GeoJSON for CesiumJS
        borehole_geojson = self._boreholes_to_geojson(borehole_data)

        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>GeoAuPredict - 3D Gold Exploration Dashboard</title>

    <!-- CesiumJS -->
    <script src="https://cesium.com/downloads/cesiumjs/releases/1.100/Build/Cesium/Cesium.js"></script>
    <link href="https://cesium.com/downloads/cesiumjs/releases/1.100/Build/Cesium/Widgets/widgets.css" rel="stylesheet">

    <style>
        body {{ margin: 0; padding: 0; overflow: hidden; }}
        #cesiumContainer {{ width: 100%; height: 100vh; }}
        .legend {{ position: absolute; top: 10px; right: 10px; background: rgba(255,255,255,0.9); padding: 10px; border-radius: 5px; }}
        .borehole-popup {{ background: rgba(255,255,255,0.95); padding: 10px; border-radius: 5px; max-width: 300px; }}
    </style>
</head>
<body>
    <div id="cesiumContainer"></div>

    <div class="legend">
        <h4>Gold Probability Legend</h4>
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="width: 20px; height: 20px; background: #ff0000; margin-right: 10px;"></div>
            <span>High Probability (>0.8)</span>
        </div>
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="width: 20px; height: 20px; background: #ffa500; margin-right: 10px;"></div>
            <span>Medium Probability (0.5-0.8)</span>
        </div>
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="width: 20px; height: 20px; background: #ffff00; margin-right: 10px;"></div>
            <span>Low Probability (0.2-0.5)</span>
        </div>
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="width: 20px; height: 20px; background: #00ff00; margin-right: 10px;"></div>
            <span>Very Low Probability (<0.2)</span>
        </div>
    </div>

    <script>
        // Initialize CesiumJS viewer
        Cesium.Ion.defaultAccessToken = 'your_cesium_token_here';

        const viewer = new Cesium.Viewer('cesiumContainer', {{
            terrainProvider: Cesium.createWorldTerrain(),
            imageryProvider: new Cesium.IonImageryProvider({{ assetId: 2 }})
        }});

        // Add Colombia bounding box
        const colombiaRectangle = Cesium.Rectangle.fromDegrees(-79.0, -4.3, -66.8, 12.5);
        viewer.camera.setView({{
            destination: colombiaRectangle,
            orientation: {{
                heading: 0.0,
                pitch: -0.5,
                roll: 0.0
            }}
        }});

        // Add gold probability overlays
        {self._generate_probability_overlays(probability_maps)}

        // Add borehole data
        {self._generate_borehole_entities(borehole_geojson)}

        // Add geological features
        {self._generate_geological_overlays(geological_features)}

        // Add interactivity
        viewer.selectedEntityChanged.addEventListener(function(entity) {{
            if (entity && entity.properties) {{
                const popup = document.createElement('div');
                popup.className = 'borehole-popup';
                popup.innerHTML = `
                    <h4>Borehole Information</h4>
                    <p><strong>Gold Concentration:</strong> ${{entity.properties.au_ppm}} ppm</p>
                    <p><strong>Lithology:</strong> ${{entity.properties.lithology}}</p>
                    <p><strong>Depth:</strong> ${{entity.properties.depth}} m</p>
                `;

                viewer.selectedEntity.description = popup.outerHTML;
            }}
        }});
    </script>
</body>
</html>
        """

        return html_template

    def _boreholes_to_geojson(self, borehole_df: pd.DataFrame) -> str:
        """Convert borehole DataFrame to GeoJSON for CesiumJS"""
        features = []

        for _, row in borehole_df.iterrows():
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row['lon'], row['lat']]
                },
                "properties": {
                    "au_ppm": row.get('au_ppm', 0),
                    "lithology": row.get('lithology_text', 'Unknown'),
                    "depth": row.get('depth_m', 0),
                    "label_gold": row.get('label_gold', 0)
                }
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        return json.dumps(geojson)

    def _generate_probability_overlays(self, probability_maps: List[Path]) -> str:
        """Generate probability overlay code for CesiumJS"""
        overlays = []

        for i, prob_map in enumerate(probability_maps):
            overlays.append(f"""
        // Probability overlay {i+1}
        const probabilityProvider{i} = new Cesium.SingleTileImageryProvider({{
            url: '{prob_map}',
            rectangle: Cesium.Rectangle.fromDegrees(-79.0, -4.3, -66.8, 12.5)
        }});

        const probabilityLayer{i} = viewer.imageryLayers.addImageryProvider(probabilityProvider{i});
        probabilityLayer{i}.alpha = 0.7;
            """)

        return "\n".join(overlays)

    def _generate_borehole_entities(self, borehole_geojson: str) -> str:
        """Generate borehole entity code for CesiumJS"""
        return f"""
        // Add borehole data
        const boreholeDataSource = new Cesium.GeoJsonDataSource();
        boreholeDataSource.load(JSON.parse('{borehole_geojson.replace(chr(39), chr(39)+chr(39))}'), {{
            clampToGround: true
        }});

        viewer.dataSources.add(boreholeDataSource);

        // Style boreholes based on gold content
        boreholeDataSource.entities.values.forEach(function(entity) {{
            const au_ppm = entity.properties.au_ppm;
            const height = Math.min(au_ppm * 10, 1000); // Scale height by gold content

            entity.billboard = undefined;
            entity.point = new Cesium.PointGraphics({{
                pixelSize: 10,
                color: au_ppm > 0.5 ?
                    Cesium.Color.RED.withAlpha(0.8) :
                    Cesium.Color.YELLOW.withAlpha(0.6),
                outlineColor: Cesium.Color.BLACK,
                outlineWidth: 1,
                heightReference: Cesium.HeightReference.RELATIVE_TO_GROUND
            }});
        }});
        """

    def _generate_geological_overlays(self, geological_features: List[Path]) -> str:
        """Generate geological feature overlay code"""
        overlays = []

        for i, geo_map in enumerate(geological_features):
            overlays.append(f"""
        // Geological feature overlay {i+1}
        const geologyProvider{i} = new Cesium.SingleTileImageryProvider({{
            url: '{geo_map}',
            rectangle: Cesium.Rectangle.fromDegrees(-79.0, -4.3, -66.8, 12.5)
        }});

        const geologyLayer{i} = viewer.imageryLayers.addImageryProvider(geologyProvider{i});
        geologyLayer{i}.alpha = 0.5;
            """)

        return "\n".join(overlays)


class KeplerGLVisualizer:
    """Visualization using Kepler.gl for large-scale geospatial data"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def create_kepler_visualization(
        self,
        probability_maps: List[Path],
        borehole_data: pd.DataFrame,
        geological_features: List[Path],
        output_file: str = "kepler_visualization.html"
    ) -> Path:
        """
        Create Kepler.gl visualization

        Args:
            probability_maps: List of gold probability map paths
            borehole_data: DataFrame with borehole information
            geological_features: List of geological feature map paths
            output_file: Output HTML file name

        Returns:
            Path to generated visualization file
        """
        logger.info("Creating Kepler.gl visualization")

        # Generate Kepler.gl configuration
        kepler_config = self._generate_kepler_config(
            probability_maps,
            borehole_data,
            geological_features
        )

        # Save configuration
        config_path = self.output_dir / "kepler_config.json"
        with open(config_path, 'w') as f:
            json.dump(kepler_config, f, indent=2)

        logger.info(f"Kepler.gl configuration saved: {config_path}")
        return config_path

    def _generate_kepler_config(
        self,
        probability_maps: List[Path],
        borehole_data: pd.DataFrame,
        geological_features: List[Path]
    ) -> Dict:
        """Generate Kepler.gl configuration"""

        config = {
            "version": "v1",
            "config": {
                "visState": {
                    "layers": [
                        {
                            "id": "probability_heatmap",
                            "type": "heatmap",
                            "config": {
                                "dataId": "probability_data",
                                "label": "Gold Probability Heatmap",
                                "color": ["#ff0000", "#ffa500", "#ffff00", "#00ff00"],
                                "columns": {
                                    "lat": "lat",
                                    "lng": "lon",
                                    "weight": "probability"
                                },
                                "isVisible": True,
                                "visConfig": {
                                    "opacity": 0.8,
                                    "colorRange": {
                                        "name": "Custom",
                                        "type": "sequential",
                                        "colors": ["#ffff00", "#ffa500", "#ff0000"]
                                    }
                                }
                            }
                        },
                        {
                            "id": "borehole_points",
                            "type": "point",
                            "config": {
                                "dataId": "borehole_data",
                                "label": "Borehole Locations",
                                "color": ["#ff6b6b"],
                                "columns": {
                                    "lat": "lat",
                                    "lng": "lon"
                                },
                                "isVisible": True,
                                "visConfig": {
                                    "radius": 10,
                                    "fixedRadius": False,
                                    "opacity": 0.8,
                                    "colorRange": {
                                        "name": "colorbrewer2",
                                        "colors": ["#ff6b6b", "#4ecdc4"]
                                    }
                                }
                            }
                        }
                    ]
                }
            }
        }

        return config


class BoreholeCrossSectionVisualizer:
    """3D borehole cross-section visualization"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def create_cross_section_visualization(
        self,
        borehole_data: pd.DataFrame,
        output_file: str = "borehole_cross_sections.json"
    ) -> Path:
        """
        Create 3D borehole cross-section data for visualization

        Args:
            borehole_data: DataFrame with borehole information
            output_file: Output JSON file name

        Returns:
            Path to generated cross-section file
        """
        logger.info("Creating 3D borehole cross-section visualization")

        # Group boreholes by study area
        study_areas = borehole_data['study_area'].unique()

        cross_sections = {}

        for area in study_areas:
            area_boreholes = borehole_data[borehole_data['study_area'] == area]

            # Create cross-section data structure
            cross_section = {
                "study_area": area,
                "boreholes": [],
                "statistics": {
                    "total_boreholes": len(area_boreholes),
                    "avg_au_ppm": area_boreholes['au_ppm'].mean(),
                    "max_depth": area_boreholes['depth_m'].max(),
                    "gold_positive_rate": (area_boreholes['label_gold'] == 1).mean()
                }
            }

            # Add individual borehole data
            for _, borehole in area_boreholes.iterrows():
                borehole_data_3d = {
                    "id": borehole.name,
                    "coordinates": [borehole['lon'], borehole['lat']],
                    "depth_profile": self._create_depth_profile(borehole),
                    "lithology_sequence": self._extract_lithology_sequence(borehole),
                    "au_concentrations": self._get_au_concentrations(borehole)
                }
                cross_section["boreholes"].append(borehole_data_3d)

            cross_sections[area] = cross_section

        # Save cross-section data
        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(cross_sections, f, indent=2)

        logger.info(f"Borehole cross-sections saved: {output_path}")
        return output_path

    def _create_depth_profile(self, borehole) -> List[Dict]:
        """Create depth profile for 3D visualization"""
        depth = borehole.get('depth_m', 0)

        # Create simplified depth profile (in production, use actual depth data)
        profile = []
        for d in np.linspace(0, depth, 10):
            profile.append({
                "depth": float(d),
                "au_ppm": borehole.get('au_ppm', 0) * np.exp(-d / 20),  # Simplified decay
                "lithology": borehole.get('lithology_text', 'Unknown')
            })

        return profile

    def _extract_lithology_sequence(self, borehole) -> List[str]:
        """Extract lithology sequence for visualization"""
        lithology_text = borehole.get('lithology_text', '')

        # Simple keyword extraction for demo
        keywords = ['arcilla', 'arena', 'grava', 'granito', 'basalto']
        sequence = []

        for keyword in keywords:
            if keyword in lithology_text.lower():
                sequence.append(keyword.capitalize())

        return sequence if sequence else ['Unknown']

    def _get_au_concentrations(self, borehole) -> List[float]:
        """Get gold concentrations along borehole depth"""
        au_ppm = borehole.get('au_ppm', 0)
        depth = borehole.get('depth_m', 0)

        # Create concentration profile
        concentrations = []
        for d in np.linspace(0, depth, 10):
            # Simplified concentration profile
            conc = au_ppm * np.exp(-d / 15)  # Decay with depth
            concentrations.append(float(conc))

        return concentrations


def create_gold_probability_heatmaps(
    model_predictions: np.ndarray,
    coordinates: np.ndarray,
    output_dir: Path,
    resolution: float = 0.01
) -> List[Path]:
    """
    Create interactive gold probability heatmaps

    Args:
        model_predictions: Array of model predictions
        coordinates: Array of coordinate pairs (lon, lat)
        output_dir: Directory to save heatmaps
        resolution: Grid resolution for heatmap

    Returns:
        List of heatmap file paths
    """
    logger.info(f"Creating gold probability heatmaps for {len(model_predictions)} predictions")

    # Create grid for heatmap
    lon_min, lat_min = coordinates.min(axis=0)
    lon_max, lat_max = coordinates.max(axis=0)

    lon_bins = np.arange(lon_min, lon_max, resolution)
    lat_bins = np.arange(lat_min, lat_max, resolution)

    # Create probability grid
    prob_grid, _, _ = np.histogram2d(
        coordinates[:, 0], coordinates[:, 1],
        bins=[lon_bins, lat_bins],
        weights=model_predictions
    )

    # Normalize probabilities
    prob_grid = prob_grid / (prob_grid.max() + 1e-6)

    # Save heatmap as GeoTIFF
    heatmap_paths = []

    for i, threshold in enumerate([0.2, 0.5, 0.7, 0.9]):
        # Create thresholded heatmap
        thresholded_grid = (prob_grid >= threshold).astype('float32')

        output_file = output_dir / f"gold_probability_threshold_{threshold}.tif"

        # Create GeoTIFF with proper georeferencing
        from rasterio.transform import from_bounds

        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, thresholded_grid.shape[1], thresholded_grid.shape[0])

        with rasterio.open(
            output_file,
            'w',
            driver='GTiff',
            height=thresholded_grid.shape[0],
            width=thresholded_grid.shape[1],
            count=1,
            dtype='float32',
            crs='EPSG:4326',
            transform=transform
        ) as dst:
            dst.write(thresholded_grid, 1)

        heatmap_paths.append(output_file)
        logger.info(f"Created heatmap: {output_file}")

    logger.info(f"Gold probability heatmaps completed. Generated {len(heatmap_paths)} files")
    return heatmap_paths


def create_exploration_dashboard(
    probability_heatmaps: List[Path],
    borehole_cross_sections: Path,
    geological_maps: List[Path],
    output_dir: Path,
    dashboard_title: str = "GeoAuPredict - Gold Exploration Dashboard"
) -> Path:
    """
    Create comprehensive exploration dashboard

    Args:
        probability_heatmaps: List of probability heatmap paths
        borehole_cross_sections: Path to borehole cross-section data
        geological_maps: List of geological feature map paths
        output_dir: Directory to save dashboard
        dashboard_title: Title for the dashboard

    Returns:
        Path to generated dashboard file
    """
    logger.info("Creating comprehensive exploration dashboard")

    # Create dashboard configuration
    dashboard_config = {
        "title": dashboard_title,
        "version": "1.0",
        "layers": {
            "probability_heatmaps": probability_heatmaps,
            "borehole_cross_sections": str(borehole_cross_sections),
            "geological_maps": geological_maps
        },
        "visualization_options": {
            "heatmap_opacity": 0.7,
            "borehole_scale": 1.0,
            "geology_opacity": 0.5,
            "color_scheme": "viridis"
        },
        "exploration_criteria": {
            "min_probability": 0.7,
            "max_uncertainty": 0.3,
            "min_borehole_distance": 1000  # meters
        }
    }

    # Save dashboard configuration
    output_file = output_dir / "exploration_dashboard.json"
    with open(output_file, 'w') as f:
        json.dump(dashboard_config, f, indent=2)

    logger.info(f"Exploration dashboard configuration saved: {output_file}")
    return output_file
