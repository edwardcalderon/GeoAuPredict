"""
Terrain Variables Extraction Module for Phase 2 Geospatial Feature Engineering

This module extracts topographic variables from Digital Elevation Models (DEM):
- Altitude/Elevation
- Slope (degrees and percent)
- Curvature (planform and profile)
- Aspect
- Topographic Wetness Index (TWI)
- Flow accumulation

Uses GDAL and richdem for efficient terrain analysis.
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
import logging
import warnings
from typing import Dict, Tuple, Optional, Union

# Try importing terrain analysis libraries
try:
    import richdem as rd
    RICHDERM_AVAILABLE = True
except ImportError:
    RICHDERM_AVAILABLE = False
    warnings.warn("richdem not available. Install with: pip install richdem")

try:
    from osgeo import gdal
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False
    warnings.warn("GDAL not available. Install with: pip install gdal")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TerrainAnalyzer:
    """Analyzer for extracting terrain variables from DEM data."""

    def __init__(self, use_richdem: bool = True):
        """
        Initialize the terrain analyzer.

        Args:
            use_richdem: Whether to use richdem for calculations (faster, more accurate)
        """
        self.use_richdem = use_richdem and RICHDERM_AVAILABLE

        if self.use_richdem:
            logger.info("Using richdem for terrain analysis")
        elif GDAL_AVAILABLE:
            logger.info("Using GDAL for terrain analysis")
        else:
            logger.warning("Neither richdem nor GDAL available. Using basic numpy operations.")

    def load_dem(self, dem_path: str) -> Tuple[np.ndarray, dict]:
        """
        Load DEM data from file.

        Args:
            dem_path: Path to DEM file

        Returns:
            Tuple of (dem_array, profile)
        """
        if not Path(dem_path).exists():
            raise FileNotFoundError(f"DEM file not found: {dem_path}")

        try:
            with rasterio.open(dem_path) as src:
                dem_array = src.read(1).astype(np.float32)
                profile = src.profile.copy()
                logger.info(f"Loaded DEM: {dem_array.shape}, CRS: {src.crs}")

                # Handle nodata values
                if src.nodata is not None:
                    dem_array = np.where(dem_array == src.nodata, np.nan, dem_array)

                return dem_array, profile

        except Exception as e:
            logger.error(f"Error loading DEM: {e}")
            raise

    def calculate_slope(self, dem: np.ndarray, method: str = 'degrees') -> np.ndarray:
        """
        Calculate slope from DEM.

        Args:
            dem: Digital elevation model array
            method: 'degrees' or 'percent'

        Returns:
            Slope array
        """
        logger.info(f"Calculating slope ({method})...")

        if self.use_richdem and RICHDERM_AVAILABLE:
            # Use richdem for accurate slope calculation
            dem_rd = rd.rdarray(dem, no_data=np.nan)
            slope_rd = rd.TerrainAttribute(dem_rd, attrib='slope_degrees')

            if method == 'percent':
                slope_rd = np.tan(np.radians(slope_rd)) * 100

            slope = slope_rd.copy()
            slope[np.isnan(slope)] = -9999

        else:
            # Use simple finite difference method
            # Get DEM dimensions
            rows, cols = dem.shape

            # Create slope array
            slope = np.zeros_like(dem, dtype=np.float32)

            # Calculate slope using 3x3 window
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    if np.isnan(dem[i, j]):
                        slope[i, j] = -9999
                        continue

                    # Get 3x3 neighborhood
                    window = dem[i-1:i+2, j-1:j+2]

                    # Calculate dz/dx and dz/dy using central differences
                    dz_dx = (window[1, 2] - window[1, 0]) / 2
                    dz_dy = (window[2, 1] - window[0, 1]) / 2

                    # Calculate slope in degrees
                    slope_val = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))

                    if method == 'percent':
                        slope_val = np.tan(np.radians(slope_val)) * 100

                    slope[i, j] = slope_val

            # Set edges to nodata
            slope[0, :] = -9999
            slope[-1, :] = -9999
            slope[:, 0] = -9999
            slope[:, -1] = -9999

        logger.info(f"Slope range: {slope.min():.3f} to {slope.max():.3f}")
        return slope

    def calculate_curvature(self, dem: np.ndarray, curvature_type: str = 'planform') -> np.ndarray:
        """
        Calculate curvature from DEM.

        Args:
            dem: Digital elevation model array
            curvature_type: 'planform', 'profile', or 'mean'

        Returns:
            Curvature array
        """
        logger.info(f"Calculating {curvature_type} curvature...")

        if self.use_richdem and RICHDERM_AVAILABLE:
            # Use richdem for accurate curvature calculation
            dem_rd = rd.rdarray(dem, no_data=np.nan)

            if curvature_type == 'planform':
                curvature_rd = rd.TerrainAttribute(dem_rd, attrib='plan_curvature')
            elif curvature_type == 'profile':
                curvature_rd = rd.TerrainAttribute(dem_rd, attrib='profile_curvature')
            elif curvature_type == 'mean':
                curvature_rd = rd.TerrainAttribute(dem_rd, attrib='mean_curvature')
            else:
                raise ValueError(f"Unknown curvature type: {curvature_type}")

            curvature = curvature_rd.copy()
            curvature[np.isnan(curvature)] = -9999

        else:
            # Use simple finite difference method
            rows, cols = dem.shape
            curvature = np.zeros_like(dem, dtype=np.float32)

            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    if np.isnan(dem[i, j]):
                        curvature[i, j] = -9999
                        continue

                    # Get 3x3 neighborhood
                    window = dem[i-1:i+2, j-1:j+2]

                    # Calculate second derivatives for curvature
                    d2z_dx2 = (window[1, 2] - 2*window[1, 1] + window[1, 0])
                    d2z_dy2 = (window[2, 1] - 2*window[1, 1] + window[0, 1])
                    d2z_dxdy = (window[2, 2] - window[2, 0] - window[0, 2] + window[0, 0]) / 4

                    # Calculate mean curvature
                    curvature_val = (d2z_dx2 + d2z_dy2) / 2

                    if curvature_type == 'planform':
                        curvature_val = d2z_dx2
                    elif curvature_type == 'profile':
                        curvature_val = d2z_dy2

                    curvature[i, j] = curvature_val

            # Set edges to nodata
            curvature[0, :] = -9999
            curvature[-1, :] = -9999
            curvature[:, 0] = -9999
            curvature[:, -1] = -9999

        logger.info(f"Curvature range: {curvature.min():.3f} to {curvature.max():.3f}")
        return curvature

    def calculate_aspect(self, dem: np.ndarray) -> np.ndarray:
        """
        Calculate aspect (direction of slope) from DEM.

        Args:
            dem: Digital elevation model array

        Returns:
            Aspect array (degrees from north, clockwise)
        """
        logger.info("Calculating aspect...")

        if self.use_richdem and RICHDERM_AVAILABLE:
            # Use richdem for accurate aspect calculation
            dem_rd = rd.rdarray(dem, no_data=np.nan)
            aspect_rd = rd.TerrainAttribute(dem_rd, attrib='aspect')

            aspect = aspect_rd.copy()
            aspect[np.isnan(aspect)] = -9999

        else:
            # Use simple finite difference method
            rows, cols = dem.shape
            aspect = np.zeros_like(dem, dtype=np.float32)

            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    if np.isnan(dem[i, j]):
                        aspect[i, j] = -9999
                        continue

                    # Get 3x3 neighborhood
                    window = dem[i-1:i+2, j-1:j+2]

                    # Calculate dz/dx and dz/dy
                    dz_dx = (window[1, 2] - window[1, 0]) / 2
                    dz_dy = (window[2, 1] - window[0, 1]) / 2

                    # Calculate aspect (direction of steepest descent)
                    aspect_val = np.degrees(np.arctan2(dz_dx, dz_dy))

                    # Convert to 0-360 degrees (from north, clockwise)
                    aspect_val = (aspect_val + 360) % 360

                    aspect[i, j] = aspect_val

            # Set edges to nodata
            aspect[0, :] = -9999
            aspect[-1, :] = -9999
            aspect[:, 0] = -9999
            aspect[:, -1] = -9999

        logger.info(f"Aspect range: {aspect.min():.3f} to {aspect.max():.3f}")
        return aspect

    def calculate_twi(self, dem: np.ndarray, flow_accumulation: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate Topographic Wetness Index (TWI).

        TWI = ln(As / tan(Slope))

        Args:
            dem: Digital elevation model array
            flow_accumulation: Pre-calculated flow accumulation (optional)

        Returns:
            TWI array
        """
        logger.info("Calculating Topographic Wetness Index...")

        # Calculate slope in radians for TWI
        slope_deg = self.calculate_slope(dem, method='degrees')
        slope_rad = np.radians(slope_deg)

        # Calculate flow accumulation if not provided
        if flow_accumulation is None:
            flow_accumulation = self.calculate_flow_accumulation(dem)

        # Calculate TWI
        with np.errstate(divide='ignore', invalid='ignore'):
            tan_slope = np.tan(slope_rad)
            tan_slope = np.where(tan_slope == 0, 1e-6, tan_slope)  # Avoid division by zero

            twi = np.log(flow_accumulation / tan_slope)
            twi = np.where(np.isinf(twi) | np.isnan(twi), -9999, twi)

        logger.info(f"TWI range: {twi.min():.3f} to {twi.max():.3f}")
        return twi

    def calculate_flow_accumulation(self, dem: np.ndarray) -> np.ndarray:
        """
        Calculate flow accumulation from DEM.

        Args:
            dem: Digital elevation model array

        Returns:
            Flow accumulation array
        """
        logger.info("Calculating flow accumulation...")

        if self.use_richdem and RICHDERM_AVAILABLE:
            # Use richdem for accurate flow accumulation
            dem_rd = rd.rdarray(dem, no_data=np.nan)
            flowacc_rd = rd.FlowAccumulation(dem_rd, method='Dinf')

            flowacc = flowacc_rd.copy()
            flowacc[np.isnan(flowacc)] = -9999

        else:
            # Simple flow accumulation using D8 method
            rows, cols = dem.shape
            flowacc = np.zeros_like(dem, dtype=np.float32)

            # Calculate slope and aspect for flow direction
            slope = self.calculate_slope(dem, method='degrees')
            aspect = self.calculate_aspect(dem)

            # Simple flow accumulation (very basic implementation)
            # This is a simplified version - for production use richdem or GDAL
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    if np.isnan(dem[i, j]) or slope[i, j] <= 0:
                        flowacc[i, j] = 1  # Minimum flow
                        continue

                    # Simple accumulation based on neighboring cells
                    neighbors = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < rows and 0 <= nj < cols:
                                if dem[ni, nj] < dem[i, j]:  # Lower elevation
                                    neighbors.append(1)
                                else:
                                    neighbors.append(0)

                    flowacc[i, j] = sum(neighbors) + 1

            # Set edges to nodata
            flowacc[0, :] = -9999
            flowacc[-1, :] = -9999
            flowacc[:, 0] = -9999
            flowacc[:, -1] = -9999

        logger.info(f"Flow accumulation range: {flowacc.min():.3f} to {flowacc.max():.3f}")
        return flowacc

    def calculate_all_terrain_variables(self, dem_path: str) -> Dict[str, np.ndarray]:
        """
        Calculate all terrain variables from DEM.

        Args:
            dem_path: Path to DEM file

        Returns:
            Dictionary of all terrain variables
        """
        logger.info(f"Starting terrain analysis for {dem_path}...")

        # Load DEM
        dem, profile = self.load_dem(dem_path)

        # Calculate all variables
        terrain_vars = {}

        # Basic elevation
        terrain_vars['elevation'] = dem.copy()

        # Slope
        terrain_vars['slope_degrees'] = self.calculate_slope(dem, method='degrees')
        terrain_vars['slope_percent'] = self.calculate_slope(dem, method='percent')

        # Curvature
        terrain_vars['plan_curvature'] = self.calculate_curvature(dem, curvature_type='planform')
        terrain_vars['profile_curvature'] = self.calculate_curvature(dem, curvature_type='profile')
        terrain_vars['mean_curvature'] = self.calculate_curvature(dem, curvature_type='mean')

        # Aspect
        terrain_vars['aspect'] = self.calculate_aspect(dem)

        # Topographic indices
        terrain_vars['flow_accumulation'] = self.calculate_flow_accumulation(dem)
        terrain_vars['twi'] = self.calculate_twi(dem, terrain_vars['flow_accumulation'])

        logger.info(f"Successfully calculated {len(terrain_vars)} terrain variables")
        return terrain_vars, profile

    def save_terrain_variables(self, terrain_vars: Dict[str, np.ndarray],
                             output_dir: str, profile: dict) -> Dict[str, str]:
        """
        Save terrain variables as GeoTIFF files.

        Args:
            terrain_vars: Dictionary of terrain variable arrays
            output_dir: Output directory path
            profile: Rasterio profile for consistent output

        Returns:
            Dictionary mapping variable names to saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        for var_name, var_array in terrain_vars.items():
            output_file = output_path / f"{var_name}.tif"

            # Update profile for this variable
            var_profile = profile.copy()
            var_profile['count'] = 1
            var_profile['nodata'] = -9999

            try:
                with rasterio.open(output_file, 'w', **var_profile) as dst:
                    # Handle nodata values
                    write_array = np.where(np.isnan(var_array), -9999, var_array)
                    dst.write(write_array.astype(np.float32), 1)

                saved_files[var_name] = str(output_file)
                logger.info(f"Saved {var_name} to {output_file}")

            except Exception as e:
                logger.error(f"Error saving {var_name}: {e}")
                continue

        logger.info(f"Saved {len(saved_files)} terrain variables")
        return saved_files


def create_sample_dem(output_dir: str = "sample_data") -> str:
    """
    Create sample DEM data for testing purposes.

    Args:
        output_dir: Directory to save sample data

    Returns:
        Path to created DEM file
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Create synthetic DEM with realistic terrain features
    rows, cols = 100, 100
    dem_data = np.zeros((rows, cols), dtype=np.float32)

    # Create base elevation with some realistic features
    x = np.linspace(0, 1, cols)
    y = np.linspace(0, 1, rows)
    X, Y = np.meshgrid(x, y)

    # Add some hills and valleys
    dem_data += 1000 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)  # Base terrain
    dem_data += 500 * np.exp(-((X-0.3)**2 + (Y-0.7)**2) / 0.1)  # Hill
    dem_data += 300 * np.exp(-((X-0.8)**2 + (Y-0.2)**2) / 0.05)  # Another hill
    dem_data += -200 * np.exp(-((X-0.5)**2 + (Y-0.5)**2) / 0.2)  # Valley

    # Add some noise for realism
    np.random.seed(42)
    dem_data += np.random.normal(0, 50, (rows, cols))

    # Ensure positive elevations
    dem_data = np.maximum(dem_data, 0)

    # Save DEM
    dem_file = output_path / "sample_dem.tif"
    with rasterio.open(
        dem_file, 'w',
        driver='GTiff',
        height=rows,
        width=cols,
        count=1,
        dtype='float32',
        crs='EPSG:4326',
        transform=from_bounds(0, 0, 1, 1, cols, rows)
    ) as dst:
        dst.write(dem_data, 1)

    logger.info(f"Created sample DEM: {dem_file}")
    return str(dem_file)


# Example usage
if __name__ == "__main__":
    # Create sample DEM for testing
    dem_file = create_sample_dem()

    # Calculate terrain variables
    analyzer = TerrainAnalyzer()
    terrain_vars, profile = analyzer.calculate_all_terrain_variables(dem_file)

    print("Calculated terrain variables:")
    for name, data in terrain_vars.items():
        valid_data = data[data != -9999]
        if len(valid_data) > 0:
            print(f"  {name}: {data.shape}, range: [{valid_data.min():.3f}, {valid_data.max():.3f}]")

    # Save terrain variables
    saved_files = analyzer.save_terrain_variables(terrain_vars, "output_terrain", profile)
    print(f"Saved {len(saved_files)} terrain variables to output_terrain/")
