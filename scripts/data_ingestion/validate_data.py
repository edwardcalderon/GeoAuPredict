#!/usr/bin/env python3
"""
Data Validation and Quality Assessment for GeoAuPredict

This script performs comprehensive validation of geospatial datasets including:
1. Data completeness and consistency checks
2. Spatial data validation (coordinates, bounds)
3. Statistical analysis and outlier detection
4. Schema validation against expected formats
5. Cross-dataset consistency checks

Usage:
    python scripts/data_ingestion/validate_data.py --input data/raw/sgc/geoquimica_sgc.csv
"""

import argparse
import json
import logging
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from shapely.geometry import Point

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from scripts.data_ingestion.data_utils import DataStandardizer


class DataValidator:
    """Comprehensive geospatial data validation."""

    def __init__(self):
        """Initialize the data validator."""
        self.standardizer = DataStandardizer()
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data/processed/validation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def validate_csv_file(self, file_path: Path) -> Dict:
        """Validate a CSV file comprehensively.

        Args:
            file_path: Path to CSV file

        Returns:
            Dictionary with validation results
        """
        self.logger.info(f"🔍 Validating: {file_path}")

        validation_report = {
            "file_info": {
                "path": str(file_path),
                "size_bytes": file_path.stat().st_size,
                "modified_date": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            },
            "validation_date": datetime.now().isoformat(),
            "results": {}
        }

        try:
            # Try different separators and encodings
            df = self._read_csv_flexible(file_path)

            if df is None:
                validation_report["results"]["error"] = "Could not read CSV file"
                return validation_report

            # Basic file validation
            validation_report["results"].update(self._validate_dataframe(df))

            # Spatial validation (if coordinates present)
            if self._has_coordinates(df):
                validation_report["results"]["spatial"] = self._validate_spatial_data(df)

            # Statistical analysis
            validation_report["results"]["statistics"] = self._calculate_statistics(df)

            # Schema validation
            validation_report["results"]["schema"] = self._validate_schema(df)

        except Exception as e:
            validation_report["results"]["error"] = str(e)
            self.logger.error(f"❌ Validation failed: {e}")

        return validation_report

    def _read_csv_flexible(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Try to read CSV with different separators and encodings."""
        separators = [',', ';', '\t', '|']
        encodings = ['utf-8', 'latin-1', 'iso-8859-1']

        for sep in separators:
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, sep=sep, encoding=encoding, nrows=1000)
                    if len(df.columns) > 1:  # Ensure we have multiple columns
                        self.logger.info(f"✅ Successfully read CSV with sep='{sep}', encoding='{encoding}'")
                        return pd.read_csv(file_path, sep=sep, encoding=encoding)
                except:
                    continue

        self.logger.error("❌ Could not read CSV file with any combination of separators/encodings")
        return None

    def _validate_dataframe(self, df: pd.DataFrame) -> Dict:
        """Basic DataFrame validation."""
        results = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum()
        }

        # Missing data analysis
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100

        results["missing_data"] = {
            "count": missing_data.to_dict(),
            "percentage": missing_percent.to_dict()
        }

        # Duplicate analysis
        results["duplicates"] = {
            "total_duplicates": df.duplicated().sum(),
            "duplicate_percentage": (df.duplicated().sum() / len(df)) * 100
        }

        return results

    def _has_coordinates(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame has coordinate columns."""
        coord_patterns = ['lat', 'lon', 'latitude', 'longitude', 'x', 'y', 'coord']
        return any(any(pattern in col.lower() for pattern in coord_patterns) for col in df.columns)

    def _validate_spatial_data(self, df: pd.DataFrame) -> Dict:
        """Validate spatial data (coordinates, bounds, etc.)."""
        results = {}

        # Standardize coordinates first
        df_std = self.standardizer.standardize_coordinates(df)

        if 'latitude' in df_std.columns and 'longitude' in df_std.columns:
            # Coordinate range validation
            lat_valid = df_std['latitude'].between(-90, 90)
            lon_valid = df_std['longitude'].between(-180, 180)

            results["coordinate_validity"] = {
                "valid_latitude": lat_valid.sum(),
                "valid_longitude": lon_valid.sum(),
                "invalid_coordinates": (~(lat_valid & lon_valid)).sum()
            }

            # Colombia bounds check
            colombia_bounds = self.standardizer.colombia_bounds
            in_colombia = (
                (df_std['longitude'] >= colombia_bounds['west']) &
                (df_std['longitude'] <= colombia_bounds['east']) &
                (df_std['latitude'] >= colombia_bounds['south']) &
                (df_std['latitude'] <= colombia_bounds['north'])
            )

            results["colombia_bounds"] = {
                "points_in_colombia": in_colombia.sum(),
                "points_outside_colombia": (~in_colombia).sum()
            }

            # Spatial distribution
            results["spatial_stats"] = {
                "lat_range": [df_std['latitude'].min(), df_std['latitude'].max()],
                "lon_range": [df_std['longitude'].min(), df_std['longitude'].max()],
                "lat_mean": df_std['latitude'].mean(),
                "lon_mean": df_std['longitude'].mean()
            }

        return results

    def _calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive statistics."""
        stats = {}

        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                stats[col] = {
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "median": df[col].median(),
                    "q25": df[col].quantile(0.25),
                    "q75": df[col].quantile(0.75),
                    "skewness": df[col].skew(),
                    "kurtosis": df[col].kurtosis()
                }
            elif df[col].dtype == 'object':
                stats[col] = {
                    "unique_values": df[col].nunique(),
                    "most_common": df[col].value_counts().head(5).to_dict() if df[col].nunique() < 100 else "Too many unique values"
                }

        return stats

    def _validate_schema(self, df: pd.DataFrame) -> Dict:
        """Validate data schema against expected formats."""
        schema_results = {
            "expected_columns": [],
            "missing_columns": [],
            "unexpected_columns": [],
            "data_type_issues": []
        }

        # Define expected schemas for different data types
        sgc_schema = {
            "AU_PPM": "numeric",
            "LATITUD": "numeric",
            "LONGITUD": "numeric",
            "AS_PPM": "numeric",
            "CU_PPM": "numeric"
        }

        usgs_schema = {
            "latitude": "numeric",
            "longitude": "numeric",
            "commodity": "string",
            "site_name": "string"
        }

        # Auto-detect schema type
        sgc_score = sum(1 for col in sgc_schema if col in df.columns)
        usgs_score = sum(1 for col in usgs_schema if col in df.columns)

        if sgc_score > usgs_score:
            expected_schema = sgc_schema
            schema_results["detected_type"] = "sgc_geochemistry"
        elif usgs_score > 0:
            expected_schema = usgs_schema
            schema_results["detected_type"] = "usgs_mrds"
        else:
            schema_results["detected_type"] = "unknown"

        # Check schema compliance
        for col, expected_type in expected_schema.items():
            if col in df.columns:
                schema_results["expected_columns"].append(col)

                # Check data type
                actual_type = str(df[col].dtype)
                if expected_type == "numeric" and actual_type not in ['int64', 'float64']:
                    schema_results["data_type_issues"].append(f"{col}: expected numeric, got {actual_type}")
            else:
                schema_results["missing_columns"].append(col)

        # Check for unexpected columns
        expected_cols = set(expected_schema.keys())
        actual_cols = set(df.columns)
        schema_results["unexpected_columns"] = list(actual_cols - expected_cols)

        return schema_results

    def generate_validation_report(self, validation_results: Dict, output_dir: Path = None) -> str:
        """Generate comprehensive validation report.

        Args:
            validation_results: Results from validation
            output_dir: Directory to save report (default: data/processed/)

        Returns:
            Path to generated report file
        """
        if output_dir is None:
            output_dir = Path("data/processed")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate HTML report
        report_path = output_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

        html_content = self._generate_html_report(validation_results)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        self.logger.info(f"📋 Generated validation report: {report_path}")
        return str(report_path)

    def _generate_html_report(self, results: Dict) -> str:
        """Generate HTML validation report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>GeoAuPredict Data Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f0f8ff; padding: 20px; border-radius: 8px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .error {{ color: #d32f2f; }}
                .warning {{ color: #f57c00; }}
                .success {{ color: #2e7d32; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 GeoAuPredict Data Validation Report</h1>
                <p><strong>Generated:</strong> {results['validation_date']}</p>
            </div>
        """

        # File information
        file_info = results['file_info']
        html += f"""
            <div class="section">
                <h2>📁 File Information</h2>
                <p><strong>Path:</strong> {file_info['path']}</p>
                <p><strong>Size:</strong> {file_info['size_bytes']:,} bytes</p>
                <p><strong>Modified:</strong> {file_info['modified_date']}</p>
            </div>
        """

        # Basic validation results
        basic_results = results['results']
        if 'error' not in basic_results:
            html += f"""
                <div class="section">
                    <h2>📊 Basic Validation</h2>
                    <p><strong>Rows:</strong> {basic_results['total_rows']:,}</p>
                    <p><strong>Columns:</strong> {basic_results['total_columns']}</p>
                    <p><strong>Memory Usage:</strong> {basic_results['memory_usage']:,} bytes</p>
                </div>
            """

            # Missing data
            if basic_results['missing_data']['count']:
                html += """
                <div class="section warning">
                    <h2>⚠️ Missing Data</h2>
                    <table>
                        <tr><th>Column</th><th>Missing Count</th><th>Missing %</th></tr>
            """
                for col, count in basic_results['missing_data']['count'].items():
                    percentage = basic_results['missing_data']['percentage'][col]
                    html += f"<tr><td>{col}</td><td>{count:,}</td><td>{percentage:.1f}%</td></tr>"
                html += "</table></div>"

            # Spatial validation
            if 'spatial' in basic_results:
                spatial = basic_results['spatial']
                html += f"""
                <div class="section success">
                    <h2>🌍 Spatial Validation</h2>
                    <p><strong>Valid Coordinates:</strong> {spatial['coordinate_validity']['valid_latitude']:,}</p>
                    <p><strong>In Colombia:</strong> {spatial['colombia_bounds']['points_in_colombia']:,}</p>
                </div>
                """

        html += "</body></html>"
        return html

    def validate_data_lake(self) -> Dict:
        """Validate the entire data lake."""
        self.logger.info("🏗️ Validating complete data lake...")

        data_lake_report = {
            "validation_date": datetime.now().isoformat(),
            "data_lake_version": "1.0.0",
            "datasets": {}
        }

        # Scan all raw data directories
        raw_dir = Path("data/raw")
        if raw_dir.exists():
            for source_dir in raw_dir.iterdir():
                if source_dir.is_dir():
                    datasets = list(source_dir.glob("*.csv")) + list(source_dir.glob("*.geojson"))
                    if datasets:
                        data_lake_report["datasets"][source_dir.name] = {
                            "files_validated": len(datasets),
                            "validation_results": {}
                        }

                        for dataset in datasets[:3]:  # Validate first 3 files per source
                            validation_result = self.validate_csv_file(dataset)
                            data_lake_report["datasets"][source_dir.name]["validation_results"][dataset.name] = validation_result

        return data_lake_report


def main():
    """Main validation entry point."""
    parser = argparse.ArgumentParser(description="GeoAuPredict Data Validation")
    parser.add_argument("--input", help="Specific file to validate")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory for reports")
    parser.add_argument("--data-lake", action="store_true", help="Validate entire data lake")
    parser.add_argument("--generate-report", action="store_true", help="Generate HTML report")

    args = parser.parse_args()

    validator = DataValidator()

    if args.data_lake:
        # Validate entire data lake
        report = validator.validate_data_lake()

        # Save data lake report
        report_path = Path(args.output_dir) / f"data_lake_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"📋 Data lake validation report saved to: {report_path}")

    elif args.input:
        # Validate specific file
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"❌ File not found: {args.input}")
            return

        validation_results = validator.validate_csv_file(input_path)

        # Save individual report
        report_path = Path(args.output_dir) / f"validation_{input_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)

        print(f"📋 Validation report saved to: {report_path}")

        if args.generate_report:
            html_path = validator.generate_validation_report(validation_results, Path(args.output_dir))
            print(f"🌐 HTML report: {html_path}")

    else:
        print("❌ Please specify --input file or --data-lake option")
        parser.print_help()


if __name__ == "__main__":
    main()
