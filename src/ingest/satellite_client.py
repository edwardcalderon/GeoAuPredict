#!/usr/bin/env python3
"""
Satellite Data Client for ObservEarth API
=========================================

This module provides a secure client for fetching satellite imagery data
from the ObservEarth API using credentials stored in environment variables.
"""

import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List

# Load environment variables from .env file
load_dotenv()


class SatelliteDataClient:
    """Secure client for satellite imagery APIs."""

    def __init__(self):
        """Initialize with API credentials from environment."""
        self.api_key = os.getenv('OBSERVEARTH_API_KEY')
        self.geometry_id = os.getenv('OBSERVEARTH_GEOMETRY_ID')

        if not self.api_key or not self.geometry_id:
            raise ValueError("❌ API credentials not found in .env file")

    def get_observearth_imagery(self,
                               item_id: str = "S2C_MSIL2A_20250320T045721_R119_T44QMG_20250320T103315",
                               image_type: str = "png",
                               index: str = "ndvi",
                               output_dir: Optional[Path] = None) -> Optional[bytes]:
        """
        Fetch satellite imagery from ObservEarth API.

        Args:
            item_id: Sentinel-2 item identifier
            image_type: Image format (png, tif, etc.)
            index: Spectral index (ndvi, clay_index, iron_index)
            output_dir: Directory to save images (default: data/external/satellite)

        Returns:
            Image data as bytes, or None if request fails
        """

        url = f"https://observearth.com/api/s2/image/{self.geometry_id}"

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        params = {
            "item_id": item_id,
            "image_type": image_type,
            "index": index
        }

        try:
            print(f"📡 Requesting {index} imagery for {item_id}...")
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()

            # Save image data if output directory is provided
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)

                filename = f"{index}_{item_id}.{image_type}"
                output_path = output_dir / filename

                with open(output_path, 'wb') as f:
                    f.write(response.content)

                print(f"✅ Saved imagery to: {output_path}")

            return response.content

        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None

    def get_multiple_indices(self,
                           item_id: str,
                           indices: Optional[List[str]] = None,
                           output_dir: Optional[Path] = None) -> Dict[str, Optional[bytes]]:
        """
        Fetch multiple spectral indices for the same satellite image.

        Args:
            item_id: Sentinel-2 item identifier
            indices: List of spectral indices to fetch (default: ['ndvi', 'clay_index', 'iron_index'])
            output_dir: Directory to save images

        Returns:
            Dictionary with index names as keys and image data as values
        """
        if indices is None:
            indices = ['ndvi', 'clay_index', 'iron_index']

        results = {}

        for index in indices:
            print(f"\n🔍 Processing {index.upper()}...")
            image_data = self.get_observearth_imagery(
                item_id,
                index=index,
                output_dir=output_dir
            )
            results[index] = image_data

        return results

    def get_available_indices(self) -> List[str]:
        """
        Get list of available spectral indices from the API.

        Returns:
            List of available index names
        """
        # This would typically make an API call to get available indices
        # For now, return the known indices
        return ['ndvi', 'clay_index', 'iron_index', 'evi', 'savi', 'msavi']

    def validate_credentials(self) -> bool:
        """
        Validate API credentials by making a test request.

        Returns:
            True if credentials are valid, False otherwise
        """
        try:
            # Make a minimal test request
            url = f"https://observearth.com/api/s2/image/{self.geometry_id}"
            headers = {"x-api-key": self.api_key}

            response = requests.get(url, headers=headers, params={"test": "true"})
            return response.status_code == 200
        except Exception:
            return False


def create_satellite_client() -> SatelliteDataClient:
    """
    Factory function to create a satellite client with proper error handling.

    Returns:
        Configured SatelliteDataClient instance

    Raises:
        ValueError: If API credentials are not configured
    """
    try:
        return SatelliteDataClient()
    except ValueError as e:
        print(f"❌ Failed to initialize satellite client: {e}")
        print("💡 Please configure OBSERVEARTH_API_KEY and OBSERVEARTH_GEOMETRY_ID in your .env file")
        raise
