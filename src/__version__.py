"""
GeoAuPredict Version Information
Loads version data from VERSION_HISTORY.json
"""

import json
from pathlib import Path

__version__ = "1.0.1"
__version_info__ = (1, 0, 1)

# Path to version history JSON
VERSION_HISTORY_PATH = Path(__file__).parent.parent / "VERSION_HISTORY.json"


def _load_version_history():
    """Load version history from JSON file"""
    if VERSION_HISTORY_PATH.exists():
        with open(VERSION_HISTORY_PATH, 'r') as f:
            return json.load(f)
    return {"versions": {}, "current_version": __version__}


def get_version():
    """Return current version string"""
    return __version__


def get_version_info():
    """Return version info tuple (major, minor, patch)"""
    return __version_info__


def get_version_history():
    """Return complete version history"""
    return _load_version_history()


def get_current_version_data():
    """Return data for current version"""
    history = _load_version_history()
    return history.get("versions", {}).get(__version__, {})


def get_latest_changes():
    """Return latest version changes"""
    version_data = get_current_version_data()
    return version_data.get("changelog", {})


def get_model_info():
    """Return current model registry info"""
    version_data = get_current_version_data()
    return version_data.get("models", {})


def get_performance_metrics():
    """Return performance metrics for current version"""
    version_data = get_current_version_data()
    return version_data.get("performance", {})


def get_deployment_info():
    """Return deployment information"""
    version_data = get_current_version_data()
    return version_data.get("deployment", {})

