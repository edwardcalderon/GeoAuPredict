#!/usr/bin/env python3
"""
Update VERSION_HISTORY.json with new version information
"""

import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
VERSION_HISTORY_PATH = PROJECT_ROOT / "VERSION_HISTORY.json"


def load_version_history():
    """Load existing version history"""
    if VERSION_HISTORY_PATH.exists():
        with open(VERSION_HISTORY_PATH, 'r') as f:
            return json.load(f)
    return {
        "current_version": "1.0.0",
        "versions": {},
        "metadata": {
            "project_name": "GeoAuPredict",
            "versioning_scheme": "semver"
        }
    }


def save_version_history(data):
    """Save version history to JSON"""
    data["metadata"]["last_updated"] = datetime.now().isoformat() + "Z"
    
    with open(VERSION_HISTORY_PATH, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Updated VERSION_HISTORY.json")


def add_version(version, version_data):
    """Add a new version to history"""
    history = load_version_history()
    
    # Update current version
    history["current_version"] = version
    
    # Add version data
    history["versions"][version] = version_data
    
    # Save
    save_version_history(history)
    print(f"✓ Added version {version} to history")


def update_model_performance(version, model_name, metrics):
    """Update performance metrics for a model"""
    history = load_version_history()
    
    if version not in history["versions"]:
        print(f"Error: Version {version} not found")
        return
    
    if "performance" not in history["versions"][version]:
        history["versions"][version]["performance"] = {"models": {}}
    
    history["versions"][version]["performance"]["models"][model_name] = metrics
    
    save_version_history(history)
    print(f"✓ Updated performance metrics for {model_name} in version {version}")


def get_version_comparison(version1, version2):
    """Compare two versions"""
    history = load_version_history()
    
    v1_data = history["versions"].get(version1, {})
    v2_data = history["versions"].get(version2, {})
    
    if not v1_data or not v2_data:
        print("Error: One or both versions not found")
        return None
    
    comparison = {
        "version1": version1,
        "version2": version2,
        "performance_diff": {}
    }
    
    # Compare performance metrics
    v1_perf = v1_data.get("performance", {}).get("models", {})
    v2_perf = v2_data.get("performance", {}).get("models", {})
    
    for model in set(list(v1_perf.keys()) + list(v2_perf.keys())):
        if model in v1_perf and model in v2_perf:
            comparison["performance_diff"][model] = {
                "auc_change": v2_perf[model].get("auc_roc", 0) - v1_perf[model].get("auc_roc", 0),
                "accuracy_change": v2_perf[model].get("accuracy", 0) - v1_perf[model].get("accuracy", 0)
            }
    
    return comparison


def print_version_info(version):
    """Print detailed information about a version"""
    history = load_version_history()
    
    if version not in history["versions"]:
        print(f"Error: Version {version} not found")
        return
    
    data = history["versions"][version]
    
    print("\n" + "="*60)
    print(f"Version {version} - {data.get('release_name', 'No name')}")
    print("="*60)
    
    print(f"\nRelease Date: {data.get('release_date', 'Unknown')}")
    print(f"Type: {data.get('type', 'Unknown')}")
    print(f"Status: {data.get('status', 'Unknown')}")
    
    if "changelog" in data:
        changelog = data["changelog"]
        
        if changelog.get("added"):
            print("\n### Added:")
            for item in changelog["added"]:
                print(f"  + {item}")
        
        if changelog.get("changed"):
            print("\n### Changed:")
            for item in changelog["changed"]:
                print(f"  * {item}")
        
        if changelog.get("fixed"):
            print("\n### Fixed:")
            for item in changelog["fixed"]:
                print(f"  ! {item}")
    
    if "models" in data:
        print("\n### Models:")
        for model, info in data["models"].items():
            status = info.get("status", "")
            print(f"  - {model}: {info.get('size_mb', '?')} MB ({status})")
    
    if "performance" in data and "models" in data["performance"]:
        print("\n### Performance:")
        for model, metrics in data["performance"]["models"].items():
            auc = metrics.get("auc_roc", 0)
            status = f" [{metrics.get('status', '')}]" if metrics.get("status") else ""
            print(f"  - {model}: AUC {auc:.4f}{status}")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python update_version_history.py show <version>")
        print("  python update_version_history.py compare <v1> <v2>")
        print("  python update_version_history.py list")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "show":
        if len(sys.argv) < 3:
            print("Error: version required")
            sys.exit(1)
        print_version_info(sys.argv[2])
    
    elif command == "compare":
        if len(sys.argv) < 4:
            print("Error: two versions required")
            sys.exit(1)
        comparison = get_version_comparison(sys.argv[2], sys.argv[3])
        if comparison:
            print(json.dumps(comparison, indent=2))
    
    elif command == "list":
        history = load_version_history()
        print(f"\nCurrent version: {history.get('current_version')}\n")
        print("Available versions:")
        for version in sorted(history["versions"].keys(), reverse=True):
            data = history["versions"][version]
            print(f"  - {version} ({data.get('release_date', 'Unknown')}) - {data.get('release_name', '')}")
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == '__main__':
    main()

