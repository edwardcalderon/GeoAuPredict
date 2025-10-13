# 📦 GeoAuPredict Versioning Guide

## Overview

GeoAuPredict uses **Semantic Versioning 2.0.0** (semver) for all releases.

**Current Version**: 1.0.0 "Gold Rush"

---

## Version Format

```
MAJOR.MINOR.PATCH
  |     |     |
  |     |     └─ Bug fixes, patches (backwards compatible)
  |     └─────── New features (backwards compatible)
  └───────────── Breaking changes (not backwards compatible)
```

### Examples
- `1.0.0` → `1.0.1`: Bug fix (patch)
- `1.0.1` → `1.1.0`: New feature (minor)
- `1.1.0` → `2.0.0`: Breaking change (major)

---

## Files Involved

### 1. `VERSION`
Simple text file with current version:
```
1.0.0
```

### 2. `VERSION_HISTORY.json`
Complete version history with:
- Release dates and names
- Changelog (added, changed, fixed, removed)
- Model registry per version
- Performance metrics
- Deployment information

### 3. `src/__version__.py`
Python module for programmatic version access:
```python
from __version__ import get_version, get_performance_metrics
print(f"Version: {get_version()}")
```

### 4. `package.json`
Node.js package version (for web components):
```json
{
  "name": "geoaupredict",
  "version": "1.0.0",
  ...
}
```

### 5. `CHANGELOG.md`
Human-readable changelog following [Keep a Changelog](https://keepachangelog.com/)

---

## Bumping Versions

### Automatic (Recommended)

Use the automated script:

```bash
# Patch release (1.0.0 → 1.0.1)
npm run version:patch
# or
python scripts/bump_version.py patch

# Minor release (1.0.0 → 1.1.0)
npm run version:minor
# or
python scripts/bump_version.py minor

# Major release (1.0.0 → 2.0.0)
npm run version:major
# or
python scripts/bump_version.py major
```

The script will:
1. ✅ Update `VERSION` file
2. ✅ Update `src/__version__.py`
3. ✅ Update `package.json`
4. ✅ Prompt for changelog entry
5. ✅ Update `CHANGELOG.md`
6. ✅ Optionally create git tag

### Manual

1. Edit `VERSION`:
   ```
   1.0.1
   ```

2. Edit `src/__version__.py`:
   ```python
   __version__ = "1.0.1"
   __version_info__ = (1, 0, 1)
   ```

3. Edit `package.json`:
   ```json
   "version": "1.0.1"
   ```

4. Update `VERSION_HISTORY.json` (add new version entry)

5. Update `CHANGELOG.md`

---

## Version History Management

### View Current Version
```bash
python scripts/update_version_history.py list
```

### View Version Details
```bash
python scripts/update_version_history.py show 1.0.0
```

### Compare Versions
```bash
python scripts/update_version_history.py compare 1.0.0 1.1.0
```

---

## When to Bump Versions

### Patch (1.0.X)
- Bug fixes
- Documentation updates
- Performance improvements (no API changes)
- Security patches

### Minor (1.X.0)
- New features (backwards compatible)
- New model releases
- New API endpoints
- Deprecations (with backwards compatibility)

### Major (X.0.0)
- Breaking API changes
- Removed deprecated features
- Major architecture changes
- Non-backwards compatible model changes

---

## Release Process

### 1. Prepare Release

```bash
# Bump version
npm run version:minor  # or patch/major

# Review changes
git diff
```

### 2. Commit Changes

```bash
git add -A
git commit -m "chore: bump version to 1.1.0"
```

### 3. Create Git Tag

```bash
git tag -a v1.1.0 -m "Release v1.1.0 - Release Name

## Highlights
- Feature 1
- Feature 2

See CHANGELOG.md for details."
```

### 4. Push to GitHub

```bash
# Push commits
git push origin main

# Push tags
git push origin v1.1.0
```

### 5. Create GitHub Release

Go to: https://github.com/edwardcalderon/GeoAuPredict/releases/new

- Tag: `v1.1.0`
- Title: `v1.1.0 - Release Name`
- Description: Copy from CHANGELOG.md
- Attach: Model files (if applicable)

---

## Model Versioning

When releasing new models:

### 1. Train and Evaluate
```bash
python src/models/ensemble_comparison.py
```

### 2. Update VERSION_HISTORY.json
Add model information to the version entry:
```json
"models": {
  "ensemble_gold_v2.pkl": {
    "size_mb": 1.8,
    "type": "voting_ensemble",
    "status": "production"
  }
}
```

### 3. Update Performance Metrics
```json
"performance": {
  "models": {
    "voting_ensemble": {
      "auc_roc": 0.9250,
      "status": "improved"
    }
  }
}
```

### 4. Bump Version
- Same model, better performance → Patch
- New model architecture → Minor
- Breaking model API changes → Major

---

## Programmatic Access

### In Python

```python
from src.__version__ import (
    get_version,
    get_version_history,
    get_performance_metrics,
    get_model_info,
    get_deployment_info
)

# Get version
print(f"Version: {get_version()}")

# Get performance metrics
metrics = get_performance_metrics()
print(f"AUC: {metrics['models']['voting_ensemble']['auc_roc']}")

# Get model info
models = get_model_info()
for model, info in models.items():
    print(f"{model}: {info['size_mb']} MB ({info['status']})")
```

### In Notebooks

```python
from __version__ import get_version
print(f"Running GeoAuPredict v{get_version()}")
```

### In API

```python
from fastapi import FastAPI
from __version__ import get_version

@app.get("/version")
def version():
    return {"version": get_version()}
```

---

## Best Practices

### ✅ DO
- Bump version for every release
- Update CHANGELOG.md
- Create git tags for releases
- Test thoroughly before releasing
- Document breaking changes clearly
- Keep VERSION_HISTORY.json up-to-date

### ❌ DON'T
- Skip versions (1.0.0 → 1.0.2)
- Reuse version numbers
- Release without changelog
- Change version manually in multiple places
- Release untested code

---

## Troubleshooting

### Version Mismatch
If different files show different versions:
```bash
# Check all version locations
cat VERSION
grep version package.json
python -c "from src.__version__ import get_version; print(get_version())"

# Fix using bump script
python scripts/bump_version.py patch
```

### Git Tag Issues
```bash
# List all tags
git tag -l

# Delete local tag
git tag -d v1.0.0

# Delete remote tag
git push origin :refs/tags/v1.0.0

# Create new tag
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

---

## Version 1.0.0 Details

**Release Name**: "Gold Rush"  
**Date**: 2025-10-13  
**Status**: Stable  

### Highlights
- ✅ Voting Ensemble (production): AUC 0.9208
- ✅ Stacking Ensemble (alternative): AUC 0.9206
- ✅ 5 models in registry
- ✅ Live deployment on Render.com
- ✅ Comprehensive versioning system

See `VERSION_HISTORY.json` or `CHANGELOG.md` for complete details.

---

**Last Updated**: 2025-10-13  
**Maintained By**: Edward Calderón

