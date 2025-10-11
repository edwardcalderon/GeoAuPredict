# Whitepaper Versioning System

This system manages versions of the GeoAuPredict whitepaper PDF with automatic version tracking and download URL management.

## Structure

```
public/
└── versions/
    ├── versions.json          # Version metadata and configuration
    ├── whitepaper-v1.0.0.pdf # Version 1.0.0
    ├── whitepaper-v1.0.1.pdf # Version 1.0.1
    ├── whitepaper-v1.0.2.pdf # Version 1.0.2 (Latest)
    └── ...
```

## Usage

### Creating a New Version

When you update the whitepaper (LaTeX source), create a new version:

```bash
# Create new version (compiles PDF, converts to Markdown, and manages versions)
python scripts/version_manager.py create --version v1.0.3 --changes "Description of changes"
```

**What happens automatically:**
1. ✅ Compiles `docs/whitepaper.tex` to `public/whitepaper-latex.pdf`
2. ✅ Converts LaTeX to `docs/whitepaper.md` for web viewing
3. ✅ Creates versioned PDF in `public/versions/whitepaper-v{version}.pdf`
4. ✅ Updates `public/versions/versions.json` with metadata
5. ✅ Cleans up auxiliary LaTeX files

### Listing Versions

```bash
python scripts/version_manager.py list
```

### Cleanup Old Versions

```bash
# Keep only latest 5 versions
python scripts/version_manager.py cleanup --keep 5
```

## Integration

The download button in the whitepaper page automatically uses the latest version from `versions.json`. The download URL and filename are dynamically generated based on the current version.

## Version Format

- Versions follow semantic versioning: `v1.0.0`, `v1.1.0`, etc.
- Each version includes:
  - Version number
  - PDF filename
  - Creation date
  - Change description

## Files Updated

- `public/whitepaper-latex.pdf` - Always points to latest version (for backward compatibility)
- `public/versions/whitepaper-v{version}.pdf` - Individual version files
- `public/versions/versions.json` - Version metadata
