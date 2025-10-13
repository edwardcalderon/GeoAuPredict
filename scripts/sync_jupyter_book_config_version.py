#!/usr/bin/env python3
"""
Sync Jupyter Book _config.yml with the latest whitepaper release timestamp and version.

Sets the line to a human-friendly format:
  Month DD, YYYY [HH:MM] vX.Y.Z   (time included only if available)

Source of truth: public/versions/whitepaper-version.json
"""

import json
import re
from datetime import datetime
from pathlib import Path


def load_whitepaper_manifest(manifest_path: Path) -> dict:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def get_current_release_timestamp_and_version(manifest: dict) -> tuple[str, str]:
    current_version = manifest.get("currentVersion")
    if not current_version:
        raise ValueError("currentVersion not found in whitepaper-version.json")

    release_timestamp = None
    for version_entry in manifest.get("versions", []):
        if version_entry.get("version") == current_version:
            # Prefer full timestamp if available; fallback to date
            release_timestamp = (
                version_entry.get("timestamp")
                or version_entry.get("date")
            )
            break

    if not release_timestamp:
        # Ultimate fallback: now (ISO-like without timezone)
        release_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Track whether original was date-only
    date_only = bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", release_timestamp))
    # Normalize date-only to include zeroed time if needed for parsing
    if date_only:
        release_timestamp = f"{release_timestamp} 00:00:00"

    # Convert to human-readable format, e.g., "October 13, 2025 06:00"
    dt = datetime.strptime(release_timestamp, "%Y-%m-%d %H:%M:%S")
    fmt = "%B %d, %Y" if date_only else "%B %d, %Y %H:%M"
    human_readable = dt.strftime(fmt)

    return human_readable, current_version


def update_config_copyright(config_path: Path, desired_text: str) -> bool:
    lines = config_path.read_text(encoding="utf-8").splitlines(keepends=True)

    for index, line in enumerate(lines):
        match = re.match(r'^(\s*copyright:\s*)([\"\'])(.*?)(\2)(\s*#.*)?\s*$', line)
        if not match:
            continue

        prefix, quote, content, _, suffix = match.groups()
        if content != desired_text:
            suffix = suffix or ""
            lines[index] = f"{prefix}{quote}{desired_text}{quote}{suffix}\n"
            config_path.write_text("".join(lines), encoding="utf-8")
            return True
        return False

    raise ValueError("copyright: line not found in _config.yml")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / "public/versions/whitepaper-version.json"
    config_path = repo_root / "jupyter_book/_config.yml"

    manifest = load_whitepaper_manifest(manifest_path)
    release_ts, version = get_current_release_timestamp_and_version(manifest)
    desired = f"{release_ts} {version}"

    changed = update_config_copyright(config_path, desired)
    if changed:
        print(f"Updated {config_path} -> {desired}")
    else:
        print("No changes needed.")


if __name__ == "__main__":
    main()


