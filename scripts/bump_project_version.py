#!/usr/bin/env python3
"""
Automated Version Bumping Script for GeoAuPredict
Updates VERSION file, src/__version__.py, and package.json
"""

import json
import sys
import re
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent


def read_current_version():
    """Read version from VERSION file"""
    version_file = PROJECT_ROOT / "VERSION"
    return version_file.read_text().strip()


def parse_version(version_str):
    """Parse semantic version string to tuple"""
    match = re.match(r'(\d+)\.(\d+)\.(\d+)', version_str)
    if not match:
        raise ValueError(f"Invalid version format: {version_str}")
    return tuple(map(int, match.groups()))


def bump_version(version_tuple, bump_type):
    """Bump version based on type (major, minor, patch)"""
    major, minor, patch = version_tuple
    
    if bump_type == 'major':
        return (major + 1, 0, 0)
    elif bump_type == 'minor':
        return (major, minor + 1, 0)
    elif bump_type == 'patch':
        return (major, minor, patch + 1)
    else:
        raise ValueError(f"Invalid bump type: {bump_type}. Use: major, minor, patch")


def update_version_file(new_version):
    """Update VERSION file"""
    version_file = PROJECT_ROOT / "VERSION"
    version_file.write_text(f"{new_version}\n")
    print(f"✓ Updated VERSION: {new_version}")


def update_version_py(new_version, bump_type):
    """Update src/__version__.py"""
    version_py = PROJECT_ROOT / "src" / "__version__.py"
    content = version_py.read_text()
    
    # Update version strings
    content = re.sub(
        r'__version__ = "[^"]*"',
        f'__version__ = "{new_version}"',
        content
    )
    
    major, minor, patch = parse_version(new_version)
    content = re.sub(
        r'__version_info__ = \([^\)]*\)',
        f'__version_info__ = ({major}, {minor}, {patch})',
        content
    )
    
    version_py.write_text(content)
    print(f"✓ Updated src/__version__.py: {new_version}")


def update_package_json(new_version):
    """Update package.json"""
    package_json = PROJECT_ROOT / "package.json"
    
    with open(package_json, 'r') as f:
        data = json.load(f)
    
    data['version'] = new_version
    
    with open(package_json, 'w') as f:
        json.dump(data, f, indent=2)
        f.write('\n')  # Add newline at end
    
    print(f"✓ Updated package.json: {new_version}")


def create_git_tag(new_version):
    """Create git tag for new version"""
    import subprocess
    
    try:
        # Create annotated tag
        subprocess.run([
            'git', 'tag', '-a', f'v{new_version}',
            '-m', f'Release v{new_version}'
        ], check=True)
        
        print(f"✓ Created git tag: v{new_version}")
        print(f"  Run 'git push origin v{new_version}' to push tag")
        
    except subprocess.CalledProcessError:
        print(f"⚠ Could not create git tag (git may not be available)")


def prompt_changelog():
    """Prompt for changelog entry"""
    print("\n" + "="*60)
    print(f"📝 CHANGELOG Entry")
    print("="*60)
    print("\nPlease describe the changes in this version:")
    print("(Enter a blank line when done)\n")
    
    changes = []
    while True:
        line = input("> ")
        if not line:
            break
        changes.append(line)
    
    return changes


def update_changelog(new_version, changes):
    """Update CHANGELOG.md"""
    changelog = PROJECT_ROOT / "CHANGELOG.md"
    
    # Read existing content
    if changelog.exists():
        existing = changelog.read_text()
    else:
        existing = "# Changelog\n\nAll notable changes to GeoAuPredict will be documented in this file.\n\n"
    
    # Create new entry
    today = datetime.now().strftime("%Y-%m-%d")
    new_entry = f"\n## [{new_version}] - {today}\n\n"
    
    if changes:
        new_entry += "### Changed\n"
        for change in changes:
            new_entry += f"- {change}\n"
    
    # Insert after header
    lines = existing.split('\n')
    header_end = 3  # After "# Changelog" and description
    
    updated = '\n'.join(lines[:header_end]) + new_entry + '\n'.join(lines[header_end:])
    
    changelog.write_text(updated)
    print(f"✓ Updated CHANGELOG.md")


def main():
    if len(sys.argv) != 2:
        print("Usage: python bump_version.py [major|minor|patch]")
        print("\nCurrent version:", read_current_version())
        sys.exit(1)
    
    bump_type = sys.argv[1].lower()
    
    if bump_type not in ['major', 'minor', 'patch']:
        print(f"Error: Invalid bump type '{bump_type}'")
        print("Use: major, minor, or patch")
        sys.exit(1)
    
    # Get current version
    current_version = read_current_version()
    current_tuple = parse_version(current_version)
    
    # Calculate new version
    new_tuple = bump_version(current_tuple, bump_type)
    new_version = '.'.join(map(str, new_tuple))
    
    print("\n" + "="*60)
    print(f"🔄 Version Bump: {current_version} → {new_version} ({bump_type})")
    print("="*60)
    
    # Confirm
    response = input(f"\nProceed with version bump? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    # Update files
    update_version_file(new_version)
    update_version_py(new_version, bump_type)
    update_package_json(new_version)
    
    # Get changelog
    changes = prompt_changelog()
    if changes:
        update_changelog(new_version, changes)
    
    # Create git tag (optional)
    create_git = input(f"\nCreate git tag v{new_version}? (y/N): ")
    if create_git.lower() == 'y':
        create_git_tag(new_version)
    
    print("\n" + "="*60)
    print(f"✅ Version bumped successfully: {new_version}")
    print("="*60)
    print("\nNext steps:")
    print(f"  1. Review changes: git diff")
    print(f"  2. Commit: git add -A && git commit -m 'chore: bump version to {new_version}'")
    print(f"  3. Push: git push")
    if create_git.lower() == 'y':
        print(f"  4. Push tag: git push origin v{new_version}")


if __name__ == '__main__':
    main()

