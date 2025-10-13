#!/usr/bin/env python3
"""
Version management script for GeoAuPredict whitepaper

UNIFIED WORKFLOW - LaTeX as Single Source of Truth
Builds PDF, Markdown, and Jupyter Book from docs/whitepaper.tex
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

class WhitepaperVersionManager:
    """Manages versioning for the whitepaper"""

    def __init__(self):
        self.versions_file = "public/versions/whitepaper-version.json"
        self.versions_dir = "public/versions"
        self.current_pdf = "public/whitepaper-latex.pdf"

    def get_git_info(self):
        """Get git author and timestamp information"""
        try:
            # Get git username/email
            author_result = subprocess.run(['git', 'config', 'user.name'],
                                         capture_output=True, text=True, cwd='.')
            author = author_result.stdout.strip()

            email_result = subprocess.run(['git', 'config', 'user.email'],
                                        capture_output=True, text=True, cwd='.')
            email = email_result.stdout.strip()

            # Get last commit info
            commit_result = subprocess.run(['git', 'log', '-1', '--format=%H %ad', '--date=iso'],
                                         capture_output=True, text=True, cwd='.')
            if commit_result.returncode == 0:
                commit_info = commit_result.stdout.strip().split(' ', 1)
                if len(commit_info) >= 2:
                    commit_hash = commit_info[0]
                    commit_date = commit_info[1]
                else:
                    commit_date = datetime.now().isoformat()
                    commit_hash = "unknown"
            else:
                commit_date = datetime.now().isoformat()
                commit_hash = "unknown"

            return {
                'author': author or "Unknown",
                'email': email or "unknown@example.com",
                'commit_hash': commit_hash,
                'commit_date': commit_date,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception:
            # Fallback if git is not available
            return {
                'author': "System",
                'email': "system@localhost",
                'commit_hash': "unknown",
                'commit_date': datetime.now().isoformat(),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

    def extract_version_from_tex(self, tex_file_path):
        """Extract version number from TeX file"""
        try:
            with open(tex_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Look for version patterns in the TeX file
            # Pattern 1: \date{... \\ {\small Version X.X.X}}
            import re

            # Find version in date line
            date_pattern = r'\\date[^}]*Version\s+([\d.]+)'
            match = re.search(date_pattern, content, re.IGNORECASE)
            if match:
                return match.group(1)

            # Pattern 2: Version X.X.X in text
            version_pattern = r'Version\s+([\d.]+)'
            match = re.search(version_pattern, content, re.IGNORECASE)
            if match:
                return match.group(1)

            return None
        except Exception as e:
            print(f"Warning: Could not extract version from TeX file: {e}")
            return None

    def get_pdf_metadata(self, pdf_path):
        """Get PDF file size and page count"""
        try:
            # Get file size
            file_size_bytes = os.path.getsize(pdf_path)
            # Convert to KB
            file_size_kb = file_size_bytes / 1024
            size_str = f"{file_size_kb:.0f}KB"

            # Get page count using PyPDF2
            try:
                from PyPDF2 import PdfReader
                with open(pdf_path, 'rb') as f:
                    pdf_reader = PdfReader(f)
                    page_count = len(pdf_reader.pages)
            except ImportError:
                print("Warning: PyPDF2 not available, using default page count")
                page_count = 9  # fallback
            except Exception as e:
                print(f"Warning: Could not read PDF pages: {e}")
                page_count = 9  # fallback

            return {
                'size': size_str,
                'pages': page_count
            }
        except Exception as e:
            print(f"Warning: Could not get PDF metadata: {e}")
            return {
                'size': "Unknown",
                'pages': 9
            }

    def validate_tex_version(self, tex_file_path, expected_version):
        """Validate that the version in TeX file matches expected version"""
        actual_version = self.extract_version_from_tex(tex_file_path)
        if actual_version is None:
            print(f"❌ Could not extract version from TeX file: {tex_file_path}")
            print("   Please ensure the version is properly formatted in the TeX file")
            return False

        if actual_version != expected_version.lstrip('v'):
            print(f"❌ Version mismatch!")
            print(f"   Expected version: {expected_version.lstrip('v')}")
            print(f"   TeX file version: {actual_version}")
            print("   Please update the version in the TeX file to match the release version")
            return False

        print(f"✅ Version validation passed: {actual_version}")
        return True

    def load_versions(self):
        """Load existing version data"""
        if os.path.exists(self.versions_file):
            with open(self.versions_file, 'r') as f:
                return json.load(f)
        return {
            "currentVersion": "v1.0.3",
            "versions": [],
            "downloadUrl": "/whitepaper-latex.pdf"
        }

    def save_versions(self, data):
        """Save version data"""
        os.makedirs(os.path.dirname(self.versions_file), exist_ok=True)
        with open(self.versions_file, 'w') as f:
            json.dump(data, f, indent=2)

    def create_version(self, new_version, changes=""):
        """Create a new version of the whitepaper with full compilation workflow"""
        print(f"🚀 Creating version {new_version}...")

        # Extract expected version number for validation (remove 'v' prefix)
        expected_version_num = new_version.lstrip('v')

        # Validate TeX file version before proceeding
        tex_file_path = "docs/whitepaper.tex"
        if not self.validate_tex_version(tex_file_path, expected_version_num):
            print("❌ Version validation failed. Please fix the version in the TeX file.")
            return False

        # Get git author information
        git_info = self.get_git_info()
        print(f"👤 Author: {git_info['author']} ({git_info['email']})")

        # Step 1: Build all outputs using unified workflow
        print("\n🔨 Building all outputs from LaTeX (single source of truth)...")
        print("   • PDF: public/whitepaper-latex.pdf")
        print("   • Markdown: docs/whitepaper.md")
        print("   • Jupyter Book: jupyter_book/*.md + _build/html/\n")
        
        try:
            # Run the unified build script
            result = subprocess.run(
                [sys.executable, "scripts/build_whitepaper.py"],
                capture_output=True,
                text=True,
                timeout=180
            )
            
            if result.returncode == 0:
                print("✅ All whitepaper outputs built successfully")
            else:
                print("❌ Build failed!")
                if result.stdout:
                    print("Output:", result.stdout)
                if result.stderr:
                    print("Errors:", result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Build timed out (took longer than 3 minutes)")
            return False
        except Exception as e:
            print(f"❌ Build failed: {e}")
            return False

        # Step 2: Version management
        print("\n📦 Step 2: Managing version files...")

        # Load existing versions
        versions_data = self.load_versions()

        # Check if version already exists
        for version in versions_data["versions"]:
            if version["version"] == new_version:
                print(f"❌ Version {new_version} already exists!")
                return False

        # Copy current PDF to versions folder
        version_filename = f"whitepaper-{new_version}.pdf"
        version_path = os.path.join(self.versions_dir, version_filename)

        shutil.copy2(self.current_pdf, version_path)
        print(f"✅ Created version: {version_path}")

        # Get PDF metadata programmatically
        pdf_metadata = self.get_pdf_metadata(self.current_pdf)

        # Add version info with git author data
        version_info = {
            "version": new_version,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "timestamp": git_info['timestamp'],
            "changes": changes,
            "author": git_info['author'],
            "email": git_info['email'],
            "commit_hash": git_info['commit_hash'],
            "pdf_url": f"/versions/{version_filename}",
            "size": pdf_metadata['size'],
            "pages": pdf_metadata['pages']
        }

        versions_data["versions"].append(version_info)
        versions_data["currentVersion"] = new_version
        versions_data["downloadUrl"] = f"/versions/{version_filename}"

        # Save updated versions
        self.save_versions(versions_data)

        print(f"\n✅ Version {new_version} created successfully!")
        print(f"   📄 PDF: {versions_data['downloadUrl']}")
        print(f"   📝 Markdown: docs/whitepaper.md")
        print(f"   📚 Jupyter Book: jupyter_book/_build/html/")
        print(f"   👤 Committed by: @{git_info['author']} on {git_info['timestamp']}")
        print(f"   📊 PDF size: {pdf_metadata['size']}, Pages: {pdf_metadata['pages']}")

        # Step 3: Sync Jupyter Book config with human-readable date and version
        try:
            sync_result = subprocess.run(
                [sys.executable, "scripts/sync_jupyter_book_config_version.py"],
                capture_output=True,
                text=True,
                check=True
            )
            sync_msg = sync_result.stdout.strip() if sync_result.stdout else "No output"
            print(f"\n🧩 Synced Jupyter Book config: {sync_msg}")
        except subprocess.CalledProcessError as e:
            print("\n⚠️  Failed to sync Jupyter Book config (_config.yml)")
            if e.stdout:
                print(e.stdout)
            if e.stderr:
                print(e.stderr)

        # Step 4: Rebuild Jupyter Book so footer reflects updated config/version
        try:
            rebuild_result = subprocess.run(
                [sys.executable, "scripts/build_jupyter_book_from_latex.py"],
                capture_output=True,
                text=True,
                check=True
            )
            print("\n📖 Rebuilt Jupyter Book after config sync")
        except subprocess.CalledProcessError as e:
            print("\n⚠️  Failed to rebuild Jupyter Book after config sync")
            if e.stdout:
                print(e.stdout)
            if e.stderr:
                print(e.stderr)

        return True

    def list_versions(self):
        """List all available versions"""
        versions_data = self.load_versions()

        print(f"\n📋 Current version: {versions_data['currentVersion']}")
        print(f"📁 Available versions: {len(versions_data['versions'])}\n")

        # Sort versions by semantic version number (highest first) for display
        sorted_versions = sorted(versions_data["versions"],
                               key=self.version_key, reverse=True)

        for version in sorted_versions:
            timestamp = version.get('timestamp', version.get('date', 'Unknown'))
            author = version.get('author', 'Unknown')
            changes = version.get('changes', 'No description')
            pdf_url = version.get('pdf_url', 'Unknown')

            print(f"  {version['version']} - {timestamp}")
            print(f"    Last change committed by @{author}")
            print(f"    {changes}")
            print(f"    File: {pdf_url}")
            print()

    def version_key(self, version_item):
        """Create a sort key for proper semantic version sorting"""
        version_str = version_item["version"]

        # Extract version components (v1.1.2 -> [1, 1, 2])
        try:
            # Remove 'v' prefix and split by '.'
            version_parts = version_str.lstrip('v').split('.')
            # Convert to integers, pad with zeros if needed for proper comparison
            return tuple(int(part) for part in version_parts)
        except (ValueError, AttributeError):
            # Fallback for malformed versions - use timestamp
            timestamp = version_item.get("timestamp", version_item.get("date", "0"))
            return (999, 999, 999, timestamp)  # Put at end

    def cleanup_old_versions(self, keep_latest=5):
        """Clean up old versions, keeping only the latest N versions"""
        versions_data = self.load_versions()

        if len(versions_data["versions"]) <= keep_latest:
            print(f"✅ No cleanup needed. Only {len(versions_data['versions'])} versions exist.")
            return

        # Sort versions by semantic version number (highest first)
        sorted_versions = sorted(versions_data["versions"],
                               key=self.version_key, reverse=True)

        versions_to_keep = sorted_versions[:keep_latest]
        versions_to_remove = sorted_versions[keep_latest:]

        print(f"📋 Keeping latest {keep_latest} versions:")
        for v in versions_to_keep:
            print(f"   ✅ {v['version']} - {v.get('timestamp', v.get('date', 'Unknown'))}")

        if versions_to_remove:
            print(f"🗑️  Removing {len(versions_to_remove)} old versions:")
            for old_version in versions_to_remove:
                print(f"   ❌ {old_version['version']} - {old_version.get('timestamp', old_version.get('date', 'Unknown'))}")

        # Remove old files and update versions list
        for old_version in versions_to_remove:
            pdf_url = old_version.get("pdf_url", "")
            old_filename = pdf_url.replace("/versions/", "") if pdf_url else ""
            old_file = os.path.join(self.versions_dir, old_filename)
            if os.path.exists(old_file):
                os.remove(old_file)
                print(f"   🗑️  Removed file: {old_filename}")

        # Update versions list
        versions_data["versions"] = versions_to_keep
        self.save_versions(versions_data)

        print(f"✅ Cleanup completed. Kept latest {keep_latest} versions.")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Manage whitepaper versions')
    parser.add_argument('action', choices=['create', 'list', 'cleanup'],
                       help='Action to perform')
    parser.add_argument('--version', '-v',
                       help='Version number (e.g., v1.0.0) for create action')
    parser.add_argument('--changes', '-c',
                       help='Description of changes for create action')
    parser.add_argument('--keep', '-k', type=int, default=5,
                       help='Number of latest versions to keep (for cleanup)')

    args = parser.parse_args()

    manager = WhitepaperVersionManager()

    if args.action == 'create':
        if not args.version:
            print("❌ Version number is required for create action")
            return
        if not args.changes:
            print("❌ Changes description is required for create action")
            return

        success = manager.create_version(args.version, args.changes)
        if success:
            print("\n🎉 Version created! Don't forget to commit the new files:")
            print("   git add public/versions/")
            print(f"   git commit -m \"Release whitepaper {args.version}\"")

    elif args.action == 'list':
        manager.list_versions()

    elif args.action == 'cleanup':
        manager.cleanup_old_versions(args.keep)

if __name__ == '__main__':
    main()
