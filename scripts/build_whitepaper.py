#!/usr/bin/env python3
"""
Complete White Paper Build Script for GeoAuPredict

This script performs a complete build of the white paper:
1. Converts LaTeX to Markdown for web display
2. Compiles LaTeX to PDF for download

Usage:
    python scripts/build_whitepaper.py

This ensures both web and PDF versions are always in sync.
"""

import sys
import subprocess
from pathlib import Path

def run_script(script_name, description):
    """Run a script and handle errors"""
    print(f"\n🚀 {description}...")
    try:
        result = subprocess.run([sys.executable, f"scripts/{script_name}"],
                              capture_output=True, text=True, check=True)
        print("✅ Completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed with exit code {e.returncode}")
        print("Error output:")
        print(e.stdout)
        print(e.stderr)
        return False
    except FileNotFoundError:
        print(f"❌ Script not found: scripts/{script_name}")
        return False

def main():
    """Main build process"""
    print("🔥 GeoAuPredict White Paper Complete Build")
    print("=" * 50)

    # Step 1: Convert LaTeX to Markdown
    if not run_script("convert_latex_to_markdown.py", "Converting LaTeX to Markdown"):
        print("\n💥 Build failed during markdown conversion!")
        sys.exit(1)

    # Step 2: Compile LaTeX to PDF (optional, may fail if LaTeX not installed)
    print("\n📄 Attempting PDF compilation (may fail if LaTeX not installed)...")
    pdf_success = run_script("compile_latex_to_pdf.py", "Compiling LaTeX to PDF")

    if pdf_success:
        print("\n🎉 Complete build successful!")
        print("   ✅ Markdown version updated")
        print("   ✅ PDF version compiled")
    else:
        print("\n⚠️  Build completed with warnings:")
        print("   ✅ Markdown version updated")
        print("   ❌ PDF compilation failed (LaTeX not installed?)")
        print("\n💡 To install LaTeX:")
        print("   - Ubuntu/Debian: sudo apt-get install texlive-latex-base texlive-latex-extra")
        print("   - macOS: brew install mactex")
        print("   - Windows: Install MikTeX")

    print("\n📋 Build Summary:")
    print("   • Markdown: docs/whitepaper.md (web display)")
    print("   • PDF: public/whitepaper-latex.pdf (downloadable)")
    print("   • Source: docs/whitepaper.tex (edit this file)")

if __name__ == '__main__':
    main()
