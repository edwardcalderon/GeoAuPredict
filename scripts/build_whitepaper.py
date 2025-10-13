#!/usr/bin/env python3
"""
Complete White Paper Build Script for GeoAuPredict

This script performs a complete build of the white paper from a SINGLE SOURCE:
1. LaTeX Source (docs/whitepaper.tex) - THE ONLY FILE YOU EDIT
2. Converts to Markdown for docs/whitepaper.md
3. Builds Jupyter Book (splits into sections automatically)
4. Compiles PDF for download

Usage:
    python scripts/build_whitepaper.py

This ensures ALL outputs (web, Jupyter Book, PDF) stay in sync from one source.
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
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed with exit code {e.returncode}")
        print("Error output:")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False
    except FileNotFoundError:
        print(f"❌ Script not found: scripts/{script_name}")
        return False

def main():
    """Main build process"""
    print("=" * 70)
    print("🔥 GeoAuPredict White Paper Complete Build")
    print("   Single Source of Truth: docs/whitepaper.tex")
    print("=" * 70)

    success_count = 0
    total_steps = 3

    # Step 1: Convert LaTeX to Markdown (for docs/whitepaper.md)
    if run_script("convert_latex_to_markdown.py", "Converting LaTeX to Markdown"):
        success_count += 1
    else:
        print("\n💥 Build failed during markdown conversion!")
        sys.exit(1)

    # Step 2: Build Jupyter Book (split LaTeX into sections)
    if run_script("build_jupyter_book_from_latex.py", "Building Jupyter Book from LaTeX"):
        success_count += 1
    else:
        print("\n⚠️  Jupyter Book build failed, continuing with PDF...")

    # Step 3: Compile LaTeX to PDF (optional, may fail if LaTeX not installed)
    print("\n📄 Attempting PDF compilation (may fail if LaTeX not installed)...")
    if run_script("compile_latex_to_pdf.py", "Compiling LaTeX to PDF"):
        success_count += 1

    # Summary
    print("\n" + "=" * 70)
    print("📋 Build Summary")
    print("=" * 70)
    
    print(f"\n✅ Completed {success_count}/{total_steps} build steps\n")
    
    print("📁 Output Files:")
    print("   • Source (EDIT THIS):     docs/whitepaper.tex")
    print("   • Markdown:               docs/whitepaper.md")
    print("   • Jupyter Book:           jupyter_book/*.md")
    print("   • Jupyter Book HTML:      jupyter_book/_build/html/")
    print("   • PDF:                    public/whitepaper-latex.pdf")

    if success_count < total_steps:
        print("\n⚠️  Some steps failed. See messages above.")
        print("\n💡 Common issues:")
        print("   - LaTeX not installed: sudo apt-get install texlive-full")
        print("   - Jupyter Book not installed: pip install jupyter-book")
    else:
        print("\n🎉 Complete build successful! All outputs in sync.")

    print("\n🔄 Workflow:")
    print("   1. Edit:  docs/whitepaper.tex (your single source)")
    print("   2. Build: python scripts/build_whitepaper.py")
    print("   3. Done:  All formats updated automatically!")

if __name__ == '__main__':
    main()
