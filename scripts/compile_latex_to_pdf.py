#!/usr/bin/env python3
"""
LaTeX to PDF Compiler for GeoAuPredict White Paper

This script compiles the LaTeX whitepaper.tex to PDF format for download.
Requires a LaTeX distribution (TeX Live, MikTeX, etc.) with pdflatex.
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path

class LaTeXToPDFCompiler:
    """Compiles LaTeX document to PDF"""

    def __init__(self):
        self.input_file = "docs/whitepaper.tex"
        self.output_file = "public/whitepaper-latex.pdf"

    def check_dependencies(self):
        """Check if pdflatex is available"""
        try:
            result = subprocess.run(['pdflatex', '--version'],
                                  capture_output=True, text=True, check=True)
            print(f"✅ LaTeX found: {result.stdout.split('\\n')[0]}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ pdflatex not found. Please install a LaTeX distribution:")
            print("   - Ubuntu/Debian: sudo apt-get install texlive-latex-base texlive-latex-extra")
            print("   - macOS: brew install mactex")
            print("   - Windows: Install MikTeX or TeX Live")
            return False

    def compile_pdf(self):
        """Compile LaTeX file to PDF"""
        print(f"🔄 Compiling {self.input_file} to PDF...")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        try:
            # Run pdflatex twice to resolve references
            for i in range(2):
                print(f"   Pass {i+1}/2...")

                # Run pdflatex with output directory and specific job name
                result = subprocess.run([
                    'pdflatex',
                    '-output-directory', 'public',
                    '-jobname', 'whitepaper-latex',
                    '-interaction=nonstopmode',
                    self.input_file
                ], capture_output=True, text=True, cwd='.')

                if result.returncode != 0:
                    print(f"❌ LaTeX compilation failed on pass {i+1}")
                    print("Error output:")
                    print(result.stdout)
                    print(result.stderr)
                    return False

            # Check if PDF was created
            if os.path.exists(self.output_file):
                file_size = os.path.getsize(self.output_file)
                print(f"✅ PDF compiled successfully: {self.output_file} ({file_size:,} bytes)")
                return True
            else:
                print(f"❌ PDF file not found: {self.output_file}")
                return False

        except Exception as e:
            print(f"❌ Error during compilation: {e}")
            return False

    def cleanup_auxiliary_files(self):
        """Clean up auxiliary LaTeX files"""
        aux_dir = "public"
        aux_extensions = ['.aux', '.log', '.out', '.toc', '.lof', '.lot', '.bbl', '.blg']

        print("🧹 Cleaning up auxiliary files...")
        cleaned = 0

        for ext in aux_extensions:
            aux_file = os.path.join(aux_dir, 'whitepaper' + ext)
            if os.path.exists(aux_file):
                try:
                    os.remove(aux_file)
                    cleaned += 1
                except OSError as e:
                    print(f"   Warning: Could not remove {aux_file}: {e}")

        if cleaned > 0:
            print(f"   Removed {cleaned} auxiliary files")

    def compile_and_cleanup(self):
        """Full compilation process with cleanup"""
        print("🚀 Starting LaTeX to PDF compilation...")

        if not self.check_dependencies():
            return False

        success = self.compile_pdf()

        if success:
            self.cleanup_auxiliary_files()
            print("🎉 PDF compilation completed successfully!")
            print(f"   PDF available at: {self.output_file}")
            print("   Web access: /whitepaper-latex.pdf")
        else:
            print("💥 PDF compilation failed!")

        return success

def main():
    parser = argparse.ArgumentParser(description='Compile LaTeX whitepaper to PDF')
    parser.add_argument('--input', '-i', default='docs/whitepaper.tex',
                       help='Input LaTeX file path')
    parser.add_argument('--output', '-o', default='public/whitepaper-latex.pdf',
                       help='Output PDF file path')
    parser.add_argument('--keep-aux', action='store_true',
                       help='Keep auxiliary LaTeX files (don\'t clean up)')

    args = parser.parse_args()

    compiler = LaTeXToPDFCompiler()
    compiler.input_file = args.input
    compiler.output_file = args.output

    success = compiler.compile_and_cleanup()

    if not success:
        sys.exit(1)

if __name__ == '__main__':
    main()
