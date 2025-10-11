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
        """Check if pdflatex and bibtex are available"""
        # Check pdflatex
        try:
            result = subprocess.run(['pdflatex', '--version'],
                                  capture_output=True, text=True, check=True)
            print(f"✅ LaTeX found: {result.stdout.split('\\n')[0]}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ pdflatex not found. Please install a LaTeX distribution:")
            print("   - Ubuntu/Debian: sudo apt-get install texlive-latex-base texlive-latex-extra")
            print("   - macOS: brew install mactex")
            print("   - Windows: Install MikTeX or TeX Live")
            return False

        # Check bibtex
        try:
            result = subprocess.run(['bibtex', '--version'],
                                  capture_output=True, text=True, check=True)
            print(f"✅ BibTeX found: {result.stdout.split('\\n')[0]}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ bibtex not found. Please install a LaTeX distribution with BibTeX:")
            print("   - Ubuntu/Debian: sudo apt-get install texlive-bibtex-extra")
            print("   - macOS: BibTeX is included with MacTeX")
            print("   - Windows: BibTeX is included with MikTeX or TeX Live")
            return False

    def compile_pdf(self):
        """Compile LaTeX file to PDF with bibliography support"""
        print(f"🔄 Compiling {self.input_file} to PDF...")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        try:
            # Step 1: Initial pdflatex run to generate auxiliary files
            print("   Pass 1/4: Initial LaTeX compilation...")
            result = subprocess.run([
                'pdflatex',
                '-output-directory', 'public',
                '-jobname', 'whitepaper-latex',
                '-interaction=nonstopmode',
                self.input_file
            ], capture_output=True, text=True, cwd='.')

            if result.returncode != 0:
                print(f"❌ LaTeX compilation failed on initial pass")
                print("Error output:")
                print(result.stdout)
                print(result.stderr)
                return False

            # Step 2: Copy bibliography file and run bibtex
            print("   Pass 2/4: Processing bibliography...")
            # Copy references.bib to public directory for bibtex
            import shutil
            bib_source = "docs/references.bib"
            bib_dest = "public/references.bib"
            if os.path.exists(bib_source):
                shutil.copy2(bib_source, bib_dest)
                print(f"   Copied bibliography file to public directory")

            bib_result = subprocess.run([
                'bibtex',
                'whitepaper-latex'
            ], capture_output=True, text=True, cwd='public')

            # Bibtex return code 1 is normal when there are warnings
            if bib_result.returncode not in [0, 1]:
                print(f"❌ BibTeX failed with return code {bib_result.returncode}")
                print("Error output:")
                print(bib_result.stdout)
                print(bib_result.stderr)
                return False

            # Step 3: Second pdflatex run to incorporate bibliography
            print("   Pass 3/4: Incorporating bibliography...")
            result2 = subprocess.run([
                'pdflatex',
                '-output-directory', 'public',
                '-jobname', 'whitepaper-latex',
                '-interaction=nonstopmode',
                self.input_file
            ], capture_output=True, text=True, cwd='.')

            if result2.returncode != 0:
                print(f"❌ LaTeX compilation failed on bibliography pass")
                print("Error output:")
                print(result2.stdout)
                print(result2.stderr)
                return False

            # Step 4: Final pdflatex run to resolve references
            print("   Pass 4/4: Final compilation...")
            result3 = subprocess.run([
                'pdflatex',
                '-output-directory', 'public',
                '-jobname', 'whitepaper-latex',
                '-interaction=nonstopmode',
                self.input_file
            ], capture_output=True, text=True, cwd='.')

            if result3.returncode != 0:
                print(f"❌ LaTeX compilation failed on final pass")
                print("Error output:")
                print(result3.stdout)
                print(result3.stderr)
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
        """Clean up auxiliary LaTeX files (keeping .bbl for bibliography)"""
        aux_dir = "public"
        # Don't remove .bbl file as it's needed for bibliography display
        aux_extensions = ['.aux', '.log', '.out', '.toc', '.lof', '.lot', '.blg']
        # Also remove the copied references.bib file
        aux_files = ['references.bib']

        print("🧹 Cleaning up auxiliary files...")
        cleaned = 0

        # Remove specific files
        for filename in aux_files:
            aux_file = os.path.join(aux_dir, filename)
            if os.path.exists(aux_file):
                try:
                    os.remove(aux_file)
                    cleaned += 1
                except OSError as e:
                    print(f"   Warning: Could not remove {aux_file}: {e}")

        # Remove auxiliary files by extension
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
            print("   Note: Keeping .bbl file for bibliography display")

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
