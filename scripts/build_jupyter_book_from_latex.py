#!/usr/bin/env python3
"""
LaTeX to Jupyter Book Converter - Single Source of Truth Workflow

This script converts whitepaper.tex into a Jupyter Book by:
1. Splitting LaTeX sections into separate markdown files
2. Maintaining proper cross-references and citations
3. Building both PDF and Jupyter Book from the same source

Usage:
    python scripts/build_jupyter_book_from_latex.py
    
The LaTeX file (docs/whitepaper.tex) is the single source of truth.
All Jupyter Book markdown files are auto-generated from it.
"""

import re
import sys
from pathlib import Path
from datetime import datetime

# Import the existing LaTeX converter
sys.path.insert(0, str(Path(__file__).parent))
from convert_latex_to_markdown import LaTeXConverter


class JupyterBookBuilder:
    """Builds Jupyter Book from LaTeX whitepaper"""
    
    def __init__(self, latex_file='docs/whitepaper.tex', output_dir='jupyter_book'):
        self.latex_file = Path(latex_file)
        self.output_dir = Path(output_dir)
        self.converter = LaTeXConverter()
        
    def split_latex_by_sections(self, content):
        """Split LaTeX content into sections based on \\section commands"""
        
        # Define section mapping for Jupyter Book structure
        # Pattern captures everything from the section start to the next section or end
        sections = {
            'intro': {
                'pattern': r'\\section\{Introduction\}(.*?)(?=\\section|\\bibliographystyle|\Z)',
                'file': 'intro.md',
                'title': 'Introduction'
            },
            'methodology': {
                'pattern': r'\\section\{(?:Materials and )?Methods?\}(.*?)(?=\\section|\\bibliographystyle|\Z)',
                'file': 'methodology.md',
                'title': 'Materials and Methods'
            },
            'results': {
                'pattern': r'\\section\{Results\}(.*?)(?=\\section|\\bibliographystyle|\Z)',
                'file': 'results.md',
                'title': 'Results'
            },
            'discussion': {
                'pattern': r'\\section\{Discussion\}(.*?)(?=\\section|\\bibliographystyle|\Z)',
                'file': 'discussion.md',
                'title': 'Discussion'
            },
            'conclusion': {
                'pattern': r'\\section\{Conclusion(?:s)?\}(.*?)(?=\\section|\\bibliographystyle|\Z)',
                'file': 'conclusion.md',
                'title': 'Conclusion'
            }
        }
        
        extracted_sections = {}
        
        for section_key, section_info in sections.items():
            match = re.search(section_info['pattern'], content, re.DOTALL | re.IGNORECASE)
            if match:
                extracted_sections[section_key] = {
                    'content': match.group(1).strip(),
                    'file': section_info['file'],
                    'title': section_info['title']
                }
                print(f"  ✓ Found section: {section_info['title']}")
            else:
                print(f"  ⚠ Section not found: {section_info['title']}")
                
        return extracted_sections
    
    def extract_preamble_info(self, content):
        """Extract version, author, abstract info from preamble"""
        info = {}
        
        # Extract version from date field or title
        version_match = re.search(r'Version\s+([\d.]+)', content)
        if version_match:
            info['version'] = version_match.group(1)
            print(f"  ✓ Found version: {info['version']}")
        
        # Extract version box/notice if present
        version_box_match = re.search(
            r'\\fbox\{%?\s*\\parbox\{[^}]+\}\{%?\s*(.*?)\s*\}%?\s*\}',
            content, re.DOTALL
        )
        if version_box_match:
            version_box_raw = version_box_match.group(1)
            # Manual cleanup before using converter (more aggressive)
            version_box_raw = re.sub(r'\\\\', '\n', version_box_raw)
            version_box_raw = re.sub(r'\\textbf\{([^}]+)\}', r'**\1**', version_box_raw)
            version_box_raw = re.sub(r'\\textit\{([^}]+)\}', r'*\1*', version_box_raw)
            version_box_raw = re.sub(r'\\url\{([^}]+)\}', r'<\1>', version_box_raw)
            # Apply converter for remaining LaTeX
            version_box_clean = self.converter.convert_text(version_box_raw)
            # Final cleanup
            version_box_clean = re.sub(r'\*\*\s*\*\*', '', version_box_clean)  # Remove empty bolds
            version_box_clean = re.sub(r'\*\s*\*', '', version_box_clean)  # Remove empty italics
            info['version_notice'] = version_box_clean.strip()
            print(f"  ✓ Found version notice box")
        
        # Extract abstract
        abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', content, re.DOTALL)
        if abstract_match:
            abstract_raw = abstract_match.group(1).strip()
            # Convert LaTeX to markdown, clean up commands
            abstract_md = self.converter.convert_text(abstract_raw)
            # Additional cleanup for abstract
            abstract_md = re.sub(r'\\noindent\s*', '', abstract_md)
            abstract_md = re.sub(r'\\vspace\{[^}]+\}', '', abstract_md)
            abstract_md = re.sub(r'~\\citep\{([^}]+)\}', r' [@\1]', abstract_md)
            abstract_md = re.sub(r'\\citep\{([^}]+)\}', r'[@\1]', abstract_md)
            info['abstract'] = abstract_md
            print(f"  ✓ Found abstract ({len(info['abstract'])} chars)")
            
        # Extract title
        title_match = re.search(r'\\title\{\\textbf\{(.*?)\}\}', content, re.DOTALL)
        if not title_match:
            title_match = re.search(r'\\title\{(.*?)\}', content, re.DOTALL)
        if title_match:
            title_raw = title_match.group(1).strip()
            # Clean up the title - remove version info if present
            title_clean = re.sub(r'\{\\small.*?\}', '', title_raw)
            title_clean = re.sub(r'\\textbf\{([^}]+)\}', r'\1', title_clean)
            info['title'] = title_clean.strip()
            print(f"  ✓ Found title: {info['title'][:50]}...")
            
        # Extract author
        author_match = re.search(r'\\author\{(.*?)\}', content, re.DOTALL)
        if author_match:
            author_raw = author_match.group(1).strip()
            # Clean up author (remove thanks, extract just name)
            author_clean = re.sub(r'\\thanks\{.*?\}', '', author_raw)
            author_clean = re.sub(r'\\href\{[^}]+\}\{[^}]+\}', '', author_clean)
            info['author'] = author_clean.strip()
            
            # Extract email if present
            email_match = re.search(r'mailto:([^}]+)\}', author_raw)
            if email_match:
                info['email'] = email_match.group(1)
            print(f"  ✓ Found author: {info.get('author', 'Unknown')}")
        
        # Extract affiliation
        affil_match = re.search(r'\\affil\{(.*?)\}', content)
        if affil_match:
            info['affiliation'] = affil_match.group(1).strip()
            print(f"  ✓ Found affiliation")
            
        # Extract keywords from abstract
        keywords_match = re.search(r'\\textbf\{Keywords:\}\s*(.*?)(?=\\end\{abstract\})', content, re.DOTALL)
        if keywords_match:
            keywords_raw = keywords_match.group(1).strip()
            keywords_clean = re.sub(r'\\noindent\s*', '', keywords_raw)
            info['keywords'] = keywords_clean.strip()
            print(f"  ✓ Found keywords")
            
        return info
    
    def build_intro_file(self, intro_content, preamble_info):
        """Build the intro.md file showing directly the Introduction (no title/abstract)"""
        # Replace LaTeX tables with markdown tables before general conversion
        intro_content = self.replace_tables_with_markdown(intro_content)
        intro_content = self.replace_bare_tabular_with_markdown(intro_content)
        # Prepare introduction content first
        intro_text = self.converter.convert_text(intro_content)
        # Fallback cleanup for simple LaTeX that may remain
        intro_text = self.fallback_simple_latex_to_md(intro_text)
        # Normalize any malformed markdown tables
        intro_text = self.normalize_markdown_tables(intro_text)
        # Remove any lingering LaTeX tabular artifacts
        intro_text = self.remove_latex_tabular_artifacts(intro_text)
        intro_text = re.sub(r'\\subsection\{([^}]+)\}', r'### \1', intro_text)
        intro_text = re.sub(r'\\subsubsection\{([^}]+)\}', r'#### \1', intro_text)
        intro_text = re.sub(r'~\\citep\{([^}]+)\}', r' [@\1]', intro_text)
        intro_text = re.sub(r'\\citep\{([^}]+)\}', r'[@\1]', intro_text)
        intro_text = re.sub(r'\\cite\{([^}]+)\}', r'[@\1]', intro_text)
        intro_text = re.sub(r'\\noindent\s*', '', intro_text)

        # Start directly with the Introduction section
        intro_md = f"## Introduction\n\n{intro_text}\n\n"

        # Optionally add a compact version notice at the end (kept for traceability)
        if 'version' in preamble_info:
            intro_md += "---\n\n"
            intro_md += "```{admonition} Version " + preamble_info['version'] + "\n"
            intro_md += ":class: tip\n"
            intro_md += f"Built: {datetime.now().strftime('%B %d, %Y')}\n"
            intro_md += "```\n\n"

        return intro_md
    
    def build_references_file(self):
        """Build the references.md file"""
        refs_md = "# References\n\n"
        refs_md += "```{bibliography}\n"
        refs_md += ":filter: docname in docnames\n"
        refs_md += "```\n"
        return refs_md
    
    def convert_section_content(self, content, title):
        """Convert a section's LaTeX content to Markdown"""
        # Replace LaTeX tables with markdown tables before general conversion
        content = self.replace_tables_with_markdown(content)
        content = self.replace_bare_tabular_with_markdown(content)
        # Convert the content
        converted = self.converter.convert_text(content)
        # Fallback cleanup for simple LaTeX that may remain
        converted = self.fallback_simple_latex_to_md(converted)
        # Normalize any malformed markdown tables
        converted = self.normalize_markdown_tables(converted)
        # Remove any lingering LaTeX tabular artifacts
        converted = self.remove_latex_tabular_artifacts(converted)
        
        # Additional cleanup for common LaTeX commands that might remain
        converted = re.sub(r'\\subsection\{([^}]+)\}', r'## \1', converted)
        converted = re.sub(r'\\subsubsection\{([^}]+)\}', r'### \1', converted)
        converted = re.sub(r'~\\citep\{([^}]+)\}', r' [@\1]', converted)
        converted = re.sub(r'\\citep\{([^}]+)\}', r'[@\1]', converted)
        converted = re.sub(r'\\citet\{([^}]+)\}', r'@\1', converted)
        converted = re.sub(r'\\cite\{([^}]+)\}', r'[@\1]', converted)
        converted = re.sub(r'\\noindent\s*', '', converted)
        converted = re.sub(r'\\vspace\{[^}]+\}', '', converted)
        converted = re.sub(r'\\hspace\{[^}]+\}', '', converted)
        
        # Build the markdown with title
        md = f"# {title}\n\n{converted}\n"
        
        return md

    def replace_tables_with_markdown(self, text: str) -> str:
        """Find LaTeX table environments and convert contained tabulars to markdown tables."""
        # Allow optional positioning options like [h] after \begin{table}
        table_pattern = r'\\begin\s*\{table\}(?:\[[^\]]*\])?(.*?)\\end\s*\{table\}'

        def replace_table(match):
            table_block = match.group(1)
            # Extract the tabular block
            tabular_match = re.search(r'\\begin\s*\{tabular\}[^{]*\{[^}]*\}(.*?)\\end\s*\{tabular\}', table_block, re.DOTALL)
            if tabular_match:
                tabular_full_match = re.search(r'\\begin\s*\{tabular\}[^{]*\{[^}]*\}.*?\\end\s*\{tabular\}', table_block, re.DOTALL)
                tabular_content = tabular_full_match.group(0) if tabular_full_match else tabular_match.group(0)
                # Extract caption if present
                caption_match = re.search(r'\\caption\s*\{([^}]+)\}', table_block)
                caption = f"**{caption_match.group(1)}**\n\n" if caption_match else ""
                return caption + self.converter.convert_table(tabular_content)
            return match.group(0)  # leave unchanged if no tabular found

        return re.sub(table_pattern, replace_table, text, flags=re.DOTALL)

    def fallback_simple_latex_to_md(self, text: str) -> str:
        """Best-effort cleanup of simple LaTeX commands left after conversion."""
        # Text styles
        text = re.sub(r'\\textbf\{([^}]*)\}', r'**\1**', text)
        text = re.sub(r'\\textit\{([^}]*)\}', r'*\1*', text)
        text = re.sub(r'\\texttt\{([^}]*)\}', r"`\1`", text)
        # Links
        text = re.sub(r'\\href\{([^}]*)\}\{([^}]*)\}', r'[\2](\1)', text)
        text = re.sub(r'\\url\{([^}]*)\}', r'<\1>', text)
        # Escapes
        text = re.sub(r'\\%', '%', text)
        text = re.sub(r'\\_', '_', text)
        text = re.sub(r'\\&', '&', text)
        # Remove noindent and stray spacing commands
        text = re.sub(r'\\noindent\s*', '', text)
        # Convert LaTeX line breaks to paragraph breaks
        text = re.sub(r'\\\\\s*', '\n\n', text)
        return text

    def remove_latex_tabular_artifacts(self, text: str) -> str:
        """Strip residual LaTeX tabular lines (begin/end, top/mid/bottomrule) possibly wrapped by pipes."""
        # Remove lines that are just LaTeX tabular control sequences, with optional leading '|'
        patterns = [
            r'^(?:\|\s*)?\\begin\s*\{tabular\}.*?(?:\|\s*)?$',
            r'^(?:\|\s*)?\\end\s*\{tabular\}\s*(?:\|\s*)?$',
            r'^(?:\|\s*)?\\toprule\s*(?:\|\s*)?$',
            r'^(?:\|\s*)?\\midrule\s*(?:\|\s*)?$',
            r'^(?:\|\s*)?\\bottomrule\s*(?:\|\s*)?$',
        ]
        for pat in patterns:
            text = re.sub(pat, '', text, flags=re.MULTILINE)
        # Collapse any excessive blank lines created
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        return text

    def normalize_markdown_tables(self, text: str) -> str:
        """Fix common table issues: ensure leading/trailing pipes and remove stray LaTeX within tables."""
        lines = text.split('\n')
        normalized_lines = []
        in_table_block = False

        def looks_like_table_row(s: str) -> bool:
            if not s.strip():
                return False
            if re.match(r'^\s*\|?\s*:?[- ]+:?\s*(\|\s*:?[- ]+:?\s*)+\|?\s*$', s):
                return True
            return s.count('|') >= 2

        for line in lines:
            # Remove residual LaTeX artifacts inside tables
            if re.search(r'\\(begin|end)\s*\{tabular\}|\\(toprule|midrule|bottomrule)', line):
                continue

            if looks_like_table_row(line):
                in_table_block = True
                s = line.strip()
                if not s.startswith('|'):
                    s = '| ' + s
                if not s.endswith('|'):
                    s = s + ' |'
                normalized_lines.append(s)
            else:
                if in_table_block and not line.strip():
                    in_table_block = False
                normalized_lines.append(line)

        result = '\n'.join(normalized_lines)
        result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)
        return result

    def replace_bare_tabular_with_markdown(self, text: str) -> str:
        """Convert standalone LaTeX tabular blocks to markdown tables, even without a table environment."""
        pattern = r'(?:\|\s*)?(\\begin\s*\{tabular\}[^{]*\{[^}]*\}.*?\\end\s*\{tabular\})(?:\s*\|)?'

        def repl(match):
            tabular_block = match.group(1)
            md = self.converter.convert_table(tabular_block)
            return f"\n{md}\n"

        return re.sub(pattern, repl, text, flags=re.DOTALL)
    
    def convert_to_jupyter_book(self):
        """Main conversion process"""
        print("=" * 60)
        print("📖 Building Jupyter Book from LaTeX Source")
        print("=" * 60)
        print(f"Source: {self.latex_file}")
        print(f"Output: {self.output_dir}/")
        print()
        
        # Check if source file exists
        if not self.latex_file.exists():
            print(f"❌ Error: LaTeX file not found: {self.latex_file}")
            return False
            
        # Read LaTeX file
        print("📄 Reading LaTeX file...")
        with open(self.latex_file, 'r', encoding='utf-8') as f:
            latex_content = f.read()
            
        # Extract document content
        doc_match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', latex_content, re.DOTALL)
        if doc_match:
            document_content = doc_match.group(1)
        else:
            print("⚠️  Warning: No \\begin{document} found, using entire file")
            document_content = latex_content
            
        # Extract preamble information
        print("\n🔍 Extracting metadata...")
        preamble_info = self.extract_preamble_info(latex_content)
        
        # Split into sections
        print("\n✂️  Splitting into sections...")
        sections = self.split_latex_by_sections(document_content)
        
        if not sections:
            print("❌ Error: No sections found in LaTeX file!")
            return False
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)
        
        # Write each section to its file
        print("\n📝 Writing markdown files...")
        for section_key, section_data in sections.items():
            output_file = self.output_dir / section_data['file']
            
            # Special handling for intro (includes abstract)
            if section_key == 'intro':
                content = self.build_intro_file(section_data['content'], preamble_info)
            else:
                # Convert section content to markdown
                content = self.convert_section_content(
                    section_data['content'], 
                    section_data['title']
                )
                
            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
                
            print(f"  ✓ Created: {output_file.name}")
            
        # Create references file
        refs_file = self.output_dir / 'references.md'
        with open(refs_file, 'w', encoding='utf-8') as f:
            f.write(self.build_references_file())
        print(f"  ✓ Created: {refs_file.name}")
        
        # Build Jupyter Book HTML
        print("\n🔨 Building Jupyter Book HTML...")
        import subprocess
        try:
            result = subprocess.run(
                ['jupyter-book', 'build', str(self.output_dir), '--all'],
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode == 0:
                print("✅ Jupyter Book built successfully!")
                
                # Copy built HTML to public/jupyter-book for deployment/UI
                public_dir = Path('public') / 'jupyter-book'
                built_html_dir = self.output_dir / '_build' / 'html'
                public_dir.mkdir(parents=True, exist_ok=True)
                
                # Remove old contents to avoid stale files
                import shutil
                for item in public_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                    else:
                        shutil.rmtree(item)
                
                # Copy new build
                shutil.copytree(built_html_dir, public_dir, dirs_exist_ok=True)
                
                print(f"\n📂 Output location: {self.output_dir}/_build/html/")
                print(f"🌐 Public copy: public/jupyter-book/")
                print(f"🌐 Open locally: public/jupyter-book/index.html")
            else:
                print("⚠️  Jupyter Book build completed with warnings:")
                if result.stderr:
                    print(result.stderr[:500])
        except subprocess.TimeoutExpired:
            print("⚠️  Jupyter Book build timed out")
        except FileNotFoundError:
            print("⚠️  jupyter-book command not found")
            print("   Install with: pip install jupyter-book")
        except Exception as e:
            print(f"⚠️  Error building Jupyter Book: {e}")
            
        print("\n" + "=" * 60)
        print("🎉 Conversion Complete!")
        print("=" * 60)
        print(f"✅ Markdown files generated in: {self.output_dir}/")
        print(f"✅ Single source of truth: {self.latex_file}")
        print("\n💡 Next steps:")
        print("   1. Edit docs/whitepaper.tex (your single source)")
        print("   2. Run this script again to regenerate everything")
        print("   3. Both PDF and Jupyter Book stay in sync!")
        
        return True


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Build Jupyter Book from LaTeX whitepaper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build with default paths
  python scripts/build_jupyter_book_from_latex.py
  
  # Build with custom paths
  python scripts/build_jupyter_book_from_latex.py --input my_paper.tex --output my_book/
        """
    )
    parser.add_argument(
        '--input', '-i',
        default='docs/whitepaper.tex',
        help='Input LaTeX file path (default: docs/whitepaper.tex)'
    )
    parser.add_argument(
        '--output', '-o',
        default='jupyter_book',
        help='Output Jupyter Book directory (default: jupyter_book)'
    )
    
    args = parser.parse_args()
    
    # Build the Jupyter Book
    builder = JupyterBookBuilder(
        latex_file=args.input,
        output_dir=args.output
    )
    
    success = builder.convert_to_jupyter_book()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

