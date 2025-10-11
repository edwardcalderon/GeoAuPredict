#!/usr/bin/env python3
"""
LaTeX to Markdown Converter for GeoAuPredict White Paper

This script converts the LaTeX whitepaper.tex to Markdown format while preserving
mathematical equations and formatting for web display with MathJax.
"""

import re
import argparse
from pathlib import Path

class LaTeXToMarkdownConverter:
    """Converts LaTeX document to Markdown format"""

    def __init__(self):
        # LaTeX commands to Markdown conversion mapping (ordered by priority)
        self.conversions = [
            # Document structure (remove)
            (r'\\documentclass\[[^\]]+\]\{[^}]+\}', ''),
            (r'\\usepackage\[[^\]]+\]\{[^}]+\}', ''),
            (r'\\usepackage\{[^}]+\}', ''),
            (r'\\geometry\{[^}]+\}', ''),
            (r'\\definecolor\{[^}]+\}\{[^}]+\}\{[^}]+\}', ''),
            (r'\\title\{([^}]+)\}', r'# \1'),
            (r'\\author\{([^}]+)\}', r'**Author:** \1  \n'),
            (r'\\date\{([^}]+)\}', r'**Date:** \1  \n\n---\n'),
            (r'\\maketitle', ''),
            (r'\\selectlanguage\{[^}]+\}', ''),
            (r'\\begin\{document\}', ''),
            (r'\\end\{document\}', ''),

            # Text colors (remove color commands but keep content) - handle nested braces with multiple passes
            (r'\\textcolor\{([^}]+)\}\{([^}]+)\}', r'\2'),

            # Sections
            (r'\\section\*?\{([^}]+)\}', r'# \1'),
            (r'\\subsection\*?\{([^}]+)\}', r'## \1'),
            (r'\\subsubsection\*?\{([^}]+)\}', r'### \1'),

            # Text formatting
            (r'\\textbf\{([^}]+)\}', r'**\1**'),
            (r'\\textit\{([^}]+)\}', r'*\1*'),
            (r'\\texttt\{([^}]+)\}', r'`\1`'),

            # Links
            (r'\\href\{([^}]+)\}\{([^}]+)\}', r'[\2](\1)'),
            (r'\\url\{([^}]+)\}', r'<\1>'),

            # Math mode (preserve for MathJax)
            (r'\\\(([^)]+)\\\)', r'$\1$'),  # Inline math
            (r'\\\[([^]]+)\\\]', r'$$\n\1\n$$'),  # Display math

            # Lists
            (r'\\begin\{itemize\}', ''),
            (r'\\end\{itemize\}', ''),
            (r'\\begin\{enumerate\}', ''),
            (r'\\end\{enumerate\}', ''),
            (r'\\item', '-'),

            # Abstract
            (r'\\begin\{abstract\}', '## Abstract\n\n'),
            (r'\\end\{abstract\}', '\n\n'),

            # Special characters
            (r'\\&', '&'),
            (r'\\%', '%'),
            (r'\\\$', '$'),
            (r'\\#', '#'),
            (r'\\_', '_'),
            (r'\\{', '{'),
            (r'\\}', '}'),

            # References/Bibliography
            (r'\\section\*\{([^}]+)\}', r'## \1'),
            (r'\\begin\{thebibliography\}[^}]*\}', '## References\n\n'),
            (r'\\end\{thebibliography\}', ''),
            (r'\\bibitem\{([^}]+)\}', r'### \1'),

            # Handle line breaks and spacing
            (r'\\newline', '\n'),
            (r'~', ' '),
        ]

    def convert_text(self, text):
        """Convert LaTeX text to Markdown"""
        # Skip comments
        if text.strip().startswith('%'):
            return ''

        # Apply conversions in order
        for pattern, replacement in self.conversions:
            text = re.sub(pattern, replacement, text)

        return text

    def convert_file(self, input_path, output_path):
        """Convert LaTeX file to Markdown"""
        print(f"Converting {input_path} to {output_path}...")

        with open(input_path, 'r', encoding='utf-8') as infile:
            content = infile.read()

        # Extract content between \begin{document} and \end{document}
        document_match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', content, re.DOTALL)
        if document_match:
            document_content = document_match.group(1)
        else:
            # Fallback: use entire content if no document environment found
            document_content = content

        # Handle title and maketitle in the preamble
        title_pattern = r'\\title\{([^}]+)\}'
        title_match = re.search(title_pattern, content)

        if title_match:
            # Extract and convert the title content
            raw_title_content = title_match.group(1)
            # Convert the title content through LaTeX conversion
            converted_title_content = self.convert_text(raw_title_content)
            # Remove title from content
            content = re.sub(title_pattern, '', content)

        # Extract author information before conversion if it appears in the document
        author_pattern = r'\\author\{([^}]+)\}'
        author_match = re.search(author_pattern, document_content)
        author_text = ""

        if author_match:
            # Extract the raw author content and convert it immediately
            raw_author_content = author_match.group(1)
            # Convert the author content through LaTeX conversion
            converted_author_content = self.convert_text(raw_author_content)
            # Clean up the converted author content
            converted_author_content = re.sub(r'\\href\{([^}]+)\}\{([^}]+)\}', r'[\2](\1)', converted_author_content)
            # Remove author from document content
            document_content = re.sub(author_pattern, '', document_content)
            # Add author info at the beginning (after title but before content)
            author_text = f"**Author:** {converted_author_content}\n\n"

        # Convert only the document content through LaTeX conversion
        converted_content = self.convert_text(document_content)

        # Clean up formatting
        lines = converted_content.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if line and not line.startswith('<!--'):
                # Handle line breaks in LaTeX
                if line.endswith('\\\\') or line.endswith('\\newline'):
                    line = line[:-2].rstrip()
                    if line:
                        cleaned_lines.append(line)
                        cleaned_lines.append('')  # Add paragraph break
                else:
                    cleaned_lines.append(line)

        # Join and clean up multiple newlines
        result = '\n'.join(cleaned_lines)
        result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)

        # Write output with header warning (as plain text, not processed through LaTeX conversion)
        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.write('# ⚠️ WORK IN PROGRESS - DOWNLOAD PDF FOR BETTER VIEW ⚠️\n\n')
            outfile.write('> **This is a constantly evolving live document, please download the PDF for the best view and version tracking.**  \n')
            outfile.write('> **To propose a change, please open an issue on GitHub, by editing the original .tex file and re-running the conversion script.**\n\n')

            # Add title if found
            if title_match:
                outfile.write(f"# {converted_title_content}\n\n")

            # Add author info if found
            if author_match:
                outfile.write(author_text)
                outfile.write('---\n\n')

            outfile.write(result)

        print(f"✅ Conversion completed: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert LaTeX whitepaper to Markdown')
    parser.add_argument('--input', '-i', default='docs/whitepaper.tex',
                       help='Input LaTeX file path')
    parser.add_argument('--output', '-o', default='docs/whitepaper.md',
                       help='Output Markdown file path')

    args = parser.parse_args()

    converter = LaTeXToMarkdownConverter()
    converter.convert_file(args.input, args.output)

if __name__ == '__main__':
    main()
