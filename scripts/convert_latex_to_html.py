#!/usr/bin/env python3
"""
LaTeX to HTML/Markdown Converter for GeoAuPredict White Paper

This script converts the LaTeX whitepaper.tex to both HTML (using tex4ht) and Markdown formats
while preserving mathematical equations and formatting for web display.
"""

import re
import argparse
import os
import subprocess
import shutil
from datetime import datetime
from pathlib import Path

class LaTeXConverter:
    """Converts LaTeX document to HTML and Markdown formats"""

    def __init__(self):
        # LaTeX commands to Markdown conversion mapping (ordered by priority)
        self.conversions = [
            # Document structure (remove)
            (r'\\documentclass\[[^\]]+\]\{[^}]+\}', ''),
            (r'\\usepackage\[[^\]]+\]\{[^}]+\}', ''),
            (r'\\usepackage\s*\{[^}]+\}', ''),
            (r'\\geometry\s*\{[^}]+\}', ''),
            (r'\\definecolor\s*\{[^}]+\}\s*\{[^}]+\}\s*\{[^}]+\}', ''),
            (r'\\title\s*\{([^}]+)\}', r'# \1'),
            (r'\\author\s*\{([^}]+)\}', r'**Author:** \1  \n'),
            (r'\\date\s*\{([^}]+)\}', r'**Date:** \1  \n\n---\n'),
            (r'\\maketitle', ''),
            (r'\\selectlanguage\s*\{[^}]+\}', ''),
            (r'\\begin\s*\{document\}', ''),
            (r'\\end\s*\{document\}', ''),

            # Text colors (remove color commands but keep content)
            (r'\\textcolor\{([^}]+)\}\{([^}]+)\}', r'\2'),

            # Sections
            (r'\\section\*?\s*\{([^}]+)\}', r'# \1'),
            (r'\\subsection\*?\s*\{([^}]+)\}', r'## \1'),
            (r'\\subsubsection\*?\s*\{([^}]+)\}', r'### \1'),

            # Text formatting
            (r'\\textbf\s*\{([^}]+)\}', r'**\1**'),
            (r'\\textit\s*\{([^}]+)\}', r'*\1*'),
            (r'\\texttt\s*\{([^}]+)\}', r'`\1`'),

            # URL links
            (r'\\url\s*\{([^}]+)\}', r'<\1>'),
            # Handle href links
            (r'\\href\s*\{([^}]+)\}\s*\{([^}]+)\}', r'[\2](\1)'),

            # Math mode (preserve for MathJax)
            (r'\\\s*\(([^)]+)\)', r'$\1$'),  # Inline math
            (r'\\\s*\[([^]]+)\)', r'$$\n\1\n$$'),  # Display math

            # Lists
            (r'\\begin\s*\{itemize\}', ''),
            (r'\\end\s*\{itemize\}', ''),
            (r'\\begin\s*\{enumerate\}', ''),
            (r'\\end\s*\{enumerate\}', ''),
            (r'\\item\s*', '- '),

            # Abstract
            (r'\\begin\s*\{abstract\}', '## Abstract\n\n'),
            (r'\\end\s*\{abstract\}', '\n\n'),

            # Special characters and commands
            (r'\\&', '&'),
            (r'\\%', '%'),
            (r'\\\$', '$'),
            (r'\\noindent', ''),  # Remove noindent
            (r'\\vspace\s*\{[^}]+\}', '\n\n'),  # Convert vspace to spacing
            (r'\\newpage', '\n\n---\n\n'),  # Convert newpage to separator
            (r'\\fbox\s*\{([^}]+)\}', r'> \1'),  # Convert fbox to blockquote
            (r'\\parbox[^}]*\{[^}]*\}', ''),  # Remove parbox
            (r'\\#', '#'),
            (r'\\_', '_'),
            (r'\\{', '{'),
            (r'\\}', '}'),

            # Citations
            (r'~\\citep\s*\{([^}]+)\}', r'[¹](#ref-\1)'),  # Tilde citation
            (r'\\citep\s*\{([^}]+)\}', r'[²](#ref-\1)'),  # Regular citation

            # Version and text size formatting
            (r'\\\{\\small\s+([^}]+)\}', r'*\1*'),  # Handle {\small Version X.X.X}
            (r'\\\{\\small\s+([^}]+)\\\}', r'*\1*'),  # Handle {\small Version X.X.X}
            (r'\\small\s+', ''),  # Remove \small commands
            (r'\\large\s+', ''),  # Remove \large commands
            (r'\\\\\s*\{\s*', ' '),  # Handle \{ patterns
            (r'\\\\\s*\}\s*', ''),   # Handle \} patterns
            (r'\\ddmmyyyydate', ''),  # Remove date command, handled in convert_text
            # (r'\\today', datetime.now().strftime('%B %d, %Y')),  # Replace \today with actual date - handled in convert_text
            # (r'\\currenttime', datetime.now().strftime('%H:%M')),  # Replace \currenttime with actual time - handled in convert_text

            # References/Bibliography
            (r'\\bibitem\{([^}]+)\}', r'### \1'),
            (r'\\bibliographystyle\{[^}]+\}', r'## References\n\n'),
            (r'\\bibliography\{([^}]+)\}', r'*(Bibliography generated from \1.bib)*\n\n'),

            # Special commands - handle thanks with href first (more specific pattern)
            (r'[\s]*\\thanks\{\\href\{mailto:([^}]+)\}\{([^}]+)\}\}', r'[*thanks: \2*](mailto:\1)'),  # Handle \thanks with href
            (r'\\thanks\{([^}]+)\}', r'*\1*'),  # Handle \thanks command
        ]

    def convert_text(self, text):
        """Convert LaTeX text to Markdown"""
        # Skip comments
        if text.strip().startswith('%'):
            return ''

        # Handle multiline \href commands first (this should be handled by conversions now)
        # text = re.sub(r'\\href\{([^}]+)\}\s*\{([^{}]*?(?:\n[^{}]*?)*?)\}', r'[\2](\1)', text, flags=re.DOTALL)

        # Handle special thanks with href patterns before general pattern matching
        text = re.sub(r'[\s]*\\thanks\{\\href\{mailto:([^}]+)\}\{([^}]+)\}\}', r'[*thanks: \2*](mailto:\1)', text)

        # Handle date and time patterns
        current_datetime = datetime.now()
        text = re.sub(r'\\today\\ at \\currenttime', f'{current_datetime.strftime("%B %d, %Y")} at {current_datetime.strftime("%H:%M")}', text)
        text = re.sub(r'\\today', current_datetime.strftime('%B %d, %Y'), text)
        text = re.sub(r'\\currenttime', current_datetime.strftime('%H:%M'), text)

        for pattern, replacement in self.conversions:
            text = re.sub(pattern, replacement, text, flags=re.DOTALL)

        # Clean up backslashes that may remain from LaTeX commands
        text = re.sub(r'\\\\([a-zA-Z]+|\\\\)', r'\1', text)
        # Additional cleanup for specific patterns
        text = re.sub(r'\\\\at', 'at', text)
        text = re.sub(r'\\\\([0-9])', r'\1', text)  # Fix version numbers like \1

        return text

    def convert_table(self, tabular_content):
        """Convert LaTeX tabular environment to markdown table"""
        # Remove LaTeX table commands
        content = re.sub(r'\\begin\s*\{tabular\}[^{]*\{[^}]*\}', '', tabular_content)
        content = re.sub(r'\\end\s*\{tabular\}', '', content)
        content = re.sub(r'\\toprule|\\midrule|\\bottomrule|\\hline', '', content)
        content = re.sub(r'\\centering', '', content)

        # Split into rows, handling both \\ and \newline
        rows = []
        for row in content.split('\\\\'):
            row = row.strip()
            if row and not row.startswith('\\newline'):
                rows.append(row)

        if not rows:
            return ''

        # Process each row
        markdown_rows = []
        for i, row in enumerate(rows):
            # Remove LaTeX column specifications and formatting
            row = re.sub(r'\\textbf\{([^}]+)\}', r'**\1**', row)
            row = re.sub(r'\\textit\{([^}]+)\}', r'*\1*', row)
            row = re.sub(r'\\texttt\{([^}]+)\}', r'`\1`', row)

            # Split by & (column separator), but handle nested braces
            cols = self._split_table_columns(row)

            # Clean up each column
            clean_cols = []
            for col in cols:
                # Remove leading/trailing whitespace and LaTeX commands
                col = re.sub(r'^\\|^\s*|\s*$', '', col)
                col = re.sub(r'\\\s*$', '', col)  # Remove trailing \\
                # Handle href links in table cells
                col = re.sub(r'\\href\{([^}]+)\}\{([^}]+)\}', r'[\2](\1)', col)
                clean_cols.append(col)

            if clean_cols and any(clean_cols):  # Only add non-empty rows
                markdown_rows.append('| ' + ' | '.join(clean_cols) + ' |')

        if not markdown_rows:
            return ''

        # Add header separator after first row (header)
        if len(markdown_rows) > 1:
            num_cols = len(markdown_rows[0].split('|')[1:-1])
            markdown_rows.insert(1, '|' + '|'.join(['---'] * num_cols) + '|')

        return '\n'.join(markdown_rows)

    def _split_table_columns(self, row):
        """Split table row by & while respecting nested braces"""
        cols = []
        current_col = ""
        brace_level = 0

        for char in row:
            if char == '{' or char == '[':
                brace_level += 1
                current_col += char
            elif char == '}' or char == ']':
                brace_level -= 1
                current_col += char
            elif char == '&' and brace_level == 0:
                cols.append(current_col)
                current_col = ""
            else:
                current_col += char

        if current_col:
            cols.append(current_col)

        return [col.strip() for col in cols]

    def convert_to_html(self, input_path, output_dir):
        """Convert LaTeX to HTML using tex4ht"""
        print("🔄 Converting LaTeX to HTML using tex4ht...")

        # Create a temporary directory for tex4ht processing
        temp_dir = Path(output_dir) / "temp_html"
        temp_dir.mkdir(exist_ok=True)

        try:
            # Copy the LaTeX file to temp directory
            shutil.copy2(input_path, temp_dir / "whitepaper.tex")

            # Change to temp directory for processing
            original_cwd = os.getcwd()
            os.chdir(temp_dir)

            # Run htlatex to convert LaTeX to HTML
            cmd = [
                'htlatex',
                'whitepaper.tex',
                'html'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                print(f"⚠️  htlatex warning: {result.stderr}")
                print("Continuing with fallback markdown conversion...")

            # Check if HTML file was created
            html_file = temp_dir / "whitepaper.html"
            if html_file.exists():
                # Read and clean up the HTML
                with open(html_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                # Clean up the HTML (remove tex4ht specific styling, keep content)
                html_content = self._clean_html(html_content)

                # Copy to output directory
                final_html = Path(output_dir) / "whitepaper.html"
                with open(final_html, 'w', encoding='utf-8') as f:
                    f.write(html_content)

                print(f"✅ HTML conversion completed: {final_html}")
                return True
            else:
                print("❌ HTML conversion failed - no output file generated")
                return False

        except subprocess.TimeoutExpired:
            print("❌ HTML conversion timed out")
            return False
        except Exception as e:
            print(f"❌ HTML conversion error: {e}")
            return False
        finally:
            # Clean up temp directory
            os.chdir(original_cwd)
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def _clean_html(self, html_content):
        """Clean up tex4ht generated HTML to make it more web-friendly"""
        # Remove tex4ht meta tags and scripts
        html_content = re.sub(r'<meta[^>]*>', '', html_content)
        html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL)

        # Remove tex4ht styling but keep content
        html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL)

        # Fix image paths (if any)
        html_content = re.sub(r'src="([^"]+)"', r'src="/\1"', html_content)

        # Clean up extra whitespace
        html_content = re.sub(r'\n\s*\n\s*\n', '\n\n', html_content)

        return html_content

    def parse_bibliography(self, bib_file_path):
        """Parse .bib file and return markdown formatted bibliography"""
        if not os.path.exists(bib_file_path):
            return "*Bibliography entries not found*"

        bibliography = []
        current_entry = {}

        with open(bib_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split by @ entries
        entries = re.split(r'\n\s*(?=@)', content.strip())

        for entry in entries:
            if not entry.strip() or entry.startswith('@'):
                continue

            # Extract key information
            key_match = re.search(r'@\w+\{([^,]+)', entry)
            title_match = re.search(r'title=\{([^}]+)\}', entry)
            author_match = re.search(r'author=\{([^}]+)\}', entry)
            journal_match = re.search(r'journal=\{([^}]+)\}', entry)
            year_match = re.search(r'year=\{([^}]+)\}', entry)
            volume_match = re.search(r'volume=\{([^}]+)\}', entry)
            pages_match = re.search(r'pages=\{([^}]+)\}', entry)
            publisher_match = re.search(r'publisher=\{([^}]+)\}', entry)
            url_match = re.search(r'url=\{([^}]+)\}', entry)

            if key_match and title_match and author_match:
                key = key_match.group(1)
                title = title_match.group(1)
                authors = author_match.group(1)

                # Format authors (simplified)
                author_list = [a.strip() for a in authors.split(' and ')]
                if len(author_list) <= 2:
                    formatted_authors = ' and '.join(author_list)
                else:
                    formatted_authors = author_list[0] + ' et al.'

                # Build citation with URL link if available
                citation = f"**{formatted_authors}** ({year_match.group(1) if year_match else 'Year unknown'}). *{title}*."

                if journal_match:
                    journal = journal_match.group(1)
                    vol_pages = ""
                    if volume_match:
                        vol_pages += f" {volume_match.group(1)}"
                    if pages_match:
                        vol_pages += f":{pages_match.group(1)}"
                    citation += f" {journal}{vol_pages}."

                if publisher_match:
                    citation += f" {publisher_match.group(1)}."

                # Add URL link if available
                if url_match:
                    url = url_match.group(1)
                    citation += f" [🔗]({url})"

                bibliography.append(f"- {citation}")

        if bibliography:
            return "\n".join(bibliography)
        else:
            return "*No bibliography entries found*"

    def convert_file(self, input_path, output_path):
        print(f"Converting {input_path} to {output_path}...")

        with open(input_path, 'r', encoding='utf-8') as infile:
            content = infile.read()
        # Replace bibliography placeholders BEFORE document extraction
        # Handle both single line and multi-line bibliography commands
        content = re.sub(r'\\bibliographystyle\{[^}]+\}\s*\\bibliography\{([^}]+)\}',
                        '', content)

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
        version = "Unknown"

        if title_match:
            # Extract and convert the title content
            raw_title_content = title_match.group(1)
            # Extract version from title if present
            version_match = re.search(r'\\small\s+Version\s+([\\d.]+)', raw_title_content)
            if version_match:
                version = version_match.group(1)
            # Convert the title content through LaTeX conversion
            converted_title_content = self.convert_text(raw_title_content)
            # Clean up any remaining LaTeX formatting in title
            converted_title_content = re.sub(r'\{\\small\s+([^}]+)\}', r'*\1*', converted_title_content)
            converted_title_content = re.sub(r'\\small\s+', '', converted_title_content)
            # Remove title from content
            content = re.sub(title_pattern, '', content)

        # Extract author and affiliation information
        author_pattern = r'\\author\{([^}]+)\}'
        affil_pattern = r'\\affil\{([^}]+)\}'
        date_pattern = r'\\date\{([^}]+)\}'

        author_match = re.search(author_pattern, content)
        affil_matches = re.findall(affil_pattern, content)
        date_match = re.search(date_pattern, content)

        author_info = {
            'name': 'Unknown Author',
            'affiliations': [],
            'email': '',
            'date': 'Unknown Date',
            'version': version
        }

        if author_match:
            author_text = self.convert_text(author_match.group(1))
            # Extract email from thanks/href structure if present (now in markdown format)
            email_match = re.search(r'\\href\{mailto:([^}]+)\}\{([^}]+)\}', author_text)
            if email_match:
                author_info['email'] = email_match.group(2)  # The display text
                author_info['name'] = re.sub(r'\\thanks\{.*?\}', '', author_text).strip()
            else:
                # Check for markdown format [display](mailto:email)
                md_email_match = re.search(r'\[([^\]]+)\]\(mailto:([^)]+)\)', author_text)
                if md_email_match:
                    author_info['email'] = md_email_match.group(1)  # The display text
                    author_info['name'] = re.sub(r'\[([^\]]+)\]\(mailto:[^)]+\)', '', author_text).strip()
                else:
                    author_info['name'] = author_text
            # Remove author from content
            content = re.sub(author_pattern, '', content)

        if affil_matches:
            for affil in affil_matches:
                converted_affil = self.convert_text(affil)
                # Check if it's an email link - href should already be converted by convert_text
                email_match = re.search(r'\[([^\]]+)\]\(([^)]+)\)', converted_affil)
                if email_match and 'mailto:' in email_match.group(2):
                    author_info['email'] = email_match.group(1)  # The email address shown
                else:
                    author_info['affiliations'].append(converted_affil)
            # Remove affiliations from content
            content = re.sub(affil_pattern, '', content)

        if date_match:
            date_content = date_match.group(1)
            # Extract version from date if present
            version_match = re.search(r'Version\s+([0-9.]+)', date_content)
            if version_match:
                version = version_match.group(1)
            author_info['date'] = self.convert_text(date_match.group(1))
            # Remove date from content
            content = re.sub(date_pattern, '', content)

        # Handle tables before general conversion
        # Find all table environments (containing tabular) and convert them
        table_pattern = r'\\begin\s*\{table\}[^{]*\}(.*?)\\end\s*\{table\}'
        def replace_table(match):
            table_content = match.group(1)
            # Extract just the tabular part for conversion
            tabular_match = re.search(r'\\begin\s*\{tabular\}[^{]*\{[^}]*\}(.*?)\\end\s*\{tabular\}', table_content, re.DOTALL)
            if tabular_match:
                tabular_content = tabular_match.group(0)
                # Extract caption if present
                caption_match = re.search(r'\\caption\s*\{([^}]+)\}', table_content)
                caption = f"**{caption_match.group(1)}**\n\n" if caption_match else ""
                return caption + self.convert_table(tabular_content)
            return ""

        document_content = re.sub(table_pattern, replace_table, document_content, flags=re.DOTALL)

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
            outfile.write('> **⚠️ WORK IN PROGRESS ⚠️** \n')
            outfile.write('> This is a constanly evolving document. \n')
            outfile.write('> To propose a change, please open an PR on GitHub, by editing the original whitepaper.tex file and re-running the version manager script. \n\n')

            # Add title if found
            if title_match:
                outfile.write(f"# {converted_title_content}\n\n")

            # Add author info if found
            if author_match:
                # Format author section - compressed version
                if author_info['version'] != "Unknown":
                    author_section = f"## {author_info['version']}\n\n"
                else:
                    author_section = "## Document Information\n\n"

                # Make author name clickable with email if available
                if author_info['email']:
                    author_section += f"**Author:** [{author_info['name']}]({author_info['email']})\n\n"
                else:
                    author_section += f"**Author:** {author_info['name']}\n\n"

                if author_info['affiliations']:
                    author_section += "**Affiliations:**\n"
                    for i, affil in enumerate(author_info['affiliations'], 1):
                        author_section += f"- {affil}\n"
                    author_section += "\n"

                if author_info['date'] != "Unknown Date":
                    if author_info['version'] != "Unknown":
                        author_section += f"**Last Modified:** {author_info['date']} ({author_info['version']})\n\n"
                    else:
                        author_section += f"**Last Modified:** {author_info['date']}\n\n"

                author_section += "---\n\n"

                outfile.write(author_section)

            outfile.write(result)

        print(f"✅ Conversion completed: {output_path}")

        # Also generate HTML version using tex4ht
        html_output_dir = Path(output_path).parent
        html_success = self.convert_to_html(input_path, str(html_output_dir))

        if html_success:
            print(f"✅ HTML version also created: {html_output_dir}/whitepaper.html")

def main():
    parser = argparse.ArgumentParser(description='Convert LaTeX whitepaper to HTML and Markdown')
    parser.add_argument('--input', '-i', default='docs/whitepaper.tex',
                       help='Input LaTeX file path')
    parser.add_argument('--output', '-o', default='docs/whitepaper.md',
                       help='Output Markdown file path')

    args = parser.parse_args()

    converter = LaTeXConverter()
    converter.convert_file(args.input, args.output)

if __name__ == '__main__':
    main()
