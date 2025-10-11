#!/usr/bin/env python3
"""
LaTeX to Markdown Converter for GeoAuPredict White Paper

This script converts the LaTeX whitepaper.tex to Markdown format while preserving
mathematical equations and formatting for web display with MathJax.
"""

import re
import argparse
import os
from datetime import datetime
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

            # URL links (mailto will be handled separately)
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

            # Version and text size formatting
            (r'\\\{\\small\s+([^}]+)\}', r'*\1*'),  # Handle {\small Version X.X.X}
            (r'\\\{\\small\s+([^}]+)\\\}', r'*\1*'),  # Handle {\small Version X.X.X}
            (r'\\small\s+', ''),  # Remove \small commands
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

        # Handle multiline \href commands first
        text = re.sub(r'\\href\{([^}]+)\}\s*\{([^{}]*?(?:\n[^{}]*?)*?)\}', r'[\2](\1)', text, flags=re.DOTALL)

        # Handle special thanks with href patterns before general pattern matching
        text = re.sub(r'[\s]*\\thanks\{\\href\{mailto:([^}]+)\}\{([^}]+)\}\}', r'[*thanks: \2*](mailto:\1)', text)

        # Handle date and time patterns
        current_datetime = datetime.now()
        text = re.sub(r'\\today\\ at \\currenttime', f'{current_datetime.strftime("%B %d, %Y")} at {current_datetime.strftime("%H:%M")}', text)
        text = re.sub(r'\\today', current_datetime.strftime('%B %d, %Y'), text)
        text = re.sub(r'\\currenttime', current_datetime.strftime('%H:%M'), text)

        for pattern, replacement in self.conversions:
            text = re.sub(pattern, replacement, text)

        # Clean up backslashes that may remain from LaTeX commands
        text = re.sub(r'\\([a-zA-Z]+|\\)', r'\1', text)
        # Additional cleanup for specific patterns
        text = re.sub(r'\\at', 'at', text)
        text = re.sub(r'\\([0-9])', r'\1', text)  # Fix version numbers like \1

        return text

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
            outfile.write('> **⚠️ WORK IN PROGRESS** \n')
            outfile.write('> This is a live document, constanly evolving. \n')
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
