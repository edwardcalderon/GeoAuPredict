# White Paper Build Scripts

## Quick Reference

### Single Command to Build Everything

```bash
python scripts/build_whitepaper.py
```

This builds:
- ✅ PDF (`public/whitepaper-latex.pdf`)
- ✅ Markdown (`docs/whitepaper.md`)
- ✅ Jupyter Book (`jupyter_book/_build/html/`)

### Individual Build Commands

```bash
# Build only Jupyter Book from LaTeX
python scripts/build_jupyter_book_from_latex.py

# Build only Markdown from LaTeX
python scripts/convert_latex_to_markdown.py

# Build only PDF from LaTeX
python scripts/compile_latex_to_pdf.py
```

## Important: Single Source of Truth

**EDIT ONLY:** `docs/whitepaper.tex`

**DON'T EDIT:**
- ❌ `jupyter_book/*.md` (auto-generated)
- ❌ `docs/whitepaper.md` (auto-generated)
- ❌ `public/whitepaper-latex.pdf` (auto-generated)

## Available Scripts

### `build_whitepaper.py`
**Purpose:** Complete build of all whitepaper formats  
**Usage:** `python scripts/build_whitepaper.py`  
**Outputs:** PDF, Markdown, Jupyter Book

### `build_jupyter_book_from_latex.py` ⭐ NEW
**Purpose:** Convert LaTeX to Jupyter Book (split by sections)  
**Usage:** `python scripts/build_jupyter_book_from_latex.py`  
**Features:**
- Extracts version, author, abstract from LaTeX
- Splits `\section{}` into separate .md files
- Builds Jupyter Book HTML
- Maintains single source of truth

**Section Mapping:**
```
LaTeX Section              → Jupyter Book File
─────────────────────────────────────────────
\section{Introduction}     → intro.md
\section{Methods}          → methodology.md
\section{Results}          → results.md
\section{Discussion}       → discussion.md
\section{Conclusion}       → conclusion.md
\bibliography{}            → references.md
```

### `convert_latex_to_markdown.py`
**Purpose:** Convert LaTeX to single Markdown file  
**Usage:** `python scripts/convert_latex_to_markdown.py`  
**Output:** `docs/whitepaper.md`

### `compile_latex_to_pdf.py`
**Purpose:** Compile LaTeX to PDF  
**Usage:** `python scripts/compile_latex_to_pdf.py`  
**Output:** `public/whitepaper-latex.pdf`  
**Requirements:** LaTeX installation (texlive)

### `whitepaper_version_manager.py`
**Purpose:** Manage whitepaper versioning  
**Usage:** `python scripts/whitepaper_version_manager.py`

## Workflow

```
1. Edit
   └─ docs/whitepaper.tex

2. Build
   └─ python scripts/build_whitepaper.py

3. All outputs updated automatically!
   ├─ public/whitepaper-latex.pdf
   ├─ docs/whitepaper.md
   └─ jupyter_book/
      ├─ intro.md
      ├─ methodology.md
      ├─ results.md
      ├─ discussion.md
      ├─ conclusion.md
      ├─ references.md
      └─ _build/html/
```

## Installation Requirements

### For Jupyter Book
```bash
pip install jupyter-book
```

### For PDF Compilation
```bash
# Ubuntu/Debian
sudo apt-get install texlive-full

# macOS
brew install mactex
```

### For Markdown Conversion
```bash
# No additional requirements (uses built-in converter)
```

## Detailed Documentation

See [`docs/WHITEPAPER_WORKFLOW.md`](../docs/WHITEPAPER_WORKFLOW.md) for:
- Complete workflow guide
- Troubleshooting
- Best practices
- Advanced usage

## Version Management

The version is controlled in `docs/whitepaper.tex`:

```latex
\date{\today\ \\ {\small Version 1.0.6}}
```

When you build, this version appears in:
- PDF header
- Jupyter Book intro page
- Web markdown metadata

## Example: Update Whitepaper

```bash
# 1. Edit the source
vim docs/whitepaper.tex

# 2. Build all formats
python scripts/build_whitepaper.py

# 3. Test locally
open jupyter_book/_build/html/index.html
open public/whitepaper-latex.pdf

# 4. Commit everything
git add docs/whitepaper.tex jupyter_book/*.md public/whitepaper-latex.pdf
git commit -m "Update whitepaper content"
git push
```

## Troubleshooting

### "jupyter-book command not found"
```bash
pip install jupyter-book
```

### "pdflatex not found"
```bash
sudo apt-get install texlive-full  # Ubuntu
brew install mactex                 # macOS
```

### Jupyter Book build fails
```bash
# Clean and rebuild
jupyter-book clean jupyter_book
python scripts/build_jupyter_book_from_latex.py
```

### Content not updating
```bash
# Force full rebuild
rm -rf jupyter_book/_build
python scripts/build_whitepaper.py
```

## Questions?

- Full documentation: `docs/WHITEPAPER_WORKFLOW.md`
- Data scripts: `scripts/data_ingestion/README.md`
- General docs: `docs/README.md`

