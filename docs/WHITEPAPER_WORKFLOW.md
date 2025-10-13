# White Paper Workflow - Single Source of Truth

## Overview

The GeoAuPredict white paper uses **LaTeX as the single source of truth**. All other formats (Markdown, Jupyter Book, PDF) are automatically generated from `docs/whitepaper.tex`.

```
                     ┌─────────────────────┐
                     │ docs/whitepaper.tex │ ← EDIT THIS ONLY
                     │  (SINGLE SOURCE)    │
                     └──────────┬──────────┘
                                │
                  ┌─────────────┴─────────────┐
                  │  python scripts/          │
                  │  build_whitepaper.py      │
                  └─────────────┬─────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
    ┌─────────────────┐ ┌─────────────┐ ┌──────────────┐
    │ docs/           │ │ jupyter_book│ │ public/      │
    │ whitepaper.md   │ │ /*.md       │ │ whitepaper   │
    │                 │ │ (split by   │ │ -latex.pdf   │
    │ (web markdown)  │ │  sections)  │ │              │
    └─────────────────┘ └──────┬──────┘ └──────────────┘
                               │
                               ▼
                     ┌─────────────────┐
                     │ jupyter_book/   │
                     │ _build/html/    │
                     │ (interactive    │
                     │  web book)      │
                     └─────────────────┘
```

## Why This Approach?

### Before (2 Sources of Truth ❌)
- **Problem:** `docs/whitepaper.tex` for PDF + separate `jupyter_book/*.md` files
- **Issues:** 
  - Content gets out of sync
  - Have to edit content twice
  - Version numbers might differ
  - Hard to maintain consistency

### After (1 Source of Truth ✅)
- **Solution:** `docs/whitepaper.tex` is the ONLY file you edit
- **Benefits:**
  - Edit once, publish everywhere
  - Guaranteed consistency across all formats
  - Version number always matches
  - LaTeX quality for PDF, accessible web formats too

## Workflow

### Daily Usage

1. **Edit the LaTeX source:**
   ```bash
   # Open and edit the ONLY file that matters
   vim docs/whitepaper.tex
   ```

2. **Build all outputs:**
   ```bash
   # One command generates everything
   python scripts/build_whitepaper.py
   ```

3. **Done!** All formats are updated:
   - ✅ PDF: `public/whitepaper-latex.pdf`
   - ✅ Markdown: `docs/whitepaper.md`
   - ✅ Jupyter Book: `jupyter_book/_build/html/`

### Individual Build Commands

If you only want to build specific formats:

```bash
# Just Jupyter Book
python scripts/build_jupyter_book_from_latex.py

# Just Markdown
python scripts/convert_latex_to_markdown.py

# Just PDF
python scripts/compile_latex_to_pdf.py
```

## File Structure

```
GeoAuPredict/
├── docs/
│   ├── whitepaper.tex          ← EDIT THIS (single source)
│   ├── whitepaper.md           ← AUTO-GENERATED
│   └── references.bib          ← Bibliography (edit as needed)
│
├── jupyter_book/
│   ├── intro.md                ← AUTO-GENERATED (from LaTeX \section{Introduction})
│   ├── methodology.md          ← AUTO-GENERATED (from LaTeX \section{Methods})
│   ├── results.md              ← AUTO-GENERATED (from LaTeX \section{Results})
│   ├── discussion.md           ← AUTO-GENERATED (from LaTeX \section{Discussion})
│   ├── conclusion.md           ← AUTO-GENERATED (from LaTeX \section{Conclusion})
│   ├── references.md           ← AUTO-GENERATED
│   ├── _config.yml             ← Edit for Jupyter Book settings
│   ├── _toc.yml                ← Edit for table of contents
│   └── _build/html/            ← Built HTML output
│
├── public/
│   └── whitepaper-latex.pdf    ← AUTO-GENERATED PDF
│
└── scripts/
    ├── build_whitepaper.py                    ← Main build script
    ├── build_jupyter_book_from_latex.py       ← LaTeX → Jupyter Book
    ├── convert_latex_to_markdown.py           ← LaTeX → Markdown
    └── compile_latex_to_pdf.py                ← LaTeX → PDF
```

## How It Works

### 1. LaTeX Parsing

The `build_jupyter_book_from_latex.py` script:
- Reads `docs/whitepaper.tex`
- Extracts metadata (title, author, version, abstract)
- Splits content by `\section{}` commands
- Converts LaTeX to Markdown

### 2. Section Mapping

LaTeX sections are automatically mapped to Jupyter Book files:

| LaTeX Section                | Jupyter Book File  |
|------------------------------|-------------------|
| `\section{Introduction}`     | `intro.md`        |
| `\section{Methods}`          | `methodology.md`  |
| `\section{Results}`          | `results.md`      |
| `\section{Discussion}`       | `discussion.md`   |
| `\section{Conclusion}`       | `conclusion.md`   |
| `\bibliography{}`            | `references.md`   |

### 3. Version Synchronization

The version is extracted from `docs/whitepaper.tex`:

```latex
\date{\today\ \\ {\small Version 1.0.6}}
```

This version appears in:
- ✅ PDF header
- ✅ Jupyter Book intro page
- ✅ Web markdown

## Common Tasks

### Adding a New Section

1. Edit `docs/whitepaper.tex`:
   ```latex
   \section{My New Section}
   Content goes here...
   ```

2. Update `jupyter_book/_toc.yml` to include the new section:
   ```yaml
   chapters:
     - file: methodology
     - file: my_new_section  # Add this
     - file: results
   ```

3. Rebuild:
   ```bash
   python scripts/build_whitepaper.py
   ```

### Updating the Version

1. Edit `docs/whitepaper.tex`:
   ```latex
   \date{\today\ \\ {\small Version 1.0.7}}
   ```

2. Rebuild - version updates everywhere:
   ```bash
   python scripts/build_whitepaper.py
   ```

### Adding Citations

1. Add to `docs/references.bib`:
   ```bibtex
   @article{my_paper,
     author = {Smith, J.},
     title = {Great Paper},
     year = {2025}
   }
   ```

2. Cite in `docs/whitepaper.tex`:
   ```latex
   This is important~\citep{my_paper}.
   ```

3. Rebuild - citations appear in all formats:
   ```bash
   python scripts/build_whitepaper.py
   ```

## Troubleshooting

### Jupyter Book Build Fails

```bash
# Install Jupyter Book
pip install jupyter-book

# Or in your virtual environment
source venv/bin/activate
pip install jupyter-book
```

### PDF Build Fails

```bash
# Ubuntu/Debian
sudo apt-get install texlive-full

# macOS
brew install mactex

# Then rebuild
python scripts/build_whitepaper.py
```

### Content Not Updating

```bash
# Force rebuild Jupyter Book
jupyter-book clean jupyter_book
python scripts/build_jupyter_book_from_latex.py

# Or rebuild everything
python scripts/build_whitepaper.py
```

## Best Practices

1. **Always edit `docs/whitepaper.tex` directly**
   - ❌ Don't edit `jupyter_book/*.md` files
   - ❌ Don't edit `docs/whitepaper.md`
   - ✅ Edit `docs/whitepaper.tex` only

2. **Run the build after changes**
   ```bash
   python scripts/build_whitepaper.py
   ```

3. **Commit both source and outputs**
   ```bash
   git add docs/whitepaper.tex
   git add jupyter_book/*.md
   git add public/whitepaper-latex.pdf
   git commit -m "Update whitepaper to v1.0.7"
   ```

4. **Test locally before pushing**
   ```bash
   # Build everything
   python scripts/build_whitepaper.py
   
   # Check Jupyter Book locally
   open jupyter_book/_build/html/index.html
   
   # Check PDF
   open public/whitepaper-latex.pdf
   ```

## Version History Integration

The build process integrates with the project's version management:

```bash
# Update version and rebuild
python scripts/bump_project_version.py --patch
python scripts/build_whitepaper.py

# Version appears in:
# - docs/whitepaper.tex
# - jupyter_book/intro.md (auto)
# - public/whitepaper-latex.pdf (auto)
# - VERSION file
```

## Deployment

The built outputs are automatically deployed:

1. **Jupyter Book** → GitHub Pages (`/GeoAuPredict/jupyter-book/`)
2. **PDF** → Public download (`/GeoAuPredict/versions/whitepaper-v1.0.6.pdf`)
3. **Markdown** → Documentation site

All from the same LaTeX source!

## Summary

✅ **Single Source:** Edit `docs/whitepaper.tex` only  
✅ **One Command:** `python scripts/build_whitepaper.py`  
✅ **Multiple Outputs:** PDF, Jupyter Book, Markdown  
✅ **Always Synced:** Version, content, citations match everywhere  
✅ **LaTeX Quality:** Professional PDF while maintaining web accessibility

---

**Questions?** Check `scripts/README.md` or open an issue.

