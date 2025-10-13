#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"

echo "[1/4] Generating TikZ diagrams via Python..."
python3 generate_diagrams.py

echo "[2/4] Compiling TikZ-only deck..."
pdflatex -interaction=nonstopmode -halt-on-error GeoAuPredict_TikZ_Diagrams.tex >/dev/null || true

echo "[3/4] Compiling comprehensive presentation..."
pdflatex -interaction=nonstopmode -halt-on-error GeoAuPredict_Presentation.tex >/dev/null || true
pdflatex -interaction=nonstopmode -halt-on-error GeoAuPredict_Presentation.tex >/dev/null || true

echo "[4/4] Compiling Spanish presentation..."
pdflatex -interaction=nonstopmode -halt-on-error GeoAuPredict_Presentacion_ES.tex >/dev/null || true
pdflatex -interaction=nonstopmode -halt-on-error GeoAuPredict_Presentacion_ES.tex >/dev/null || true

echo "[Cleanup] Removing LaTeX auxiliary files..."
rm -f \
  GeoAuPredict_TikZ_Diagrams.{aux,log,nav,out,snm,toc,vrb,fls,fdb_latexmk,synctex.gz,bcf,run.xml,bbl,blg} \
  GeoAuPredict_Presentation.{aux,log,nav,out,snm,toc,vrb,fls,fdb_latexmk,synctex.gz,bcf,run.xml,bbl,blg} \
  GeoAuPredict_Presentacion_ES.{aux,log,nav,out,snm,toc,vrb,fls,fdb_latexmk,synctex.gz,bcf,run.xml,bbl,blg}

echo "[Done] Outputs:"
echo " - $(pwd)/GeoAuPredict_TikZ_Diagrams.pdf (optional)"
echo " - $(pwd)/GeoAuPredict_Presentation.pdf"
echo " - $(pwd)/GeoAuPredict_Presentacion_ES.pdf"

