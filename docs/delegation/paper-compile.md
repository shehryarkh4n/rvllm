# Paper Compile and Sanity Check Spec

## Goal
Compile both LaTeX paper variants and fix any compilation errors. Sanity check content for obvious issues.

## Files owned (only these may be edited)
- docs/paper/rvllm.tex
- docs/paper/rvllm-bw.tex
- docs/paper/references.bib
- docs/paper/*.pdf (generated output)
- docs/paper/*.aux, *.bbl, *.blg, *.log (build artifacts)

## Files forbidden to touch
- README.md
- docs/index.html
- bench/*
- Any file under crates/ or kernels/

## Requirements
1. Both rvllm.tex and rvllm-bw.tex must compile cleanly with pdflatex + bibtex
2. If pdflatex is not available locally, install texlive-base or report what's needed
3. Fix any LaTeX errors (missing packages, bad references, etc.)
4. Verify both PDFs render correctly (sections, tables, references)
5. The GPU benchmark table should show only verified numbers:
   - Startup: ~7s
   - Binary: 15 MB
   - CPU RSS: 333 MB
   - GPU VRAM: 6,357 MiB
   - Output quality: Coherent
   - Throughput: Pending
   - P50/P95: Pending
6. No fabricated numbers anywhere

## Output expected
- docs/paper/rvllm.pdf (color version)
- docs/paper/rvllm-bw.pdf (B&W version)
- Report of any fixes made
- Confirmation that both compile cleanly
