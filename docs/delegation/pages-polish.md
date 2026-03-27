# GitHub Pages Polish Spec

## Goal
Improve docs/index.html so the B&W/Color toggle is clean and the verified-vs-pending benchmark distinction is explicit.

## Files owned (only this file may be edited)
- docs/index.html

## Files forbidden to touch
- README.md
- docs/paper/*
- bench/*
- Any file under crates/ or kernels/

## Requirements
1. The toggle button should be a clean pill-shaped toggle in the top-right corner
2. Default should be B&W (grayscale)
3. Color mode should add subtle blue table headers, syntax coloring in code blocks
4. Toggle state should persist via localStorage
5. The benchmark section (Section 5) must clearly distinguish:
   - Verified measurements (startup, binary size, RSS, VRAM, coherence)
   - Pending measurements (throughput, latency) marked with italic "Pending"
6. Add a note under the benchmark table: "Full throughput comparison is being refreshed. See bench/run.sh to reproduce."
7. Typography: serif body (Libre Baskerville or similar), sans-serif headers
8. Mobile responsive
9. No download button, no PDF links
10. Self-contained (inline CSS/JS, only external dep is font import)

## Output expected
- Updated docs/index.html that passes visual review
