# Video Demo Tool Spec

## Goal
Create a one-shot script that records a side-by-side terminal demo of rvLLM vs Python vLLM serving the same prompts, showing inference speed visually.

## Files owned (only these may be created/edited)
- bench/video_demo.sh
- bench/video/README.md
- bench/video/Dockerfile (if needed)
- bench/video/*.sh (helper scripts)

## Files forbidden to touch
- README.md
- docs/index.html
- docs/paper/*
- bench/run.sh
- Any file under crates/ or kernels/

## Context
- The solidsf-agent repo at /Users/andy/solidsf-agent has docker + video recording tooling under deploy/ and demo/
- The rvLLM server binary is at target/release/rvllm
- Python vLLM can be installed via pip install vllm
- Both servers expose OpenAI-compatible /v1/completions API
- Model: Qwen/Qwen2.5-1.5B

## Output expected
1. bench/video_demo.sh -- one-shot script that:
   - Starts rvLLM server on port 8000
   - Starts Python vLLM server on port 8001
   - Sends identical prompts to both simultaneously
   - Records terminal output showing side-by-side streaming responses
   - Uses tmux split panes or similar for visual comparison
   - Captures output as .cast (asciinema) or .mp4
   - Cleans up both servers on exit using stored PIDs
2. bench/video/README.md -- brief usage instructions
