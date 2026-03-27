# Video Demo -- rvLLM vs vLLM

Side-by-side terminal recording comparing rvLLM and Python vLLM streaming inference on the same prompts.

## Prerequisites

- `tmux`, `asciinema`, `curl`, `python3`
- `agg` + `ffmpeg` (optional, for MP4 output)
- rvLLM binary built at `target/release/rvllm`
- Python vLLM installed (`pip install vllm`)

## Usage

```bash
# Full run: starts both servers, records, converts to MP4
bash bench/video_demo.sh

# Servers already running on 8000/8001
bash bench/video_demo.sh --skip-servers

# Custom prompt count
bash bench/video_demo.sh --prompts 5

# Override model/ports
MODEL=meta-llama/Llama-3-8B RVLLM_PORT=9000 VLLM_PORT=9001 bash bench/video_demo.sh
```

## Output

| File | Description |
|------|-------------|
| `bench/video/rvllm-vs-vllm.cast` | asciinema recording (playable with `asciinema play`) |
| `bench/video/rvllm-vs-vllm.mp4` | MP4 video (if agg+ffmpeg available) |

## Playback

```bash
# Terminal playback
asciinema play bench/video/rvllm-vs-vllm.cast

# Upload to asciinema.org
asciinema upload bench/video/rvllm-vs-vllm.cast
```
