#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text())


def parse_benchmark(path: Path):
    if not path.exists():
        return None
    data = load_json(path)
    results = data.get("results", [])
    if not results:
        return None
    row = results[0]
    return {
        "engine": data.get("engine"),
        "model": data.get("model"),
        "output_len": data.get("output_len", data.get("max_tokens")),
        "n": int(row["n"]),
        "tok_per_sec": float(row["tok_per_sec"]),
        "total_tokens": int(row["total_tokens"]),
        "elapsed_ms": int(row["elapsed_ms"]),
        "failed": int(row.get("failed", 0)),
    }


def parse_nsys_table(path: Path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("Time") or line.startswith("-"):
            continue
        parts = re.split(r"\s{2,}", line)
        if len(parts) < 3:
            continue
        try:
            pct = float(parts[0].replace("%", "").replace(",", ""))
        except ValueError:
            continue
        rows.append((pct, parts))
    return rows


def parse_kernel_summary(path: Path):
    kernels = []
    for pct, parts in parse_nsys_table(path):
        if len(parts) < 2:
            continue
        name = parts[-1]
        total_time_ns = 0
        for field in parts[1:-1]:
            cleaned = field.replace(",", "")
            if cleaned.isdigit():
                total_time_ns = int(cleaned)
                break
        kernels.append(
            {
                "pct": pct,
                "total_time_ns": total_time_ns,
                "name": name,
            }
        )
    kernels.sort(key=lambda x: x["pct"], reverse=True)
    return kernels


def parse_mem_summary(path: Path):
    ops = {}
    for pct, parts in parse_nsys_table(path):
        if not parts:
            continue
        name = parts[-1]
        ops[name] = {"pct": pct}
    return ops


def color_for_engine(engine: str) -> str:
    return {"rvllm": "#1f77b4", "vllm": "#ff7f0e"}.get(engine, "#888888")


def kernel_palette(i: int) -> str:
    colors = [
        "#0f4c5c",
        "#2c7da0",
        "#84a98c",
        "#bc4749",
        "#f2cc8f",
        "#7a5195",
    ]
    return colors[i % len(colors)]


def escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def build_summary(manifest, manifest_dir: Path):
    summary = {
        "model": manifest["model"],
        "n_values": manifest["n_values"],
        "output_len": manifest["output_len"],
        "profile_n_values": manifest["profile_n_values"],
        "profile_output_len": manifest["profile_output_len"],
        "gpu_memory_utilization": manifest.get("gpu_memory_utilization"),
        "temperature": manifest.get("temperature"),
        "engines": {},
    }
    for engine, cfg in manifest["engines"].items():
        bench = {}
        for n, rel in cfg["benchmark"].items():
            row = parse_benchmark(manifest_dir / rel)
            if row:
                bench[int(n)] = row
        profiles = {}
        for n, rels in cfg["profiles"].items():
            profiles[int(n)] = {
                "kernels": parse_kernel_summary(manifest_dir / rels["kern_txt"])[:5],
                "mem_ops": parse_mem_summary(manifest_dir / rels["mem_txt"]),
            }
        summary["engines"][engine] = {"benchmark": bench, "profiles": profiles}
    return summary


def render_svg(summary, out_path: Path):
    width = 1440
    height = 1120
    margin = 70
    chart_top = 140
    chart_height = 320
    chart_width = 900
    ratio_top = chart_top
    ratio_left = 1020
    ratio_width = 320
    row_top = 560
    row_height = 96

    n_values = summary["n_values"]
    rv = [summary["engines"]["rvllm"]["benchmark"].get(n, {}).get("tok_per_sec", 0.0) for n in n_values]
    vv = [summary["engines"]["vllm"]["benchmark"].get(n, {}).get("tok_per_sec", 0.0) for n in n_values]
    ymax = max(rv + vv + [1.0])
    ymax *= 1.12

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Helvetica,Arial,sans-serif;fill:#111}",
        ".title{font-size:28px;font-weight:700}",
        ".subtitle{font-size:14px;fill:#555}",
        ".section{font-size:18px;font-weight:700}",
        ".small{font-size:12px;fill:#555}",
        ".axis{font-size:12px;fill:#666}",
        ".label{font-size:13px}",
        ".mono{font-family:Menlo,Consolas,monospace;font-size:12px}",
        "</style>",
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        f'<text x="{margin}" y="52" class="title">rvLLM vs vLLM 0.19 GPU Profile Comparison</text>',
        f'<text x="{margin}" y="78" class="subtitle">{escape(summary["model"])} on H100 · benchmark output_len={summary["output_len"]} · profile output_len={summary["profile_output_len"]} · temperature={summary.get("temperature", 0.0)}</text>',
    ]

    # Legend
    legend_y = 108
    for i, engine in enumerate(("rvllm", "vllm")):
        x = margin + i * 150
        parts.append(f'<rect x="{x}" y="{legend_y-12}" width="16" height="16" rx="3" fill="{color_for_engine(engine)}"/>')
        parts.append(f'<text x="{x+24}" y="{legend_y}" class="label">{engine}</text>')

    # Throughput chart
    parts.append(f'<text x="{margin}" y="{chart_top-24}" class="section">Throughput by Batch</text>')
    parts.append(f'<rect x="{margin}" y="{chart_top}" width="{chart_width}" height="{chart_height}" fill="white" stroke="#e5e5e5"/>')
    for tick in range(5):
        val = ymax * tick / 4
        y = chart_top + chart_height - (val / ymax) * chart_height
        parts.append(f'<line x1="{margin}" y1="{y:.1f}" x2="{margin+chart_width}" y2="{y:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{margin-8}" y="{y+4:.1f}" text-anchor="end" class="axis">{int(val):,}</text>')
    group_w = chart_width / max(len(n_values), 1)
    bar_w = min(32, group_w * 0.28)
    for i, n in enumerate(n_values):
        cx = margin + group_w * i + group_w / 2
        for j, (engine, values) in enumerate((("rvllm", rv), ("vllm", vv))):
            val = values[i]
            h = (val / ymax) * chart_height
            x = cx + (-bar_w - 6 if j == 0 else 6)
            y = chart_top + chart_height - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" fill="{color_for_engine(engine)}" rx="4"/>')
        parts.append(f'<text x="{cx:.1f}" y="{chart_top+chart_height+20}" text-anchor="middle" class="label">N={n}</text>')

    # Ratio chart
    parts.append(f'<text x="{ratio_left}" y="{ratio_top-24}" class="section">rvLLM / vLLM Ratio</text>')
    parts.append(f'<rect x="{ratio_left}" y="{ratio_top}" width="{ratio_width}" height="{chart_height}" fill="white" stroke="#e5e5e5"/>')
    parts.append(f'<line x1="{ratio_left}" y1="{ratio_top+chart_height/2}" x2="{ratio_left+ratio_width}" y2="{ratio_top+chart_height/2}" stroke="#ddd"/>')
    for i, n in enumerate(n_values):
        r = rv[i] / vv[i] if vv[i] else 0.0
        bar_h = abs(r - 1.0) / max(0.35, max(abs((x / y) - 1.0) if y else 0.0 for x, y in zip(rv, vv))) * (chart_height * 0.42)
        x = ratio_left + 26 + i * (ratio_width - 52) / max(len(n_values), 1)
        y = ratio_top + chart_height/2 - bar_h if r >= 1.0 else ratio_top + chart_height/2
        fill = "#2a9d8f" if r >= 1.0 else "#e76f51"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="28" height="{bar_h:.1f}" fill="{fill}" rx="4"/>')
        parts.append(f'<text x="{x+14:.1f}" y="{ratio_top+chart_height+20}" text-anchor="middle" class="label">{n}</text>')
        parts.append(f'<text x="{x+14:.1f}" y="{(y-8 if r>=1.0 else y+bar_h+16):.1f}" text-anchor="middle" class="small">{r:.2f}x</text>')

    # Top kernels / mem ops table
    parts.append(f'<text x="{margin}" y="{row_top-28}" class="section">Top Kernels and Transfer Tax</text>')
    parts.append(f'<text x="{margin}" y="{row_top-8}" class="subtitle">Kernel shares come from nsys cuda_gpu_kern_sum; transfer tax comes from cuda_gpu_mem_time_sum.</text>')
    parts.append(f'<rect x="{margin}" y="{row_top}" width="{width-2*margin}" height="{len(n_values)*row_height+50}" fill="white" stroke="#e5e5e5"/>')
    parts.append(f'<text x="{margin+12}" y="{row_top+26}" class="label">Batch</text>')
    parts.append(f'<text x="{margin+90}" y="{row_top+26}" class="label">rvLLM top kernels</text>')
    parts.append(f'<text x="{margin+670}" y="{row_top+26}" class="label">vLLM top kernels</text>')
    for idx, n in enumerate(n_values):
        y = row_top + 44 + idx * row_height
        parts.append(f'<line x1="{margin}" y1="{y+24}" x2="{width-margin}" y2="{y+24}" stroke="#f0f0f0"/>')
        parts.append(f'<text x="{margin+12}" y="{y+16}" class="label">N={n}</text>')
        for col, engine in enumerate(("rvllm", "vllm")):
            x0 = margin + (90 if col == 0 else 670)
            profile = summary["engines"][engine]["profiles"].get(n, {})
            kernels = profile.get("kernels", [])
            mem_ops = profile.get("mem_ops", {})
            h2d = mem_ops.get("[CUDA memcpy Host-to-Device]", {}).get("pct", 0.0)
            x = x0
            total_w = 430
            if kernels:
                for i, kernel in enumerate(kernels[:4]):
                    w = total_w * (kernel["pct"] / 100.0)
                    parts.append(f'<rect x="{x:.1f}" y="{y}" width="{max(w,1):.1f}" height="18" fill="{kernel_palette(i)}"/>')
                    x += w
                parts.append(f'<rect x="{x0}" y="{y}" width="{total_w}" height="18" fill="none" stroke="#ddd"/>')
                ty = y + 36
                for i, kernel in enumerate(kernels[:3]):
                    label = f'{kernel["name"][:44]} {kernel["pct"]:.1f}%'
                    parts.append(f'<text x="{x0}" y="{ty + i*14}" class="small">{escape(label)}</text>')
            else:
                parts.append(f'<text x="{x0}" y="{y+14}" class="small">no profile export</text>')
            parts.append(f'<text x="{x0+320}" y="{y+36}" class="small">HtoD {h2d:.1f}%</text>')

    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def render_html(summary_path: Path, svg_path: Path, out_path: Path):
    summary = load_json(summary_path)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>rvLLM vs vLLM profile compare</title>
  <style>
    body{{font-family:Helvetica,Arial,sans-serif;margin:24px;background:#f7f7f7;color:#111}}
    .wrap{{max-width:1480px;margin:0 auto}}
    .card{{background:white;border:1px solid #e5e5e5;border-radius:12px;padding:16px;margin-bottom:16px}}
    pre{{overflow:auto;background:#fafafa;border:1px solid #eee;padding:12px}}
    img{{max-width:100%;height:auto;display:block}}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>rvLLM vs vLLM 0.19 profile compare</h1>
      <p>Model: {escape(summary["model"])} · benchmark output_len={summary["output_len"]} · profile output_len={summary["profile_output_len"]}</p>
    </div>
    <div class="card">
      <img src="{svg_path.name}" alt="profile comparison">
    </div>
    <div class="card">
      <h2>Summary JSON</h2>
      <pre>{escape(json.dumps(summary, indent=2))}</pre>
    </div>
  </div>
</body>
</html>
"""
    out_path.write_text(html)


parser = argparse.ArgumentParser(description="Render rvLLM vs vLLM profile comparison")
parser.add_argument("--manifest", required=True)
parser.add_argument("--out-dir", required=True)
args = parser.parse_args()

manifest = load_json(Path(args.manifest))
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

summary = build_summary(manifest, Path(args.manifest).resolve().parent)
summary_path = out_dir / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n")

svg_path = out_dir / "profile_compare.svg"
render_svg(summary, svg_path)

html_path = out_dir / "profile_compare.html"
render_html(summary_path, svg_path, html_path)

print(svg_path)
print(html_path)
