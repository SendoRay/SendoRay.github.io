#!/usr/bin/env python3
"""Draw the original diagrams used by two blog posts:

  - content/posts/2026-08-14-cuda-basics.md        -> static/images/posts/cuda-basics/
  - content/posts/2026-08-14-delta-weight-sync.md  -> static/images/posts/delta-weight-sync/

All figures are drawn programmatically with matplotlib (no external images).
Each figure re-expresses, in graphical form, an ASCII diagram that used to
live in the article body; labels inside the figures are English / code terms
(Chinese explanations go into the Hugo figure captions instead).

Re-run to regenerate everything:

    python3 scripts/draw_blog_diagrams.py

Requires: matplotlib (headless, Agg backend).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

BLUE = "#4A90D9"
BLUE_L = "#DCEAF8"
ORANGE = "#E8A33D"
ORANGE_L = "#FBEDD5"
GRAY = "#6B7280"
GRAY_L = "#E8EAEE"
INK = "#1F2937"

DPI = 200

ROOT = Path(__file__).resolve().parent.parent
CUDA_DIR = ROOT / "static" / "images" / "posts" / "cuda-basics"
DELTA_DIR = ROOT / "static" / "images" / "posts" / "delta-weight-sync"

plt.rcParams.update({
    "font.family": "sans-serif",
    "text.color": INK,
})


def new_canvas(w, h):
    """Blank white canvas whose data coords match its size in inches."""
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def rbox(ax, x, y, w, h, text=None, title=None, body=None,
         fc=BLUE_L, ec=BLUE, lw=1.6, ls="-",
         fs=11, tfs=11, bfs=10, mono=False, mono_body=False):
    """Rounded box. Either a single centered `text`, or `title` (bold, at the
    top) plus `body` (centered in the remaining space)."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
        fc=fc, ec=ec, lw=lw, linestyle=ls))
    family = "monospace" if mono else "sans-serif"
    if text is not None:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=INK, family=family, linespacing=1.5)
    if title is not None:
        ax.text(x + w / 2, y + h - 0.3, title, ha="center", va="center",
                fontsize=tfs, color=INK, fontweight="bold")
    if body is not None:
        # center in the space left below the title (or the whole box if none)
        body_y = y + (h - 0.55) / 2 if title is not None else y + h / 2
        ax.text(x + w / 2, body_y, body, ha="center", va="center",
                fontsize=bfs, color=INK, linespacing=1.55,
                family="monospace" if mono_body else "sans-serif")


def arrow(ax, p0, p1, color=GRAY, lw=1.8, ls="-", rad=0.0, ms=17):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=ms, color=color,
        lw=lw, linestyle=ls, connectionstyle=f"arc3,rad={rad}",
        shrinkA=2, shrinkB=2))


def label(ax, x, y, text, fs=10, color=GRAY, ha="center", va="center",
          mono=False, weight="normal", rot=0):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color,
            family="monospace" if mono else "sans-serif",
            fontweight=weight, rotation=rot, linespacing=1.5)


def cell(ax, x, y, w, h, text, fc=BLUE_L, ec=BLUE, fs=9, lw=1.1):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=INK, family="monospace")


# ---------------------------------------------------------------------------
# CUDA post — figure 1: 1D / 2D / 3D thread blocks
# ---------------------------------------------------------------------------

def fig_thread_blocks():
    fig, ax = new_canvas(13.2, 4.8)

    # --- panel 1: 1D block ---
    label(ax, 2.15, 4.1, "1D block — dim3(6)", fs=12, color=INK, weight="bold")
    for i in range(6):
        cell(ax, 0.5 + i * 0.55, 2.9, 0.55, 0.55, str(i), fs=10)
    label(ax, 2.15, 2.4, "threadIdx.x = 0..5", fs=10, mono=True)

    # --- panel 2: 2D block ---
    label(ax, 6.7, 4.1, "2D block — dim3(4, 3)", fs=12, color=INK, weight="bold")
    gx, gtop, cw, ch = 5.0, 3.6, 0.85, 0.6
    for j in range(4):  # column headers
        label(ax, gx + (j + 0.5) * cw, gtop + 0.18, f"x={j}", fs=9)
    for i in range(3):  # rows, y=0 on top
        cy = gtop - (i + 1) * ch
        label(ax, gx - 0.32, cy + ch / 2, f"y={i}", fs=9)
        for j in range(4):
            cell(ax, gx + j * cw, cy, cw, ch, f"({j},{i})", fs=8.5)
    label(ax, 6.7, 1.25, "each thread gets (threadIdx.x, threadIdx.y)", fs=10, mono=True)

    # --- panel 3: 3D block (front layer z=0, offset back layer z=1) ---
    label(ax, 11.05, 4.1, "3D block — dim3(4, 3, 2)", fs=12, color=INK, weight="bold")
    fx, fy, cw, ch, off = 9.5, 1.7, 0.7, 0.5, 0.32
    for i in range(3):  # back layer first (z=1)
        for j in range(4):
            cell(ax, fx + off + j * cw, fy + off + i * ch, cw, ch, "",
                 fc=GRAY_L, ec=GRAY, lw=0.9)
    for i in range(3):  # front layer (z=0)
        for j in range(4):
            cell(ax, fx + j * cw, fy + i * ch, cw, ch, f"({j},{2 - i})", fs=8)
    label(ax, fx + 4 * cw + off + 0.15, fy + 3 * ch + off - 0.1, "z=1",
          fs=9.5, ha="left")
    label(ax, fx - 0.15, fy + 0.2, "z=0", fs=9.5, ha="right")
    label(ax, 11.05, 1.0, "threadIdx.z adds a 3rd axis", fs=10, mono=True)

    save(fig, CUDA_DIR / "thread-blocks.png")


# ---------------------------------------------------------------------------
# CUDA post — figure 2: block -> SM mapping
# ---------------------------------------------------------------------------

def fig_block_sm_mapping():
    fig, ax = new_canvas(10.5, 7.4)

    label(ax, 2.2, 7.0, "Software: Grid", fs=13, color=INK, weight="bold")
    label(ax, 8.1, 7.0, "Hardware: SMs", fs=13, color=INK, weight="bold")

    sm_ids = [0, 2, 4, 6]
    for i in range(4):
        y = 5.5 - 1.35 * i
        # block box with two threads inside
        rbox(ax, 0.7, y, 3.0, 1.05)
        label(ax, 1.35, y + 0.52, f"Block {i}", fs=11, color=INK, weight="bold")
        for t in range(2):
            cell(ax, 2.15 + t * 0.75, y + 0.3, 0.62, 0.45, f"T{t}",
                 fc="white", ec=BLUE, fs=9)
        # SM box
        rbox(ax, 6.8, y, 2.6, 1.05, text=f"SM {sm_ids[i]}", fc=GRAY_L, ec=GRAY, fs=12)
        arrow(ax, (3.78, y + 0.52), (6.72, y + 0.52), color=ORANGE, lw=2.2)

    label(ax, 5.25, 1.15,
          "all threads of one Block → always the same SM",
          fs=11, color=INK, weight="bold")
    label(ax, 5.25, 0.72,
          "different Blocks → may be scheduled onto different SMs",
          fs=11, color=GRAY)

    save(fig, CUDA_DIR / "block-sm-mapping.png")


# ---------------------------------------------------------------------------
# CUDA post — figure 3: SM internal architecture (A100, simplified)
# ---------------------------------------------------------------------------

def fig_sm_architecture():
    fig, ax = new_canvas(11.5, 8.2)

    # outer SM box
    rbox(ax, 0.6, 1.3, 10.3, 6.3, fc="#FBFCFD", ec=GRAY, lw=2.0)
    label(ax, 5.75, 7.25, "SM  (one of 108 on A100)", fs=13, color=INK, weight="bold")

    xs = [0.95, 3.45, 5.95, 8.45]
    for i, x in enumerate(xs):  # warp schedulers
        rbox(ax, x, 5.75, 2.25, 1.0,
             text=f"Warp Scheduler {i}\n+ Dispatch Unit", fs=10)
        arrow(ax, (x + 1.125, 5.72), (x + 1.125, 5.28))

    rbox(ax, 0.95, 4.4, 9.75, 0.85, text="Register File",
         fc=ORANGE_L, ec=ORANGE, fs=12)
    for x in xs:
        arrow(ax, (x + 1.125, 4.37), (x + 1.125, 3.93))

    units = ["FP32 Units", "INT32 Units", "Tensor Cores", "LD/ST Units"]
    for x, name in zip(xs, units):
        rbox(ax, x, 2.9, 2.25, 1.0, text=name, fc=GRAY_L, ec=GRAY, fs=11)
    arrow(ax, (5.75, 2.87), (5.75, 2.43))

    rbox(ax, 0.95, 1.55, 9.75, 0.85,
         text="L1 Data Cache / Shared Memory  (same on-chip SRAM)",
         fc=ORANGE_L, ec=ORANGE, fs=11)

    label(ax, 5.75, 0.95,
          "each cycle: every scheduler picks one READY warp and issues its next instruction",
          fs=10.5)
    label(ax, 5.75, 0.5,
          "stalled warps (memory wait / __syncthreads) are switched out — this hides latency",
          fs=10.5)

    save(fig, CUDA_DIR / "sm-architecture.png")


# ---------------------------------------------------------------------------
# CUDA post — figure 4: three logical memory scopes
# ---------------------------------------------------------------------------

def fig_memory_hierarchy():
    fig, ax = new_canvas(12.5, 6.6)

    rows = [
        (4.7, GRAY_L, GRAY, "Global Memory — visible to ALL threads (Grid scope)",
         "cudaMalloc'd · capacity = full VRAM · ~100s of cycles"),
        (2.85, ORANGE_L, ORANGE, "Shared Memory — shared within one Block",
         "__shared__ · on-chip SRAM, same silicon as L1 · ~10s of cycles"),
        (1.0, BLUE_L, BLUE, "Registers / Local Memory — private to one Thread",
         "registers: ~1 cycle, up to 255 per thread · Local actually lives in device memory (slow!)"),
    ]
    for y, fc, ec, head, sub in rows:
        rbox(ax, 2.3, y, 8.6, 1.35, fc=fc, ec=ec)
        label(ax, 6.6, y + 0.93, head, fs=12, color=INK, weight="bold")
        label(ax, 6.6, y + 0.44, sub, fs=10)

    # connectors between the three scopes
    arrow(ax, (6.6, 2.85), (6.6, 2.4), lw=1.4)
    arrow(ax, (6.6, 4.7), (6.6, 4.25), lw=1.4)

    # side annotations
    arrow(ax, (1.35, 1.2), (1.35, 5.85), color=GRAY, lw=1.8)
    label(ax, 1.0, 3.55, "larger scope · slower access", fs=10.5, rot=90)
    arrow(ax, (11.7, 5.85), (11.7, 1.2), color=BLUE, lw=1.8)
    label(ax, 12.05, 3.55, "smaller scope · faster access", fs=10.5,
          color=BLUE, rot=270)

    save(fig, CUDA_DIR / "memory-hierarchy.png")


# ---------------------------------------------------------------------------
# CUDA post — figure 5: dot_product memory request flow
# ---------------------------------------------------------------------------

def fig_global_memory_dataflow():
    fig, ax = new_canvas(10.0, 9.0)

    boxes = [
        (7.55, BLUE_L, BLUE, "dot_product kernel",
         "single thread, 16-iteration loop: reads a[i], b[i]"),
        (5.4, ORANGE_L, ORANGE, "L1 Cache  (per-SM)",
         "28 hits out of 32 requests"),
        (3.25, ORANGE_L, ORANGE, "L2 Cache  (device-wide)",
         "hit rate 33.33%  (2 hits / 6 accesses)"),
        (1.1, GRAY_L, GRAY, "Device Memory",
         "arrays a, b: 2 × 64 B = 4 sectors = 128 B"),
    ]
    for y, fc, ec, head, sub in boxes:
        rbox(ax, 2.3, y, 5.4, 1.15, fc=fc, ec=ec)
        label(ax, 5.0, y + 0.79, head, fs=12, color=INK, weight="bold")
        label(ax, 5.0, y + 0.36, sub, fs=10)

    flows = [
        (7.52, 6.6, "32 load requests  (16 iters × 2 arrays)"),
        (5.37, 4.45, "4 sector requests  (only 4 L1 misses)"),
        (3.22, 2.3, "4 sectors = 128 B  (fetched as 64 B pairs)"),
    ]
    for y0, y1, text in flows:
        arrow(ax, (5.0, y0), (5.0, y1), lw=2.0)
        label(ax, 5.3, (y0 + y1) / 2, text, fs=10.5, ha="left", color=INK)

    save(fig, CUDA_DIR / "global-memory-dataflow.png")


# ---------------------------------------------------------------------------
# CUDA post — figure 6: float[16] split into 32 B sectors
# ---------------------------------------------------------------------------

def fig_sector_layout():
    fig, ax = new_canvas(13.2, 5.6)

    label(ax, 6.6, 5.2,
          "1 float = 4 B      ·      1 sector = 32 B = 8 floats      ·      "
          "float[16] = 64 B = 2 sectors",
          fs=12, color=INK, weight="bold")

    def array_row(name, y, show_headers):
        label(ax, 1.45, y + 0.3, name, fs=11, mono=True, ha="right", color=INK)
        for i in range(16):
            fc, ec = (BLUE_L, BLUE) if i < 8 else (ORANGE_L, ORANGE)
            cell(ax, 1.7 + i * 0.68, y, 0.68, 0.6, f"{name[0]}[{i}]",
                 fc=fc, ec=ec, fs=8)
        if show_headers:
            label(ax, 1.7 + 4 * 0.68, y + 0.95, "sector 0  (32 B)",
                  fs=10.5, color=BLUE, weight="bold")
            label(ax, 1.7 + 12 * 0.68, y + 0.95, "sector 1  (32 B)",
                  fs=10.5, color=ORANGE, weight="bold")

    array_row("a", 3.5, show_headers=True)
    array_row("b", 2.4, show_headers=False)

    label(ax, 6.6, 1.5,
          "reading a[0] fetches the whole sector 0 (32 B)  →  "
          "later reads of a[1..7] hit in cache",
          fs=11, color=INK)
    label(ax, 6.6, 0.95, "arrays a + b  →  2 × 2 = 4 sectors in total (128 B)",
          fs=11)

    save(fig, CUDA_DIR / "sector-layout.png")


# ---------------------------------------------------------------------------
# CUDA post — figure 7: Python -> binding.cpp -> .cu call chain
# ---------------------------------------------------------------------------

def fig_pybind11_call_chain():
    fig, ax = new_canvas(12.0, 9.4)

    # Python layer
    rbox(ax, 0.7, 7.35, 7.6, 1.5, title="Python", tfs=12,
         body='import example_kernels\nexample_kernels.roll_call()',
         bfs=10, mono_body=True)
    arrow(ax, (4.5, 7.32), (4.5, 6.83), lw=2.0)
    label(ax, 4.8, 7.08, "pybind11-generated binding", fs=10, ha="left")

    # C++ binding layer
    rbox(ax, 0.7, 4.95, 7.6, 1.85, fc=ORANGE_L, ec=ORANGE,
         title="roll_call_binding.cpp — compiled by g++", tfs=11,
         body='PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)\n'
              '  m.def("roll_call", &roll_call_binding)\n'
              'roll_call_binding() → roll_call_launcher()  // forward decl only',
         bfs=9.5, mono_body=True)
    arrow(ax, (4.5, 4.92), (4.5, 4.43), lw=2.0)
    label(ax, 4.8, 4.68, "symbol resolved at link time", fs=10, ha="left")

    # CUDA layer
    rbox(ax, 0.7, 2.4, 7.6, 2.0, fc=BLUE_L, ec=BLUE,
         title="roll_call.cu — compiled by nvcc", tfs=11,
         body='roll_call_launcher()\n'
              '  └─ roll_call_kernel<<<1, 5>>>()   // runs on GPU\n'
              '     cudaDeviceSynchronize()',
         bfs=9.5, mono_body=True)

    # setup.py on the side, linking the two compiled files
    rbox(ax, 9.2, 3.6, 2.3, 2.6, fc=GRAY_L, ec=GRAY,
         title="setup.py", tfs=11,
         body="CUDAExtension(\n 'example_kernels',\n [binding.cpp,\n  roll_call.cu])\n+ BuildExtension",
         bfs=8.5, mono_body=True)
    arrow(ax, (9.15, 5.35), (8.35, 5.85), ls="--", lw=1.5)
    arrow(ax, (9.15, 4.45), (8.35, 3.5), ls="--", lw=1.5)
    label(ax, 10.35, 3.15, "compiles & links both files", fs=9)
    label(ax, 10.35, 2.83, "into ONE extension module", fs=9)

    save(fig, CUDA_DIR / "pybind11-call-chain.png")


# ---------------------------------------------------------------------------
# Delta post — figure 8: sender pipeline
# ---------------------------------------------------------------------------

def fig_sender_pipeline():
    fig, ax = new_canvas(13.0, 9.6)

    label(ax, 6.5, 9.3, "Sender pipeline — runs on the PP-source rank only",
          fs=13, color=INK, weight="bold")

    # sources: GPU weights + CPU pinned snapshot
    rbox(ax, 0.6, 7.3, 3.6, 1.1, text="GPU: current weights W_t", fs=11)
    rbox(ax, 0.6, 5.0, 3.6, 1.5, fc=GRAY_L, ec=GRAY,
         title="CPU pinned snapshot", tfs=11,
         body="full-weight copy;\nh2d_stream prefetches next chunk", bfs=9.5)

    # diff
    rbox(ax, 5.4, 5.7, 3.6, 2.0, fc=ORANGE_L, ec=ORANGE,
         title="Bytewise diff", tfs=11.5,
         body="current.view(int_dtype)\n!= snapshot.view(int_dtype)\n"
              "≈ 1–3% elements changed", bfs=9.5, mono_body=True)
    arrow(ax, (4.25, 7.85), (5.38, 7.15), lw=2.0)
    arrow(ax, (4.25, 5.75), (5.38, 6.3), lw=2.0)

    # snapshot write-back loop (side stream)
    arrow(ax, (5.6, 5.66), (4.28, 5.25), color=ORANGE, lw=1.8, ls="--", rad=0.25)
    label(ax, 2.6, 4.55,
          "d2h_stream (side-stream): write back changed values\n"
          "→ snapshot becomes next step's diff baseline", fs=9.5)

    # encode
    arrow(ax, (7.2, 5.67), (7.2, 5.23), lw=2.0)
    rbox(ax, 5.0, 3.35, 4.4, 1.85, fc=BLUE_L, ec=BLUE,
         title="Encode → __positions__ + __values__", tfs=11,
         body="indices      int32   4 B/nnz\n"
              "deltas       uint16  ~2 B/nnz (gap-delta)\n"
              "deltas_zstd  + zstd level 1",
         bfs=9, mono_body=True)

    # bucket
    arrow(ax, (7.2, 3.32), (7.2, 2.83), lw=2.0)
    label(ax, 7.5, 3.08, "accumulate by buffer size", fs=9.5, ha="left")
    rbox(ax, 5.9, 2.0, 2.6, 0.8, text="Bucket & flush",
         fc=ORANGE_L, ec=ORANGE, fs=11)

    # two transports
    arrow(ax, (6.4, 1.98), (3.4, 1.42), lw=2.0)
    arrow(ax, (8.0, 1.98), (9.9, 1.42), lw=2.0)
    rbox(ax, 0.9, 0.25, 4.9, 1.1, fc=BLUE_L, ec=BLUE,
         title="NCCL broadcast", tfs=10.5,
         body="positions H2D → broadcast together with values", bfs=9.5)
    rbox(ax, 7.4, 0.25, 5.2, 1.1, fc=GRAY_L, ec=GRAY,
         title="Disk: safetensors (background thread)", tfs=10.5,
         body="values D2H → encode + zstd + fsync + atomic rename", bfs=9.5)

    save(fig, DELTA_DIR / "sender-pipeline.png")


# ---------------------------------------------------------------------------
# Delta post — figure 9: receiver apply flow
# ---------------------------------------------------------------------------

def fig_receiver_apply():
    fig, ax = new_canvas(12.0, 12.0)

    label(ax, 6.0, 11.65,
          "Receiver apply flow — inside SGLang (PR #26519)",
          fs=13, color=INK, weight="bold")

    # two transport entries
    rbox(ax, 0.7, 9.9, 4.9, 1.5, fc=BLUE_L, ec=BLUE,
         title="NCCL entry", tfs=11,
         body="RPC → DeltaSpec metadata\nprealloc recv buffers → broadcast", bfs=9.5)
    rbox(ax, 6.4, 9.9, 4.9, 1.5, fc=ORANGE_L, ec=ORANGE,
         title="Disk entry", tfs=11,
         body="thread-pool file reads → zstd decompress\n→ parse safetensors header",
         bfs=9.5)
    arrow(ax, (3.15, 9.87), (5.3, 9.3), lw=2.0)
    arrow(ax, (8.85, 9.87), (6.7, 9.3), lw=2.0)

    # merged sparse payload
    rbox(ax, 3.6, 8.15, 4.8, 1.1, fc=GRAY_L, ec=GRAY,
         title="Sparse payload on GPU", tfs=11,
         body="__positions__ + __values__", bfs=10, mono_body=True)

    # checksum
    arrow(ax, (6.0, 8.12), (6.0, 7.65), lw=2.0)
    rbox(ax, 3.6, 6.55, 4.8, 1.05, fc=ORANGE_L, ec=ORANGE,
         title="Checksum verify", tfs=11,
         body="torch.hash_tensor + XOR-reduce", bfs=9.5, mono_body=True)
    arrow(ax, (8.45, 7.07), (10.4, 7.07), ls="--", lw=1.6)
    label(ax, 9.4, 7.33, "mismatch → raise", fs=10)

    # decode / densify
    arrow(ax, (6.0, 6.52), (6.0, 6.15), lw=2.0)
    rbox(ax, 3.3, 4.55, 5.4, 1.55, fc=BLUE_L, ec=BLUE,
         title="Per-param decode / densify", tfs=11,
         body="NaN-filled full-shape buffer (NaN = 'unchanged')\n"
              "unpack positions → cumsum → index_copy_\n"
              "sparse on wire, dense on apply", bfs=9.5)

    # _delta_apply_context (dashed) with two inner steps
    arrow(ax, (6.0, 4.52), (6.0, 3.7), lw=2.0)
    rbox(ax, 2.7, 1.3, 6.6, 2.85, fc="none", ec=GRAY, ls="--", lw=1.6)
    label(ax, 3.0, 3.9, "_delta_apply_context", fs=10.5, mono=True,
          ha="left", color=GRAY, weight="bold")
    rbox(ax, 3.3, 2.75, 5.4, 0.9, fc=BLUE_L, ec=BLUE,
         body="model.load_weights(chunk) — native loader\n"
              "512 MB chunks, engine logic unchanged", bfs=9.5)
    arrow(ax, (6.0, 2.72), (6.0, 2.53), lw=1.8)
    rbox(ax, 3.3, 1.5, 5.4, 1.0, fc=ORANGE_L, ec=ORANGE,
         body="monkeypatched: copy_ → dst[~isnan(src)] = src[…]\n"
              "fill_(NaN) → no-op", bfs=9, mono_body=True)

    # final state
    arrow(ax, (6.0, 1.27), (6.0, 0.98), lw=2.0)
    rbox(ax, 3.3, 0.1, 5.4, 0.85, fc=GRAY_L, ec=GRAY)
    label(ax, 6.0, 0.52, "GPU weights updated in place — bit-identical, no drift",
          fs=11, color=INK, weight="bold")

    save(fig, DELTA_DIR / "receiver-apply.png")


# ---------------------------------------------------------------------------

def main():
    print("Drawing CUDA post figures ->", CUDA_DIR.relative_to(ROOT))
    fig_thread_blocks()
    fig_block_sm_mapping()
    fig_sm_architecture()
    fig_memory_hierarchy()
    fig_global_memory_dataflow()
    fig_sector_layout()
    fig_pybind11_call_chain()
    print("Drawing delta post figures ->", DELTA_DIR.relative_to(ROOT))
    fig_sender_pipeline()
    fig_receiver_apply()
    print("Done.")


if __name__ == "__main__":
    main()
