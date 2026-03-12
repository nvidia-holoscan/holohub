# G-SHARP Technical Guide

> A deep-dive companion to `README.md` for developers tuning, debugging, and
> extending the G-SHARP scene reconstruction pipeline.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Phase-by-Phase Walkthrough](#phase-by-phase-walkthrough)
3. [Progress Monitor Workflow](#progress-monitor-workflow)
4. [Tuning Parameters for Better Reconstruction](#tuning-parameters-for-better-reconstruction)
5. [Troubleshooting Poor Performance](#troubleshooting-poor-performance)
6. [Advanced: Incremental and Partial Runs](#advanced-incremental-and-partial-runs)
7. [Codebase Layout](#codebase-layout)

---

## Architecture Overview

G-SHARP (Gaussian Splatting for Holographic Anatomical Reconstruction Pipeline)
transforms a sequence of RGB frames into an interactive 3D Gaussian Splatting
model. The pipeline runs streaming inference (DA2 + MedSAM3 in parallel with
HoloViz live preview), then batch pose estimation, then format conversion and
GSplat training, and finally a HoloViz render viewer.

**Pipeline flowchart (typical throughput, ~90-frame clip, single GPU):**

```text
  RGB frames
       │
       ├────────────────────┬────────────────────┐
       ▼                     ▼                    │
  ┌──────────┐          ┌──────────┐         ┌────────────┐
  │   DA2    │          │ MedSAM3  │         │  HoloViz   │
  │  depth   │          │  masks   │         │  live      │  ← 3-panel: source / depth / mask
  │ ~15 fps  │          │ ~15 fps  │         │  preview   │
  └────┬─────┘          └────┬─────┘         └────────────┘
       └──────────┬──────────┘
                  ▼  depth + masks
          ┌───────────────┐     ┌─────────────────────────────────┐     ┌────────────┐
          │  VGGT Pose    │────▶│  EndoNeRF conversion + GSplat   │────▶│  HoloViz   │
          │  (batch)      │     │  training                       │     │  viewer    │
          │  ~2–5 fps     │     │  poses_bounds + coarse→fine     │     │  30 fps    │
          └───────────────┘     │  → checkpoint                   │     └────────────┘
                poses + K       └─────────────────────────────────┘      render loop
```

Viewer FPS is configurable (default 30). Training time depends on `--training-iterations`.

### Why the Hybrid Architecture?

| Phase | Execution Model | Rationale |
| ----- | --------------- | --------- |
| **1** | Holoscan streaming | DA2 and MedSAM3 are frame-by-frame GPU inference — natural streaming operators with parallel scheduling and zero-copy GPU sharing |
| **2** | Standalone batch | VGGT is a vision transformer that needs all frames at once for globally consistent poses; doesn't fit the streaming model |
| **3** | Standalone CPU | Pure NumPy, <5 seconds, no GPU; Holoscan overhead would add complexity without benefit |
| **4** | Standalone iterative | 2000+ backward passes over the same data; iterative PyTorch optimization, not streaming |
| **5** | Holoscan streaming | Real-time rendering at 15–30 fps is a natural streaming output |

---

## Phase-by-Phase Walkthrough

### Phase 1: DA2 + MedSAM3 Streaming Inference

**Entry point:** `gsplat_scene_recon.py` (Holoscan Application)

The Holoscan app runs DA2 depth estimation and MedSAM3 tool segmentation in
parallel on each frame. A `DataPrepOp` synchronizes and writes results to disk.
When not running headless, an `OverlayComposerOp` renders a 3-panel live preview
(source / depth colormap / mask overlay) via HoloViz.

**Key files:**

- `operators/depth_anything_v2_op.py` — DA2 Holoscan operator
- `operators/medsam3_segmentation_op.py` — MedSAM3 Holoscan operator
- `operators/data_prep_op.py` — Disk writer, synchronizes by frame index
- `models/depth_anything_v2/` — Custom DA2 model architecture
- `models/medsam3/sam3_inference.py` — SAM3 inference wrapper

**Output:** `<output>/phase1_raw/images/`, `depth_raw/`, `masks/`

**Resolution handling:** The image source (`ImageDirectorySourceOp`) reads frames
as-is from disk. If the input directory contains frames from different sources
(e.g. `frame_00000.png` at 240×320 and `frame-000000.color.png` at 512×640),
Phase 1 would historically write mixed resolutions and cause training to fail
with shape/broadcast errors. `DataPrepOp` now normalizes every frame to the
**first frame’s (H, W)** before writing; any later frame with different
dimensions is resized so Phase 2, Phase 3, and training always see a single
resolution. Phase 3 (`format_conversion.py`) also resizes as a safety net.

### Phase 2: VGGT Batch Pose Estimation

**Entry point:** `models/vggt/vggt_inference.py` (standalone script)

VGGT-1B processes frames in configurable batches (default 30) and produces
globally consistent camera poses. Poses from different batches are stitched
using overlapping frames and normalized so frame 0 = identity.

**Key parameters:**

- `--batch-size` — Frames per VGGT batch. Reduce to 15–20 on 24GB GPUs.

**Output:** `<output>/phase2_vggt/poses.npy`, `intrinsics.npy`

### Phase 3: EndoNeRF Format Conversion

**Entry point:** `stages/format_conversion.py` (standalone script)

Converts DA2 raw depth and VGGT poses into the EndoNeRF dataset format expected
by the GSplat training code. This is a critical stage where depth quantization
and scale alignment happen.

**Key transformation:**

```text
DA2 raw depth (float32, meters) × DEPTH_SCALE → uint8 (0–255)
VGGT translations (meters) × DEPTH_SCALE → centimeters
```

**Key parameter:**

- `--depth-scale` — Default 100 (centimeters). This value directly
  controls training stability through `scene_scale` (see Tuning section).

**Output:** `<output>/phase3_endonerf/` with `poses_bounds.npy`, `images/`,
`depth/`, `masks/`

### Phase 4: GSplat Training

**Entry point:** `training/train_standalone.py` → `training/gsplat_train.py`

Two-stage training:

1. **Coarse stage** — Static Gaussians, no deformation. Establishes geometry.
2. **Fine stage** — Deformation network enabled. Learns temporal dynamics.

Densification periodically splits/clones Gaussians to improve detail. Training
logs PSNR, loss components, and Gaussian count. The best checkpoint (by eval
PSNR) is saved automatically.

**Output:** `<output>/phase4_training/trained_model/ckpts/fine_best_psnr.pt`

### Phase 5: Live Render Viewer

**Entry point:** `utils/render_viewer.py` (Holoscan Application)

Loads the trained checkpoint and renders each camera pose from the training set.
Opens a HoloViz window showing GT vs Rendered side-by-side at configurable fps.
Close the window to exit.

---

## Progress Monitor Workflow

During Phases 2–4, a background Holoscan app (`utils/progress_monitor.py`) displays
real-time progress bars in a HoloViz window. The system works through a shared
JSON file:

```text
Pipeline phases          Progress JSON file         HoloViz display
─────────────           ──────────────────         ────────────────
                         progress.json
Phase 2 (VGGT)    ──→   write(vggt, 30/91)  ──→   ┌──────────────┐
                                                    │ VGGT   33%   │
Phase 3 (Convert)  ──→   write(format, 91/91) ──→  │ Format 100%  │
                                                    │ Train   0%   │
Phase 4 (Training) ──→   write(train, 700/1400)──→  │ Train  50%   │
                                                    └──────────────┘
```

### How it works

1. `run_gsharp.py` launches `utils/progress_monitor.py` as a background process before
   Phase 2 starts.

2. Each phase writes progress updates to `progress.json` using
   `stages/progress.py`:

   ```python
   update_progress(progress_file, "vggt", "VGGT Pose Estimation",
                   current=30, total=91, detail="Batch 1/3", status="running")
   ```

3. `utils/progress_monitor.py` polls the JSON file every 400ms and renders:
   - Colored progress bars (blue = running, green = complete, red = error)
   - Percentage and detail text per stage
   - Title header

4. The training wrapper (`stages/train_with_progress.py`) parses tqdm output
   from `train_standalone.py` to extract step counts and best PSNR, then updates
   the progress file in real-time.

5. The orchestrator kills the progress monitor after Phase 4 completes.

### Disabling the progress monitor

The progress monitor requires a display. In headless mode (`--headless`), the
orchestrator skips launching it. Progress is still printed to stdout.

---

## Tuning Parameters for Better Reconstruction

### Orchestrator-Level Parameters (CLI)

| Parameter | Default | What it does | When to change |
| --------- | ------- | ------------ | --------------- |
| `--training-iterations` | 1400 | Total training steps (coarse + fine) | Increase for more detail (3000–7000 for production) |
| `--coarse-iterations` | 200 | Steps with static Gaussians | Increase if geometry is poor before deformation kicks in |
| `--no-deformation` | off | Disable temporal deformation | Use for static scenes (synthetic data, rigid objects) |
| `--batch-size` | 30 | VGGT frames per batch | Reduce on low-VRAM GPUs; increase for larger scenes |
| `--depth-scale` | 100.0 | Depth quantization factor | See "The Critical `depth-scale`" below |

### The Critical `depth-scale`

`depth-scale` controls how metric depth (meters) maps to uint8 pixel values.
The EndoNeRF training code computes `scene_scale` from the maximum point cloud
extent, and the deformation learning rate is `base_lr × scene_scale`. If the
scale is wrong, training diverges.

| depth-scale | Units | Typical depth range | scene_scale | Outcome |
| ----------- | ----- | ------------------- | ----------- | ------- |
| 1000 | mm | 305–1306 (uint16) | ~1420 | **NaN losses, divergence** |
| **100** | **cm** | **31–131 (uint8)** | **~142** | **Stable (default)** |
| 10 | dm | 3–13 (uint8) | ~14 | Extremely slow convergence |

**Rule of thumb:** The resulting `scene_scale` should be in the range **50–300**.
If your scene has very different camera-to-subject distances, adjust
`depth-scale` accordingly.

### Training-Level Parameters (in `training/gsplat_train.py` → `EndoConfig`)

These require editing the source code or passing them through the training
script:

| Parameter | Default | Effect | Tuning advice |
| --------- | ------- | ------ | ------------- |
| `means_lr` | 1.6e-4 | Gaussian position learning rate | Increase for faster geometry convergence; decrease if positions oscillate |
| `deformation_lr` | 1.0e-5 | Deformation network LR | Increase if deformation looks static; decrease if NaN appears late in training |
| `grid_lr` | 1.0e-5 | Hexplane grid LR | Typically matches `deformation_lr` |
| `ssim_lambda` | 0.2 | SSIM loss weight | Increase for sharper textures, decrease if training becomes unstable |
| `depth_lambda` | 0.001 | Depth supervision weight | Increase to enforce depth consistency; too high overrides RGB signal |
| `tv_lambda` | 0.03 | Total variation smoothness | Increase if artifacts/noise in deformation; decrease if over-smoothed |
| `use_masks` | true | Mask out tools from loss | Set `false` (via `--no_masks`) to reconstruct the full scene including tools |
| `densify_grad_threshold_fine_init` | 0.0002 | Gradient threshold for splitting | Lower → more Gaussians (more detail, more VRAM); higher → fewer Gaussians |

### Recommended Configurations

**Quick validation (2 min):**

```bash
--training-iterations 1400 --coarse-iterations 200
```

**Production quality (5–8 min):**

```bash
--training-iterations 5000 --coarse-iterations 500
```

**Maximum quality (15–20 min):**

```bash
--training-iterations 10000 --coarse-iterations 1000
```

---

## Troubleshooting Poor Performance

### Symptom: NaN Losses During Training

**Check these in order:**

1. **Depth scale is too large.** If you see `scene_scale > 500` in the training
   logs, the effective deformation LR is too high. Reduce `--depth-scale`.

2. **Deformation network on static data.** If the scene is truly static, the
   deformation network can fit noise and diverge. Add `--no-deformation`.

3. **Too few frames.** The deformation network needs enough temporal variation
   to learn meaningful dynamics. Below ~30 frames, it may overfit.

### Symptom: Mixed Resolutions / Broadcast Error During Training

If training fails with `ValueError: operands could not be broadcast together with
shapes (H1,W1) (H2,W2)`, the ingestion contained images (or depth/masks) at two
different resolutions. **Root cause:** the input frame directory had mixed
sources (e.g. synthetic `frame_*.png` and original `frame-*.color.png`), and
Phase 1 used to write each frame at its native size. **Fix:** Phase 1
`DataPrepOp` now resizes every frame to the first frame’s resolution before
writing; Phase 3 also resizes when building the EndoNeRF dataset. For best
results, use a single resolution and avoid mixing frame sources.

### Symptom: Low PSNR (< 25 dB)

**Check these in order:**

1. **Verify Phase 1 depth maps.** Inspect `<output>/phase1_raw/depth/` visually.
   If depth maps are blank, saturated, or noisy, the DA2 checkpoint may be
   wrong or the input images may be too dark/overexposed.

2. **Verify Phase 2 poses.** VGGT requires sufficient visual overlap between
   frames. If the camera moves too fast (motion blur) or the scene has
   repetitive textures, poses may be poor. Check `<output>/phase2_vggt/` for
   reasonable pose trajectories.

3. **Verify Phase 3 depth quantization.** Open the uint8 depth PNGs in
   `<output>/phase3_endonerf/depth/`. If most pixels are clipped to 1 or 255,
   the depth scale is wrong. Values should span a reasonable range (e.g.,
   30–130 for surgical endoscopy at `depth-scale=100`).

4. **Increase training iterations.** 1400 is a quick validation budget. For
   production, use 3000–7000 iterations.

5. **Check mask quality.** If `use_masks=True` but MedSAM3 is masking out
   important scene regions, try `--no_masks` to reconstruct the full scene.

### Symptom: Blurry or Over-Smoothed Reconstruction

1. **Increase `ssim_lambda`** (default 0.2) to 0.4–0.6 for sharper textures.

2. **Lower `tv_lambda`** (default 0.03) to allow more deformation detail.

3. **Lower `densify_grad_threshold`** to create more Gaussians in detail-rich
   areas (at the cost of more VRAM).

4. **Increase training iterations** — fine detail emerges in later steps.

### Symptom: Training Stalls or Slow Convergence

1. **`depth-scale` too small.** If `scene_scale < 20`, learning rates are
   effectively very low. Increase `--depth-scale`.

2. **Too many Gaussians too early.** Check the Gaussian count in training logs.
   If it grows past 200K early, densification may be too aggressive.
   Increase `densify_grad_threshold`.

### Symptom: Phase 1 Crash

1. **`ModuleNotFoundError: No module named 'depth_anything_v2'`** — The
   `.dockerignore` may be excluding `models/`. Verify the Docker build
   context includes `models/depth_anything_v2/`.

2. **`AttributeError: 'NoneType' object has no attribute 'shape'`** — Input
   frames are symlinks pointing outside the container. Use hard links or
   copy the files instead.

3. **SAM3 checkpoint missing or 0 bytes** — Place the real checkpoint in the same data tree as your frames (e.g. `data/gsplat_scene_recon/medsam3/checkpoint.pt`) or bind-mount and pass `--sam3-checkpoint`.

### Symptom: Phase 2 (VGGT) Out-of-Memory

Reduce `--batch-size` from 30 to 15 or 10. VGGT's attention mechanism scales
quadratically with batch size.

### Symptom: Phase 5 Viewer Doesn't Open

1. Ensure X11 forwarding is configured: `-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix`
2. Run `xhost +local:docker` on the host before launching the container.
3. If running via SSH, ensure `$DISPLAY` is set and X forwarding is enabled.

---

## Advanced: Incremental and Partial Runs

The `--skip-*` flags let you re-run specific phases without repeating earlier
(expensive) phases:

```bash
# Re-train with more iterations, reusing Phase 1–3 output
--skip-phase1 --skip-phase2 --skip-phase3 --training-iterations 7000

# Re-run only the viewer with an existing checkpoint
--skip-phase1 --skip-phase2 --skip-phase3 --skip-training

# Re-run Phase 3 + 4 + 5 (e.g., to test a different depth-scale)
--skip-phase1 --skip-phase2 --depth-scale 150
```

The output directory structure is stable across runs — each phase writes to
its own subdirectory (`phase1_raw/`, `phase2_vggt/`, etc.), so skipped phases
simply reuse the existing output.

---

## Codebase Layout

```text
gsplat_scene_recon/
├── run_gsharp.py                  — Top-level orchestrator (entry point)
├── gsplat_scene_recon.py          — Phase 1: Holoscan streaming app
│
├── utils/                         — Helpers and Phase 5 viewer
│   ├── progress_monitor.py        — HoloViz progress bar display
│   ├── render_viewer.py           — Phase 5: Holoscan render viewer
│   └── phase1_config.yaml         — Phase 1 operator config (optional)
│
├── operators/                     — Holoscan operators
│   ├── depth_anything_v2_op.py    — DA2 inference operator
│   ├── medsam3_segmentation_op.py — MedSAM3 inference operator
│   ├── data_prep_op.py            — Frame sync + disk writer
│   └── overlay_composer_op.py     — 3-panel live preview
│
├── stages/                        — Pipeline stage scripts
│   ├── format_conversion.py       — Phase 3: EndoNeRF format conversion
│   ├── stage2_medsam3.py          — Standalone MedSAM3 runner
│   ├── train_with_progress.py     — Training wrapper with progress reporting
│   └── progress.py                — Shared progress JSON read/write utilities
│
├── models/                        — Inference model code (NVIDIA-adapted)
│   ├── depth_anything_v2/         — Custom DA2 architecture (metric depth, sigmoid)
│   ├── medsam3/                   — SAM3 inference wrapper
│   └── vggt/                      — VGGT inference wrapper
│
├── training/                      — GSplat training code
│   ├── train_standalone.py        — Entry point (data accumulation + training)
│   ├── gsplat_train.py            — EndoConfig + EndoRunner (training loop)
│   ├── scene/                     — Gaussian scene modules
│   │   ├── deformation.py         — Hexplane deformation network
│   │   ├── endo_loader.py         — EndoNeRF dataset loader
│   │   └── cameras.py             — Camera utilities
│   └── utils/                     — Loss functions, SH utils, image utils
│
├── docs/                          — Documentation
│   └── tech_guide.md              — This technical guide
│
├── Dockerfile                     — Self-contained Docker image
└── README.md                      — User-facing documentation
```
