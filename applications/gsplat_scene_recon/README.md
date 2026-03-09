# G-SHARP: Gaussian Splatting for Holographic Anatomical Reconstruction Pipeline

Real-time surgical scene reconstruction using Gaussian Splatting, powered by
the NVIDIA Holoscan SDK.

## Pipeline Overview

G-SHARP runs five sequential phases to reconstruct a 3D scene from surgical
video frames:

| Phase | Component | Framework |
|-------|-----------|-----------|
| **1** | Parallel DA2 depth + MedSAM3 segmentation | Holoscan streaming |
| **2** | VGGT batch camera pose estimation | Standalone PyTorch |
| **3** | EndoNeRF format assembly | Python script |
| **4** | GSplat training with deformation network | Standalone PyTorch |
| **5** | Live render viewer / interactive 3D viewer | Holoscan streaming |

## Requirements

### Hardware

- **NVIDIA GPU** with CUDA 12+ and Vulkan support (tested on RTX 6000 Ada)
- **Display** configured for X11 (for Phase 1 / Phase 5 visualization)

### Software

- **Docker** with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- **Base image**: `holohub:surgical_scene_recon` (see Holohub build instructions)

## Assets (Checkpoints)

Before building the Docker image, place the following checkpoints in the
`assets/` directory:

| File | Path | Source |
|------|------|--------|
| Depth Anything V2 (Small) | `assets/da2/depth_anything_v2_vits.pth` | [HuggingFace](https://huggingface.co/depth-anything/Depth-Anything-V2-Small) |
| MedSAM3 checkpoint | `assets/medsam3/checkpoint_8_new_best.pt` | Custom fine-tuned checkpoint |

> **Important**: The MedSAM3 checkpoint must be a real `.pt` file (not a
> placeholder). Verify it is non-empty before building:
> `ls -lh assets/medsam3/checkpoint_8_new_best.pt`

VGGT weights (`facebook/VGGT-1B`) are downloaded automatically from
HuggingFace on first run. Mount `~/.cache/huggingface` to persist them across
runs so they are not re-downloaded each time.

## Quick Start

### 1. Build the Docker Image

```bash
cd applications/gsplat_scene_recon

docker build -t gsplat_scene_recon:latest \
  --build-arg BASE_IMAGE=holohub:surgical_scene_recon .
```

The build installs all Python dependencies, pip-installs VGGT and SAM3 from
their upstream repos, pre-caches VGG-16 weights for the LPIPS loss, and copies
the application code into the image. Dependency layers are cached, so
subsequent rebuilds after code-only changes complete in seconds.

### 2. Prepare Input Data

The pipeline expects a directory of sequential PNG frames as input. Frame
filenames should sort alphabetically in temporal order (e.g.,
`frame-000000.color.png`, `frame-000001.color.png`, ...).

### 3. Run the Full End-to-End Pipeline

```bash
docker run --name gsharp --rm \
  --runtime nvidia --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /path/to/input/frames:/workspace/data/frames:ro \
  -v /path/to/output:/workspace/output \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e PYTHONUNBUFFERED=1 \
  gsplat_scene_recon:latest \
  python /workspace/app/run_gsharp.py \
    --data-dir /workspace/data/frames \
    --output-dir /workspace/output \
    --training-iterations 1400 \
    --coarse-iterations 200
```

Replace `/path/to/input/frames` with the directory containing your PNG frames,
and `/path/to/output` with where you want the pipeline artifacts written.

> **Note on the MedSAM3 checkpoint**: If the `assets/medsam3/` checkpoint
> baked into the image is a placeholder (0 bytes), bind-mount the real file:
>
> ```bash
> -v /path/to/checkpoint_8_new_best.pt:/workspace/app/assets/medsam3/checkpoint_8_new_best.pt:ro
> ```

### 4. What Happens

The orchestrator (`run_gsharp.py`) runs all five phases sequentially:

1. **Phase 1** — DA2 depth estimation + MedSAM3 segmentation (Holoscan
   streaming app). Writes `images/`, `depth/`, `masks/` to
   `<output>/phase1_raw/`.
2. **Phase 2** — VGGT batch camera pose estimation. Writes camera extrinsics
   and intrinsics to `<output>/phase2_vggt/`.
3. **Phase 3** — EndoNeRF format conversion. Assembles everything into
   `<output>/phase3_endonerf/`.
4. **Phase 4** — GSplat training with deformation network. Writes checkpoints
   to `<output>/phase4_training/`.
5. **Phase 5** — Live render viewer. Opens a HoloViz window looping through
   rendered frames at 30 fps. Close the window to exit.

### 5. Headless / Automated Runs

To run without any display (e.g., on a remote server), add `--headless` and
optionally `--skip-viewer`:

```bash
docker run --name gsharp --rm \
  --runtime nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /path/to/input/frames:/workspace/data/frames:ro \
  -v /path/to/output:/workspace/output \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e PYTHONUNBUFFERED=1 \
  gsplat_scene_recon:latest \
  python /workspace/app/run_gsharp.py \
    --data-dir /workspace/data/frames \
    --output-dir /workspace/output \
    --headless \
    --skip-viewer
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data-dir` | Directory containing input PNG frames | **Required** |
| `--output-dir` | Base output directory for all pipeline artifacts | **Required** |
| `--training-iterations` | Total GSplat training iterations | `1400` |
| `--coarse-iterations` | Coarse stage iterations (fixed camera) | `200` |
| `--no-deformation` | Disable deformation network (static scene) | `False` |
| `--batch-size` | VGGT batch size (frames per batch) | `30` |
| `--depth-scale` | Depth scale factor (100 = centimeters) | `100.0` |
| `--fps` | Render viewer playback FPS | `30` |
| `--headless` | Run Holoscan apps without visualization | `False` |
| `--skip-phase1` | Skip Phase 1 (reuse existing depth/masks) | `False` |
| `--skip-phase2` | Skip Phase 2 (reuse existing VGGT poses) | `False` |
| `--skip-phase3` | Skip Phase 3 (reuse existing EndoNeRF data) | `False` |
| `--skip-training` | Skip Phase 4 (no training) | `False` |
| `--skip-viewer` | Skip Phase 5 (no live viewer) | `False` |
| `--da2-root` | DA2 model code directory | Auto (bundled) |
| `--da2-checkpoint` | DA2 `.pth` checkpoint | Auto (bundled) |
| `--da2-encoder` | DA2 encoder variant (`vits`, `vitb`, `vitl`) | `vits` |
| `--sam3-checkpoint` | MedSAM3 `.pt` checkpoint | Auto (bundled) |
| `--train-script` | Path to `train_standalone.py` | Auto (bundled) |
| `--progress-file` | JSON file for progress monitor | Auto-generated |

## Incremental Runs

Use `--skip-*` flags to re-run only specific phases. For example, to re-train
with more iterations while reusing Phase 1–3 output:

```bash
docker run --name gsharp --rm \
  --runtime nvidia --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /path/to/output:/workspace/output \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e PYTHONUNBUFFERED=1 \
  gsplat_scene_recon:latest \
  python /workspace/app/run_gsharp.py \
    --data-dir /workspace/data/frames \
    --output-dir /workspace/output \
    --skip-phase1 --skip-phase2 --skip-phase3 \
    --training-iterations 7000 \
    --coarse-iterations 1000
```

## Output Structure

After a full run, `<output-dir>/` contains:

```
<output-dir>/
├── phase1_raw/
│   ├── images/          # Resized input frames
│   ├── depth/           # DA2 depth maps (*.npy)
│   └── masks/           # MedSAM3 segmentation masks (*.png)
├── phase2_vggt/
│   ├── poses.npy        # Camera extrinsics (4x4 matrices)
│   └── intrinsics.npy   # Camera intrinsics
├── phase3_endonerf/
│   ├── images/          # Final frames in EndoNeRF layout
│   ├── depth/           # Scaled depth maps
│   ├── masks/           # Binary masks
│   └── poses_bounds.npy # LLFF-format camera parameters
├── phase4_training/
│   └── trained_model/
│       └── ckpts/
│           ├── fine_best_psnr.pt   # Best checkpoint by PSNR
│           └── fine_step*.pt       # Step checkpoints
└── progress.json        # Live progress (updated during run)
```

## Third-Party Dependencies and Licenses

This application uses several third-party libraries. **By building and running
this application, you acknowledge and accept the license terms of these
libraries.**

### VGGT (Visual Geometry Grounded Transformer)

- **Source**: <https://github.com/facebookresearch/vggt>
- **License**: [Meta License](https://github.com/facebookresearch/vggt/blob/main/LICENSE)
- **Installation**: Pip-installed from source at Docker build time
- **Model weights**: `facebook/VGGT-1B` from HuggingFace

> **Important**: The VGGT code and model weights are released under a custom
> Meta license. By using VGGT, you agree to be bound by its terms. The
> `facebook/VGGT-1B` checkpoint is licensed for **non-commercial use** only.
> A separate `facebook/VGGT-1B-Commercial` checkpoint is available for
> commercial use — see the [VGGT repository](https://github.com/facebookresearch/vggt)
> for details.

### SAM3 (Segment Anything Model 3)

- **Source**: <https://github.com/facebookresearch/sam3>
- **License**: [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE)
- **Installation**: Pip-installed from source at Docker build time

> **Important**: SAM3 is released under a custom SAM License. By using SAM3,
> you agree to be bound by its terms. Key restrictions include:
>
> - No military, weapons, ITAR, nuclear, espionage, or sanctions-violating use
> - Publications using SAM3 must acknowledge the SAM Materials
> - Redistribution must include the same license terms
>
> HuggingFace checkpoints for SAM3 may require **access approval** at
> <https://huggingface.co/facebook/sam3>.

### Depth Anything V2

- **Source**: <https://github.com/DepthAnything/Depth-Anything-V2>
- **License**: Apache-2.0 (code and Small model)
- **Bundled**: `models/depth_anything_v2/`

> The Small model (`depth_anything_v2_vits.pth`) used by this application is
> Apache-2.0 licensed. Base/Large/Giant models are CC-BY-NC-4.0.

### EndoGaussian / GSplat Training

- **Source**: Derived from <https://github.com/yifliu3/EndoGaussian>
- **License**: MIT (upstream EndoGaussian) + Apache-2.0 (NVIDIA modifications)
- **Bundled**: `training/`

> The bundled training code is a custom derivative that replaces the original
> CUDA rasterizer with the `gsplat` library. It is not a direct copy of the
> upstream repository.

### Additional Python Libraries

| Library | License | URL |
|---------|---------|-----|
| PyTorch | BSD-3-Clause | <https://github.com/pytorch/pytorch> |
| CuPy | MIT | <https://github.com/cupy/cupy> |
| gsplat | Apache-2.0 | <https://github.com/nerfstudio-project/gsplat> |
| LPIPS | BSD-2-Clause | <https://github.com/richzhang/PerceptualSimilarity> |
| HuggingFace Hub | Apache-2.0 | <https://github.com/huggingface/huggingface_hub> |
| einops | MIT | <https://github.com/arogozhnikov/einops> |
| timm | Apache-2.0 | <https://github.com/huggingface/pytorch-image-models> |
