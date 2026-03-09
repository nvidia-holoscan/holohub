# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

############################################################
# G-SHARP Surgical Scene Reconstruction — Self-Contained Image
#
# This image bundles all model code (DA2, VGGT, MedSAM3, GSplat
# training) so only two volume mounts are needed at runtime:
#
#   1. assets/  — model checkpoints  (see assets/README.md)
#   2. data/    — input frames and pipeline output
#
# Build:
#   docker build -t gsplat_scene_recon:latest \
#     --build-arg BASE_IMAGE=holohub:surgical_scene_recon .
#
# Run:
#   docker run --gpus all --rm --runtime nvidia \
#     --ipc=host --network host \
#     -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
#     -v /path/to/assets:/workspace/app/assets \
#     -v /path/to/data:/workspace/data \
#     -v /path/to/output:/workspace/output \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     gsplat_scene_recon:latest \
#     python /workspace/app/run_gsharp.py \
#       --data-dir /workspace/data/frames \
#       --output-dir /workspace/output
############################################################

ARG BASE_IMAGE=holohub:surgical_scene_recon
FROM ${BASE_IMAGE}

# ── Python dependencies for all pipeline stages ──────────────
RUN pip install --no-cache-dir \
    huggingface_hub \
    einops \
    safetensors \
    timm \
    ftfy \
    regex \
    iopath \
    decord \
    pycocotools \
    psutil \
    scikit-image \
    scikit-learn \
    pandas \
    matplotlib \
    "imageio[ffmpeg]" \
    gsplat \
    lpips \
    torchmetrics \
    plyfile \
    fpsample==1.0.2 \
    open3d==0.19.0

# ── Fix decord↔libxcb conflict for HoloViz/Vulkan ───────────
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        libxcb1 libxcb-xfixes0 libxcb-shape0 libxcb-shm0 libxcb-render0 \
        libxcb-xinerama0 libxcb-randr0 && \
    rm -rf /var/lib/apt/lists/* && \
    DECORD_LIBS=$(python -c "import pathlib; print(next(pathlib.Path('/usr/local/lib').rglob('decord.libs'), ''))" 2>/dev/null) && \
    if [ -n "$DECORD_LIBS" ] && [ -d "$DECORD_LIBS" ]; then \
        for f in "$DECORD_LIBS"/libxcb*.so*; do \
            base=$(basename "$f" | sed 's/-[a-f0-9]*\./\./') && \
            sys_lib=$(find /usr/lib/x86_64-linux-gnu -name "${base%%.*}*" -type f 2>/dev/null | head -1) && \
            if [ -n "$sys_lib" ]; then ln -sf "$sys_lib" "$f"; fi; \
        done; \
    fi

# ── Install VGGT and SAM3 from their upstream repos ──────────
# These are NOT bundled in the repo; they are installed at build
# time. Pinned to specific commits for reproducibility.
#
# VGGT: Meta custom license — see https://github.com/facebookresearch/vggt
# SAM3: SAM License — see https://github.com/facebookresearch/sam3/blob/main/LICENSE
RUN pip install --no-cache-dir \
    "git+https://github.com/facebookresearch/vggt.git@44b3afbd1869d8bde4894dd8ea1e293112dd5eba" \
    "git+https://github.com/facebookresearch/sam3.git@f6e51f59500a87c576c2df2323ce56b9fd7a12de"

# ── Pre-cache VGG-16 weights (used by LPIPS loss) ───────────
RUN python -c "import torch; torch.hub.load_state_dict_from_url( \
    'https://download.pytorch.org/models/vgg16-397923af.pth')"

# ── Copy the full application into the image ─────────────────
COPY . /workspace/app

# ── Workspace layout ─────────────────────────────────────────
# /workspace/app/                  — Application code (baked in)
#   ├── run_gsharp.py              — Pipeline orchestrator
#   ├── operators/                 — Holoscan operators
#   ├── stages/                    — Pipeline stage scripts
#   ├── models/                    — Inference model code (NVIDIA-adapted)
#   │   ├── depth_anything_v2/    — DA2 model architecture
#   │   ├── medsam3/              — MedSAM3 inference wrapper
#   │   └── vggt/                 — VGGT inference wrapper
#   ├── training/                  — GSplat training code + scene/utils
#   └── assets/                   — Checkpoints (mount at runtime)
#       ├── da2/                  — depth_anything_v2_vits.pth
#       └── medsam3/              — checkpoint_8_new_best.pt
#
# VGGT and SAM3 core libraries are pip-installed (see above).
#
# /workspace/data/                 — Input frames (mount at runtime)
# /workspace/output/               — Pipeline output (mount at runtime)
# ~/.cache/huggingface/            — VGGT-1B weights (mount to persist)

WORKDIR /workspace
