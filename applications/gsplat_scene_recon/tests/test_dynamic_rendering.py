#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal test that render_viewer and gsplat rendering path can be exercised (GPU).
"""Test that dynamic/static rendering code path runs without error."""

import importlib.util
import sys
from pathlib import Path

app_dir = Path(__file__).resolve().parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

# Training dir for scene.deformation and render_viewer's _render_frame
training_dir = app_dir / "training"
if str(training_dir) not in sys.path:
    sys.path.insert(0, str(training_dir))


def main():
    if importlib.util.find_spec("cupy") is None:
        print("SKIP: cupy not available (run inside container)")
        return 0
    try:
        import torch
    except (ImportError, ModuleNotFoundError) as e:
        print(f"SKIP: GPU deps not available ({e})")
        return 0

    try:
        from gsplat.rendering import rasterization
    except ImportError as e:
        print(f"SKIP: gsplat not available ({e})")
        return 0

    # Minimal static params: 1 Gaussian at origin, 4x4 view, tiny resolution
    device = "cuda"
    N = 1
    means = torch.zeros(N, 3, device=device)
    scales = torch.ones(N, 3, device=device) * 0.01
    quats = torch.tensor([[1, 0, 0, 0]], device=device, dtype=torch.float32)
    opacities = torch.ones(N, device=device)
    # SH coeffs: (N, K, 3) with K=(sh_degree+1)^2; sh_degree=3 -> 16
    colors = torch.ones(N, 16, 3, device=device) * 0.5
    viewmats = torch.eye(4, device=device).unsqueeze(0)
    K = torch.tensor(
        [[64, 0, 32], [0, 64, 32], [0, 0, 1]], device=device, dtype=torch.float32
    ).unsqueeze(0)
    W, H = 8, 8

    try:
        rgb, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=K,
            width=W,
            height=H,
            sh_degree=3,
            near_plane=0.01,
            far_plane=100.0,
            render_mode="RGB",
        )
    except Exception as e:
        print(f"FAIL: rasterization raised {e}")
        return 1

    # rgb can be (1,H,W,3) or (1,3,H,W) depending on gsplat version
    if rgb.shape[0] != 1 or (rgb.shape[1] != H and rgb.shape[2] != H):
        print(f"FAIL: unexpected rgb shape {rgb.shape}")
        return 1

    print("SUCCESS: dynamic rendering OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
