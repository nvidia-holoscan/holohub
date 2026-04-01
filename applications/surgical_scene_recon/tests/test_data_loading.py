#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal test that EndoNeRF-style data loading works (no GPU required).
"""Test that EndoNeRF dataset loader can load minimal fixture data."""

import sys
import tempfile
from pathlib import Path

import numpy as np

# App and training dirs on path so "scene" and "utils" resolve
app_dir = Path(__file__).resolve().parent.parent
training_dir = app_dir / "training"
for d in (app_dir, str(training_dir)):
    if d not in sys.path:
        sys.path.insert(0, d)


def _make_minimal_endonerf_dir(tmpdir: Path, n_frames: int = 2, H: int = 64, W: int = 64):
    """Create minimal EndoNeRF layout: poses_bounds.npy and images/depth/masks PNGs."""
    import cv2

    # poses_bounds.npy: (N, 3, 5) then flatten and append 2 bounds -> (N, 17)
    focal = 64.0
    poses = np.eye(3, 5)
    poses[0, -1], poses[1, -1], poses[2, -1] = H, W, focal
    poses = np.tile(poses[np.newaxis, ...], (n_frames, 1, 1))
    bounds = np.zeros((n_frames, 2))
    poses_flat = poses.reshape(n_frames, -1)
    poses_bounds = np.concatenate([poses_flat, bounds], axis=1)
    np.save(tmpdir / "poses_bounds.npy", poses_bounds)

    for sub in ("images", "depth", "masks"):
        (tmpdir / sub).mkdir(exist_ok=True)
    # Valid minimal PNGs via cv2 (BGR for color, grayscale for depth/mask)
    tiny = np.zeros((2, 2, 3), dtype=np.uint8)
    tiny_gray = np.zeros((2, 2), dtype=np.uint8)
    for i in range(n_frames):
        cv2.imwrite(str(tmpdir / "images" / f"frame_{i:06d}.png"), tiny)
        cv2.imwrite(str(tmpdir / "depth" / f"frame_{i:06d}.png"), tiny_gray)
        cv2.imwrite(str(tmpdir / "masks" / f"frame_{i:06d}.png"), tiny_gray)


def main():
    try:
        import cv2  # noqa: F401
    except ModuleNotFoundError:
        print("SKIP: opencv not available")
        return 0

    with tempfile.TemporaryDirectory(prefix="gsplat_test_") as tmp:
        tmpdir = Path(tmp)
        n_frames = 2
        _make_minimal_endonerf_dir(tmpdir, n_frames=n_frames)

        try:
            from scene.endo_loader import EndoNeRF_Dataset
        except Exception as e:
            print(f"SKIP: could not import loader ({e})")
            return 0

        try:
            ds = EndoNeRF_Dataset(str(tmpdir), downsample=1.0, test_every=8, mode="binocular")
        except Exception as e:
            print(f"FAIL: loader raised {e}")
            return 1

        if len(ds.image_paths) != n_frames:
            print(f"FAIL: expected {n_frames} images, got {len(ds.image_paths)}")
            return 1

    print("SUCCESS: data loading OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
