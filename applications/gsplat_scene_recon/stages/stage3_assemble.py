# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Stage 3: Validate and Finalize EndoNeRF Dataset

Verifies that Stage 1 (frames, depths, poses) and Stage 2 (masks) outputs
are consistent and complete. Reports warnings for missing or mismatched data.

Usage:
    python stage3_assemble.py --data-dir <endonerf_output_dir>
"""

import sys
from pathlib import Path
from argparse import ArgumentParser

import numpy as np


def validate_endonerf_dataset(data_dir: str) -> bool:
    """
    Check that all EndoNeRF components exist and are consistent.

    Returns True if the dataset passes all critical checks.
    """
    data_path = Path(data_dir)
    images_dir = data_path / "images"
    masks_dir = data_path / "masks"
    depth_dir = data_path / "depth"
    poses_file = data_path / "poses_bounds.npy"

    critical_errors = []
    warnings = []

    # --- Existence checks ---
    if not images_dir.exists():
        critical_errors.append(f"Missing images directory: {images_dir}")
    if not masks_dir.exists():
        critical_errors.append(f"Missing masks directory: {masks_dir}")
    if not depth_dir.exists():
        critical_errors.append(f"Missing depth directory: {depth_dir}")
    if not poses_file.exists():
        critical_errors.append(f"Missing poses file: {poses_file}")

    if critical_errors:
        for e in critical_errors:
            print(f"[Stage3] ERROR: {e}")
        return False

    # --- Count checks ---
    images = sorted(images_dir.glob("*.png"))
    masks = sorted(masks_dir.glob("*.png"))
    depths = sorted(depth_dir.glob("*.png"))
    n_images = len(images)
    n_masks = len(masks)
    n_depths = len(depths)

    print(f"[Stage3] Dataset: {data_path}")
    print(f"[Stage3]   Images: {n_images}")
    print(f"[Stage3]   Masks:  {n_masks}")
    print(f"[Stage3]   Depths: {n_depths}")

    if n_images == 0:
        critical_errors.append("No images found")

    if n_masks == 0:
        warnings.append("No masks found (Stage 2 may not have run)")
    elif n_masks != n_images:
        warnings.append(f"Mask count ({n_masks}) != image count ({n_images})")

    if n_depths == 0:
        warnings.append("No depth maps found")
    elif n_depths != n_images:
        warnings.append(f"Depth count ({n_depths}) != image count ({n_images})")

    # --- Poses validation ---
    poses = np.load(str(poses_file))
    print(f"[Stage3]   Poses:  {poses.shape}")

    if poses.shape[1] != 17:
        critical_errors.append(
            f"poses_bounds.npy should have 17 columns, got {poses.shape[1]}"
        )
    else:
        pose_3x5 = poses[:, :15].reshape(-1, 3, 5)
        rotation_matrices = pose_3x5[:, :3, :3]
        zero_poses = np.all(rotation_matrices == 0, axis=(1, 2))
        n_valid_poses = int(np.sum(~zero_poses))
        print(f"[Stage3]   Valid poses (non-zero R): {n_valid_poses}/{poses.shape[0]}")

        if n_valid_poses == 0:
            critical_errors.append("All camera poses are zero — tracking may have failed")
        elif n_valid_poses < n_images * 0.5:
            warnings.append(
                f"Only {n_valid_poses}/{n_images} poses are non-zero — "
                "tracking coverage is incomplete"
            )

    if poses.shape[0] < n_images:
        warnings.append(
            f"Fewer poses ({poses.shape[0]}) than images ({n_images})"
        )

    # --- Report ---
    for w in warnings:
        print(f"[Stage3] WARNING: {w}")
    for e in critical_errors:
        print(f"[Stage3] ERROR: {e}")

    if critical_errors:
        print("[Stage3] Dataset validation FAILED")
        return False

    print("[Stage3] Dataset validation PASSED")
    return True


if __name__ == "__main__":
    parser = ArgumentParser(description="Stage 3: Validate EndoNeRF dataset")
    parser.add_argument("--data-dir", required=True, help="EndoNeRF output directory")
    args = parser.parse_args()

    valid = validate_endonerf_dataset(args.data_dir)
    sys.exit(0 if valid else 1)
