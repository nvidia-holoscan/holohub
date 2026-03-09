# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Camera pose conversion utilities.

Converts Long Tracks 4D camera parameters (rotation_params, position_params)
to the EndoNeRF poses_bounds.npy format.
"""

import ast

import numpy as np


def long_tracks_to_c2w(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    """
    Convert Long Tracks 4D camera parameters to a 4x4 camera-to-world matrix.

    Long Tracks 4D outputs:
        rotation_params: [3, 3] rotation matrix (camera orientation in world)
        position_params: [3] translation vector (camera position in world)

    Returns:
        c2w: [4, 4] camera-to-world homogeneous matrix
    """
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = position
    return c2w


def c2w_to_poses_bounds_row(c2w: np.ndarray, height: int, width: int, focal: float,
                            near: float = 0.01, far: float = 100.0) -> np.ndarray:
    """
    Convert a single c2w matrix to one row of poses_bounds.npy.

    EndoNeRF format: each row is [3x5 flattened, near, far] = 17 values
    where the 3x5 block is [R | T | [H, W, focal]^T]

    The convention stores R and T from the c2w matrix as:
        pose_3x4 = c2w[:3, :4]   (3x4 = [R | T])
        hwf = [[H], [W], [focal]]
        row_3x5 = [pose_3x4 | hwf]
    """
    pose_3x4 = c2w[:3, :4]
    hwf = np.array([[height], [width], [focal]], dtype=np.float32)
    row_3x5 = np.concatenate([pose_3x4, hwf], axis=1)
    row_flat = row_3x5.flatten()
    return np.concatenate([row_flat, [near, far]]).astype(np.float32)


def build_poses_bounds(rotations: np.ndarray, positions: np.ndarray,
                       height: int, width: int, focal: float,
                       near: float = 0.01, far: float = 100.0) -> np.ndarray:
    """
    Build a full poses_bounds.npy array from Long Tracks 4D outputs.

    Args:
        rotations: [T, 3, 3] per-frame rotation matrices
        positions: [T, 3] per-frame translation vectors
        height: Image height in pixels
        width: Image width in pixels
        focal: Focal length in pixels
        near: Near clipping plane
        far: Far clipping plane

    Returns:
        poses_bounds: [T, 17] array in EndoNeRF format
    """
    T = rotations.shape[0]
    rows = []
    for t in range(T):
        c2w = long_tracks_to_c2w(rotations[t], positions[t])
        row = c2w_to_poses_bounds_row(c2w, height, width, focal, near, far)
        rows.append(row)
    return np.stack(rows, axis=0)


def extract_intrinsics_from_calibration(calibration_matrix_str: str) -> dict:
    """
    Parse the calibration matrix string from Long Tracks 4D config
    and extract focal length and principal point.

    Args:
        calibration_matrix_str: String like "[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]"

    Returns:
        dict with keys: fx, fy, cx, cy, K (3x3 numpy array)
    """
    K = np.array(ast.literal_eval(calibration_matrix_str), dtype=np.float32)
    return {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "K": K,
    }
