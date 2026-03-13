# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Depth conversion utilities.

Converts Long Tracks 4D sparse per-point depths into dense depth images
suitable for GSplat training.
"""

import warnings

import numpy as np


def sparse_depths_to_dense(
    tracks_2d: np.ndarray,
    depths: np.ndarray,
    visibility: np.ndarray,
    height: int,
    width: int,
    method: str = "nearest",
) -> np.ndarray:
    """
    Convert sparse per-point depths to a dense depth image.

    Args:
        tracks_2d: [N, 2] pixel coordinates (x, y) of tracked points
        depths: [N] depth values at tracked points
        visibility: [N] boolean visibility mask
        height: Output image height
        width: Output image width
        method: Interpolation method ("nearest" or "linear")

    Returns:
        dense_depth: [H, W] dense depth image (float32, 0 where unknown)
    """
    n_pts = len(tracks_2d)
    if n_pts != len(depths) or n_pts != len(visibility):
        raise ValueError(
            "sparse_depths_to_dense: tracks_2d, depths, and visibility must have the same "
            f"first dimension, got {n_pts}, {len(depths)}, {len(visibility)}"
        )
    if tracks_2d.ndim != 2 or tracks_2d.shape[1] != 2:
        raise ValueError(
            f"sparse_depths_to_dense: tracks_2d must be [N, 2], got shape {tracks_2d.shape}"
        )
    if method not in ("nearest", "linear"):
        raise ValueError(
            f"sparse_depths_to_dense: method must be 'nearest' or 'linear', got {method!r}"
        )
    if height <= 0 or width <= 0:
        raise ValueError(
            f"sparse_depths_to_dense: height and width must be positive, got {height}, {width}"
        )

    dense_depth = np.zeros((height, width), dtype=np.float32)

    visible = visibility.astype(bool) if visibility.dtype != bool else visibility
    valid = visible & (depths > 0)

    if not np.any(valid):
        return dense_depth

    pts = tracks_2d[valid]
    d = depths[valid]

    x = np.clip(np.round(pts[:, 0]).astype(int), 0, width - 1)
    y = np.clip(np.round(pts[:, 1]).astype(int), 0, height - 1)

    # When multiple tracks round to the same (y, x), use median depth per pixel
    # to avoid discarding valid depth and to be robust to outliers (CodeRabbit).
    linear_idx = y * width + x
    unique_idx = np.unique(linear_idx)
    dense_flat = dense_depth.ravel()
    for i in unique_idx:
        sel = linear_idx == i
        dense_flat[i] = np.median(d[sel]).astype(np.float32)

    if method == "linear" and np.sum(valid) >= 4:
        try:
            from scipy.interpolate import griddata

            grid_y, grid_x = np.mgrid[0:height, 0:width]
            points = np.stack([x, y], axis=-1).astype(np.float64)
            interpolated = griddata(
                points,
                d.astype(np.float64),
                (grid_x, grid_y),
                method="linear",
                fill_value=0.0,
            )
            mask = dense_depth == 0
            dense_depth[mask] = interpolated[mask].astype(np.float32)
        except ImportError:
            # SciPy is not available; skip linear interpolation and keep the
            # nearest/median depth result computed above.
            warnings.warn(
                "scipy is not installed; falling back to non-interpolated "
                "depths in sparse_depths_to_dense(method='linear').",
                RuntimeWarning,
                stacklevel=2,
            )

    return dense_depth


def depth_to_png16(depth: np.ndarray, scale: float = 1000.0) -> np.ndarray:
    """
    Convert float32 depth to uint16 PNG-compatible format.

    Follows the EndoNeRF convention where depth is stored as raw values
    in a single-channel image.

    Args:
        depth: [H, W] float32 depth in meters
        scale: Scale factor (depth_mm = depth_m * scale)

    Returns:
        depth_uint16: [H, W] uint16 depth image
    """
    depth_scaled = (depth * scale).clip(0, 65535)
    return depth_scaled.astype(np.uint16)


def depth_to_png8(depth: np.ndarray) -> np.ndarray:
    """
    Convert float32 depth to uint8 PNG format (normalized).

    For EndoNeRF binocular depth: values are stored as-is (the loader
    clips to percentile bounds). We normalize to the valid range.

    Args:
        depth: [H, W] float32 depth

    Returns:
        depth_uint8: [H, W] uint8 depth image
    """
    valid = depth > 0
    if not np.any(valid):
        return np.zeros_like(depth, dtype=np.uint8)

    d_min = np.percentile(depth[valid], 1.0)
    d_max = np.percentile(depth[valid], 99.0)

    if d_max <= d_min:
        return np.zeros_like(depth, dtype=np.uint8)

    normalized = np.clip((depth - d_min) / (d_max - d_min), 0, 1)
    normalized[~valid] = 0
    return (normalized * 255).astype(np.uint8)


def unproject_depth_to_points(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Unproject a dense depth image to 3D points in camera space.

    Args:
        depth: [H, W] depth image
        K: [3, 3] camera intrinsic matrix

    Returns:
        points: [M, 3] 3D points where depth > 0
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid = depth > 0

    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy

    return np.stack([x, y, z], axis=-1).astype(np.float32)
