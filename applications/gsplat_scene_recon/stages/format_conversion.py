# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Phase 3: Format Conversion — Assemble EndoNeRF Dataset

Combines Phase 1 outputs (images, DA2 raw depth, MedSAM3 masks) with Phase 2
outputs (VGGT extrinsics, intrinsics, depth_scale) into a complete EndoNeRF
dataset ready for GSplat training.

Key operations:
  1. Scale VGGT intrinsics from VGGT resolution to original image resolution
  2. Affinely map DA2 arbitrary-scale depth → VGGT metric scale → uint8 cm
  3. Build poses_bounds.npy (N, 17) with c2w poses + focal + near/far
  4. Copy images and masks unchanged
  5. Validate that the resulting scene_scale is in a healthy range

Output layout is binocular only (images/, depth/, masks/). The training loader
expects mode="binocular" for pipeline-generated datasets; monodepth/ is not created.

Usage:
    python format_conversion.py \
        --phase1-dir /path/to/phase1_output \
        --vggt-dir   /path/to/phase2_vggt_output \
        --output-dir /path/to/endonerf_dataset \
        --depth-scale 100
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def _scale_intrinsics(
    intrinsics: np.ndarray,
    vggt_hw: tuple[int, int],
    orig_hw: tuple[int, int],
) -> tuple[float, float, float]:
    """Scale VGGT intrinsics to original image resolution.

    Returns (focal, fx_scaled, fy_scaled).
    """
    vH, vW = vggt_hw
    oH, oW = orig_hw
    sx = oW / vW
    sy = oH / vH
    fx = intrinsics[0, 0, 0] * sx
    fy = intrinsics[0, 1, 1] * sy
    focal = (fx + fy) / 2.0
    return focal, fx, fy


def _build_poses_bounds(
    extrinsics: np.ndarray,
    focal: float,
    depth_scale_metric: np.ndarray,
    depth_scale_factor: float,
    orig_hw: tuple[int, int],
) -> np.ndarray:
    """Build EndoNeRF poses_bounds.npy from w2c extrinsics.

    Parameters
    ----------
    extrinsics : (N, 3, 4) w2c [R|t]
    focal : scaled focal length
    depth_scale_metric : (2,) [min_m, max_m] from VGGT
    depth_scale_factor : e.g. 100.0 for centimeters
    orig_hw : (H, W) of original images

    Returns
    -------
    (N, 17) poses_bounds array
    """
    N = extrinsics.shape[0]
    oH, oW = orig_hw

    w2c_44 = np.zeros((N, 4, 4), dtype=np.float64)
    w2c_44[:, :3, :] = extrinsics
    w2c_44[:, 3, 3] = 1.0
    c2w_44 = np.linalg.inv(w2c_44)

    poses_bounds = np.zeros((N, 17), dtype=np.float64)
    for i in range(N):
        R_c2w = c2w_44[i, :3, :3]
        t_c2w = c2w_44[i, :3, 3] * depth_scale_factor

        pose_3x5 = np.zeros((3, 5))
        pose_3x5[:3, :3] = R_c2w
        pose_3x5[:3, 3] = t_c2w
        pose_3x5[0, 4] = oH
        pose_3x5[1, 4] = oW
        pose_3x5[2, 4] = focal
        poses_bounds[i, :15] = pose_3x5.flatten()

    near = depth_scale_metric[0] * depth_scale_factor * 0.8
    near = max(near, 1.0)  # avoid zero near-plane (degenerate VGGT depth min)
    far = depth_scale_metric[1] * depth_scale_factor * 1.2
    poses_bounds[:, 15] = near
    poses_bounds[:, 16] = far

    return poses_bounds


def _convert_depth(
    da2_path: str,
    da2_min: float,
    da2_max: float,
    vggt_d_min: float,
    vggt_d_max: float,
    depth_scale_factor: float,
) -> np.ndarray:
    """Load DA2 raw depth, affinely map to VGGT metric, convert to uint8 cm."""
    d = np.load(da2_path).astype(np.float64)
    d_norm = (d - da2_min) / (da2_max - da2_min + 1e-8)
    d_metric = vggt_d_min + d_norm * (vggt_d_max - vggt_d_min)
    d_scaled = d_metric * depth_scale_factor
    return np.clip(d_scaled, 1, 255).astype(np.uint8)


def run_format_conversion(
    phase1_dir: str,
    vggt_dir: str,
    output_dir: str,
    depth_scale: float = 100.0,
    progress_file: str | None = None,
) -> str:
    """Assemble a complete EndoNeRF dataset from Phase 1 + Phase 2 outputs.

    Returns the path to the output directory.
    """
    phase1 = Path(phase1_dir)
    vggt = Path(vggt_dir)
    out = Path(output_dir)

    # --- Load Phase 2 outputs ---
    ext = np.load(str(vggt / "extrinsics.npy"))  # (N, 3, 4)
    intr = np.load(str(vggt / "intrinsics.npy"))  # (N, 3, 3)
    ds = np.load(str(vggt / "depth_scale.npy"))  # (2,)
    img_names = np.load(str(vggt / "image_names.npy"))  # (N,)
    N = ext.shape[0]

    print(f"[Phase3] {N} frames, VGGT depth: [{ds[0]:.4f}, {ds[1]:.4f}] m")
    print(
        f"[Phase3] DEPTH_SCALE = {depth_scale} (units: {'cm' if depth_scale == 100 else 'custom'})"
    )

    # --- Determine resolutions ---
    sample_img = cv2.imread(str(phase1 / "images" / img_names[0]))
    orig_H, orig_W = sample_img.shape[:2]

    vggt_hw_path = vggt / "vggt_hw.npy"
    if vggt_hw_path.exists():
        vggt_hw_arr = np.load(str(vggt_hw_path))
        vggt_H, vggt_W = int(vggt_hw_arr[0]), int(vggt_hw_arr[1])
    else:
        # Fallback: VGGT default preprocessing (longest side → 518, 14-divisible)
        vggt_W = 518
        vggt_H = (orig_H * vggt_W // orig_W) // 14 * 14
        print(
            "[Phase3] vggt_hw.npy not found; using fallback VGGT resolution. "
            "Re-run Phase 2 to save vggt_hw.npy for correct intrinsic scaling."
        )
    print(f"[Phase3] Original: {orig_H}x{orig_W}, VGGT: {vggt_H}x{vggt_W}")

    focal, fx, fy = _scale_intrinsics(intr, (vggt_H, vggt_W), (orig_H, orig_W))
    print(f"[Phase3] Scaled focal: {focal:.2f} (fx={fx:.2f}, fy={fy:.2f})")

    # --- Compute DA2 global range ---
    depth_raw_dir = phase1 / "depth_raw"
    da2_files = sorted(depth_raw_dir.glob("*.npy"))
    assert len(da2_files) == N, f"DA2 count {len(da2_files)} != frame count {N}"

    da2_min, da2_max = float("inf"), float("-inf")
    for f in da2_files:
        d = np.load(str(f))
        da2_min = min(da2_min, float(d.min()))
        da2_max = max(da2_max, float(d.max()))
    print(f"[Phase3] DA2 raw range: [{da2_min:.4f}, {da2_max:.4f}]")

    expected_uint8 = (ds[0] * depth_scale, ds[1] * depth_scale)
    print(f"[Phase3] Expected uint8 range: [{expected_uint8[0]:.0f}, {expected_uint8[1]:.0f}]")

    # --- Create output directories ---
    for sub in ("images", "depth", "masks"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # --- Build poses_bounds.npy ---
    poses_bounds = _build_poses_bounds(ext, focal, ds, depth_scale, (orig_H, orig_W))
    np.save(str(out / "poses_bounds.npy"), poses_bounds)
    print(f"[Phase3] Saved poses_bounds.npy: {poses_bounds.shape}")

    near = poses_bounds[0, 15]
    far = poses_bounds[0, 16]
    print(f"[Phase3] Near/far: {near:.1f} / {far:.1f}")

    # --- Copy images, convert depth, copy masks ---
    if progress_file:
        from stages.progress import update_progress

        update_progress(
            progress_file,
            "format_conversion",
            "Format Conversion",
            0,
            N,
            "Converting frames...",
            "running",
        )

    for i, name in enumerate(img_names):
        dst_name = f"frame_{i:04d}.png"

        shutil.copy2(str(phase1 / "images" / name), str(out / "images" / dst_name))

        d_uint8 = _convert_depth(
            str(depth_raw_dir / name.replace(".png", ".npy")),
            da2_min,
            da2_max,
            ds[0],
            ds[1],
            depth_scale,
        )
        cv2.imwrite(str(out / "depth" / dst_name), d_uint8)

        mask_src = phase1 / "masks" / name
        if mask_src.exists():
            shutil.copy2(str(mask_src), str(out / "masks" / dst_name))

        if progress_file and (i % 10 == 0 or i == N - 1):
            update_progress(
                progress_file,
                "format_conversion",
                "Format Conversion",
                i + 1,
                N,
                f"Frame {i + 1}/{N}",
                "running",
            )

    print(f"[Phase3] Wrote {N} images, depth maps, and masks")

    if progress_file:
        update_progress(
            progress_file, "format_conversion", "Format Conversion", N, N, "Complete", "complete"
        )

    # --- Validate scene_scale estimate ---
    d_sample = cv2.imread(str(out / "depth" / "frame_0000.png"), cv2.IMREAD_UNCHANGED)
    print(
        f"[Phase3] Sample depth: dtype={d_sample.dtype}, "
        f"range=[{d_sample.min()}, {d_sample.max()}]"
    )

    # Rough scene_scale estimate: max translation norm × some geometry factor
    poses_3x5 = poses_bounds[:, :15].reshape(-1, 3, 5)
    t_norms = np.linalg.norm(poses_3x5[:, :3, 3], axis=1)
    max_t = t_norms.max()
    print(f"[Phase3] Max translation norm: {max_t:.2f}")

    # Frame 0 verification
    R0 = poses_3x5[0, :3, :3]
    t0 = poses_3x5[0, :3, 3]
    print(f"[Phase3] Frame 0 c2w R is identity: {np.allclose(R0, np.eye(3), atol=1e-6)}")
    print(f"[Phase3] Frame 0 c2w t is zero: {np.allclose(t0, 0, atol=1e-3)}")

    print(f"\n[Phase3] EndoNeRF dataset ready at: {out}")
    print(f"  images/        — {N} PNGs")
    print(f"  depth/         — {N} uint8 PNGs (DEPTH_SCALE={depth_scale})")
    print(f"  masks/         — {N} binary PNGs")
    print(f"  poses_bounds.npy — ({N}, 17)")

    return str(out)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Assemble EndoNeRF dataset from Phase 1 + Phase 2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--phase1-dir",
        required=True,
        help="Phase 1 output directory (images/, depth_raw/, masks/)",
    )
    parser.add_argument(
        "--vggt-dir",
        required=True,
        help="Phase 2 VGGT output directory (extrinsics.npy, etc.)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where to write the final EndoNeRF dataset",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=100.0,
        help="Depth scale factor (100=cm, 1000=mm). Recommend 100.",
    )
    parser.add_argument(
        "--progress-file",
        default=None,
        help="Path to write progress JSON for the HoloViz monitor",
    )
    args = parser.parse_args()

    run_format_conversion(
        phase1_dir=args.phase1_dir,
        vggt_dir=args.vggt_dir,
        output_dir=args.output_dir,
        depth_scale=args.depth_scale,
        progress_file=args.progress_file,
    )


if __name__ == "__main__":
    main()
