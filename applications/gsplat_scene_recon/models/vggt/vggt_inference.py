# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Phase 2: VGGT Batch Pose Estimation

Runs Visual Geometry Grounded Transformer (VGGT) on accumulated frames from
Phase 1 to produce globally consistent camera extrinsics, intrinsics, and a
metric depth scale.

VGGT processes frames as a batch (not streaming) — all frames in a batch
share a common coordinate system. When the total frame count exceeds
``batch_size``, frames are split into overlapping windows so that adjacent
batches share reference frames for coordinate alignment.

Outputs (saved to ``--output-dir``):
    extrinsics.npy   — (N, 3, 4) world-to-camera [R|t], normalised so frame 0 = identity
    intrinsics.npy   — (N, 3, 3) camera intrinsic matrices (at VGGT resolution)
    depth_scale.npy  — [min, max] metric depth across all frames
    image_names.npy  — ordered list of source image filenames

Usage:
    python vggt_inference.py \
        --image-dir /path/to/phase1_output/images \
        --output-dir /path/to/vggt_output \
        --batch-size 30
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch


def _load_vggt_model(device: str):
    """Load VGGT-1B and return (model, dtype)."""
    from vggt.models.vggt import VGGT

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    print(f"[VGGT] Loading VGGT-1B on {device} (dtype={dtype})")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    return model, dtype


def _discover_images(image_dir: str) -> list[str]:
    """Return sorted list of image paths in *image_dir*."""
    exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    paths: list[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
    paths.sort()
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return paths


def _run_single_batch(
    model,
    image_paths: list[str],
    device: str,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run VGGT on a single batch and return (extrinsics, intrinsics, depth_scale).

    Returns
    -------
    extrinsics : (S, 3, 4) float64
    intrinsics : (S, 3, 3) float64
    depth_scale : (2,) float64 — [min, max] of metric depth for this batch
    """
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    images = load_and_preprocess_images(image_paths).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            preds = model(images)

    ext, intr = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])
    ext_np = ext.cpu().numpy().squeeze(0).astype(np.float64)
    intr_np = intr.cpu().numpy().squeeze(0).astype(np.float64)

    depth = preds["depth"].cpu().numpy().squeeze(0).squeeze(-1)  # (S, H, W)
    d_scale = np.array([float(depth.min()), float(depth.max())], dtype=np.float64)

    del images, preds, ext, intr
    torch.cuda.empty_cache()
    return ext_np, intr_np, d_scale


def _normalize_to_frame0(extrinsics: np.ndarray) -> np.ndarray:
    """Transform all w2c extrinsics so that frame 0 becomes the identity.

    Parameters
    ----------
    extrinsics : (N, 3, 4) w2c [R|t]

    Returns
    -------
    (N, 3, 4) normalised w2c — frame 0 is exactly [I|0].
    """
    N = extrinsics.shape[0]
    w2c_44 = np.zeros((N, 4, 4), dtype=np.float64)
    w2c_44[:, :3, :] = extrinsics
    w2c_44[:, 3, 3] = 1.0

    E0_inv = np.linalg.inv(w2c_44[0])
    for i in range(N):
        w2c_44[i] = w2c_44[i] @ E0_inv

    return w2c_44[:, :3, :]


def _align_batch_to_reference(
    ref_ext: np.ndarray,
    new_ext: np.ndarray,
) -> np.ndarray:
    """Align *new_ext* into the coordinate system of *ref_ext*.

    Uses a single frame pair: last frame of *ref_ext* and first frame of *new_ext*
    (same physical frame). Computes the rigid transform T such that T @ new = ref,
    then applies T to all of *new_ext*.
    """
    ref_44 = np.eye(4, dtype=np.float64)
    ref_44[:3, :] = ref_ext[-1]  # last frame of ref

    new_44 = np.eye(4, dtype=np.float64)
    new_44[:3, :] = new_ext[0]  # first frame of new (same physical frame)

    # T such that T @ new_44 = ref_44  →  T = ref_44 @ inv(new_44)
    T = ref_44 @ np.linalg.inv(new_44)

    N = new_ext.shape[0]
    aligned = np.zeros_like(new_ext)
    for i in range(N):
        m = np.eye(4, dtype=np.float64)
        m[:3, :] = new_ext[i]
        m_aligned = T @ m
        aligned[i] = m_aligned[:3, :]

    return aligned


def run_vggt_inference(
    image_dir: str,
    output_dir: str,
    batch_size: int = 30,
    overlap: int = 1,
    device: str = "cuda",
    progress_file: str | None = None,
) -> dict:
    """Run VGGT batch inference and save results.

    Parameters
    ----------
    image_dir : str
        Directory with images from Phase 1.
    output_dir : str
        Where to save .npy outputs.
    batch_size : int
        Max frames per VGGT forward pass.
    overlap : int
        Number of frames shared between consecutive batches for alignment.
    device : str
        "cuda" or "cpu".
    progress_file : str or None
        Optional path to write progress JSON for the HoloViz monitor.

    Returns
    -------
    dict with keys: extrinsics, intrinsics, depth_scale, image_names, output_dir
    """
    os.makedirs(output_dir, exist_ok=True)

    image_paths = _discover_images(image_dir)
    N = len(image_paths)
    print(f"[VGGT] Found {N} images in {image_dir}")

    model, dtype = _load_vggt_model(device)

    # --- Determine batching strategy ---
    if N <= batch_size:
        batches = [(0, N)]
    else:
        step = batch_size - overlap
        batches = []
        start = 0
        while start < N:
            end = min(start + batch_size, N)
            batches.append((start, end))
            if end == N:
                break
            start += step

    print(f"[VGGT] Processing {N} frames in {len(batches)} batch(es)")
    for i, (s, e) in enumerate(batches):
        print(f"  Batch {i}: frames [{s}, {e})")

    # --- Process batches ---
    all_extrinsics = []
    all_depth_scales = []
    intrinsics_first = None

    if progress_file:
        from stages.progress import update_progress
        update_progress(progress_file, "vggt", "VGGT Pose Estimation",
                        0, len(batches), "Loading model...", "running")

    t0 = time.time()
    for batch_idx, (start, end) in enumerate(batches):
        batch_paths = image_paths[start:end]
        detail = f"Batch {batch_idx + 1}/{len(batches)}: frames {start}-{end - 1}"
        print(f"\n[VGGT] {detail} ({len(batch_paths)} frames)")

        if progress_file:
            update_progress(progress_file, "vggt", "VGGT Pose Estimation",
                            batch_idx, len(batches), detail, "running")

        ext_batch, intr_batch, ds_batch = _run_single_batch(
            model, batch_paths, device, dtype
        )

        if intrinsics_first is None:
            intrinsics_first = intr_batch

        all_depth_scales.append(ds_batch)

        if batch_idx == 0:
            all_extrinsics.append(ext_batch)
        else:
            aligned = _align_batch_to_reference(
                ref_ext=all_extrinsics[-1][-overlap:] if overlap > 0 else all_extrinsics[-1][-1:],
                new_ext=ext_batch,
            )
            all_extrinsics.append(aligned[overlap:])

    elapsed = time.time() - t0
    print(f"\n[VGGT] Inference done in {elapsed:.1f}s")

    if progress_file:
        update_progress(progress_file, "vggt", "VGGT Pose Estimation",
                        len(batches), len(batches), f"Done in {elapsed:.1f}s", "complete")

    del model
    torch.cuda.empty_cache()

    # --- Concatenate and normalise ---
    extrinsics = np.concatenate(all_extrinsics, axis=0)  # (N, 3, 4)
    assert extrinsics.shape[0] == N, (
        f"Expected {N} extrinsics, got {extrinsics.shape[0]}"
    )

    extrinsics = _normalize_to_frame0(extrinsics)

    depth_scales = np.stack(all_depth_scales)
    depth_scale = np.array(
        [float(depth_scales[:, 0].min()), float(depth_scales[:, 1].max())],
        dtype=np.float64,
    )

    # Use intrinsics from the first batch for all frames (focal length is constant)
    intrinsics = np.tile(intrinsics_first[:1], (N, 1, 1))

    # --- Verification ---
    print("\n[VGGT] Verification:")
    R0 = extrinsics[0, :3, :3]
    t0_vec = extrinsics[0, :3, 3]
    print(f"  Frame 0 R deviation from I: {np.abs(R0 - np.eye(3)).max():.2e}")
    print(f"  Frame 0 t norm: {np.linalg.norm(t0_vec):.2e}")
    print(f"  Depth scale: [{depth_scale[0]:.4f}, {depth_scale[1]:.4f}]")
    print(f"  Intrinsics (frame 0): fx={intrinsics[0, 0, 0]:.2f}, "
          f"fy={intrinsics[0, 1, 1]:.2f}")

    translations = extrinsics[:, :3, 3]
    t_norms = np.linalg.norm(translations, axis=1)
    print(f"  Translation norm range: [{t_norms.min():.6f}, {t_norms.max():.6f}]")

    # --- Save ---
    image_names = [os.path.basename(p) for p in image_paths]

    np.save(os.path.join(output_dir, "extrinsics.npy"), extrinsics)
    np.save(os.path.join(output_dir, "intrinsics.npy"), intrinsics)
    np.save(os.path.join(output_dir, "depth_scale.npy"), depth_scale)
    np.save(os.path.join(output_dir, "image_names.npy"), np.array(image_names))

    print(f"\n[VGGT] Saved outputs to {output_dir}:")
    print(f"  extrinsics.npy  — {extrinsics.shape}")
    print(f"  intrinsics.npy  — {intrinsics.shape}")
    print(f"  depth_scale.npy — {depth_scale}")
    print(f"  image_names.npy — {len(image_names)} names")

    return {
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "depth_scale": depth_scale,
        "image_names": image_names,
        "output_dir": output_dir,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: VGGT batch pose estimation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image-dir", required=True,
        help="Path to Phase 1 images/ directory",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory for VGGT outputs (extrinsics, intrinsics, etc.)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=30,
        help="Max frames per VGGT forward pass",
    )
    parser.add_argument(
        "--overlap", type=int, default=1,
        help="Overlap frames between batches for coordinate alignment",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Torch device",
    )
    parser.add_argument(
        "--progress-file", default=None,
        help="Path to write progress JSON for the HoloViz monitor",
    )
    args = parser.parse_args()

    result = run_vggt_inference(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        overlap=args.overlap,
        device=args.device,
        progress_file=args.progress_file,
    )

    print(f"\n[Phase 2] Complete. {result['extrinsics'].shape[0]} poses saved.")


if __name__ == "__main__":
    main()
