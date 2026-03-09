# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Stage 2: MedSAM3 Tool Segmentation

Runs MedSAM3 text-prompted segmentation on all frames saved by Stage 1,
producing per-frame binary tool masks.

This is a pure Python script (no Holoscan, no TensorRT) to avoid the CUDA
stream conflicts that arise when running PyTorch alongside TensorRT CUDA
graph capture in the same process.

Usage:
    python stage2_medsam3.py \\
        --input-dir <stage1_output_dir> \\
        [--sam3-checkpoint <path_to_medsam3_checkpoint>] \\
        [--text-prompt "surgical tool"] \\
        [--score-threshold 0.3]

Uses the bundled MedSAM3 wrapper in models/medsam3/ (sam3 package is pip-installed).
"""

import os
import sys
import time
from pathlib import Path
from argparse import ArgumentParser

import cv2
import numpy as np
import torch


def run_medsam3_segmentation(
    input_dir: str,
    checkpoint_path: str,
    text_prompt: str = "surgical tool",
    score_threshold: float = 0.3,
    max_masks: int = 0,
    mask_value: int = 255,
) -> bool:
    """
    Run MedSAM3 on all frames in input_dir/images/ and write masks to input_dir/masks/.

    Returns True on success, False on failure.
    """
    input_path = Path(input_dir)
    images_dir = input_path / "images"
    masks_dir = input_path / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(images_dir.glob("*.png"))
    if not frame_files:
        print(f"[Stage2] ERROR: No frames found in {images_dir}")
        return False

    print(f"[Stage2] Found {len(frame_files)} frames in {images_dir}")
    print("[Stage2] Loading MedSAM3...")

    # sam3 is pip-installed; our wrapper lives in models/medsam3/
    app_dir = Path(__file__).resolve().parent.parent
    medsam3_dir = str(app_dir / "models" / "medsam3")
    if medsam3_dir not in sys.path:
        sys.path.insert(0, medsam3_dir)

    from sam3_inference import SAM3Model

    model = SAM3Model(
        checkpoint_path=checkpoint_path if checkpoint_path else None,
        confidence_threshold=0.1,
        device="cuda",
    )
    model.load_model()
    print("[Stage2] MedSAM3 loaded successfully.")

    t0 = time.time()
    for i, frame_file in enumerate(frame_files):
        frame_bgr = cv2.imread(str(frame_file))
        if frame_bgr is None:
            print(f"[Stage2] Warning: Could not read {frame_file}")
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        inference_state = model.encode_image(frame_rgb)
        mask = model.predict_text_union(
            inference_state,
            text_prompt,
            score_threshold=score_threshold,
            max_masks=max_masks,
        )

        H, W = frame_rgb.shape[:2]
        if mask is not None:
            mask = np.squeeze(mask)
            if mask.shape != (H, W):
                mask = cv2.resize(
                    mask.astype(np.uint8), (W, H),
                    interpolation=cv2.INTER_NEAREST,
                )
            binary_mask = (mask > 0).astype(np.uint8) * mask_value
        else:
            binary_mask = np.zeros((H, W), dtype=np.uint8)

        mask_file = masks_dir / frame_file.name
        cv2.imwrite(str(mask_file), binary_mask)

        if i % 10 == 0 or i == len(frame_files) - 1:
            tool_pct = (binary_mask > 0).sum() / binary_mask.size * 100
            elapsed = time.time() - t0
            fps = (i + 1) / elapsed if elapsed > 0 else 0
            print(
                f"[Stage2] Frame {i+1}/{len(frame_files)}: "
                f"tool coverage {tool_pct:.1f}%, "
                f"{fps:.1f} fps"
            )

    elapsed = time.time() - t0
    print(f"[Stage2] Wrote {len(frame_files)} masks to {masks_dir} ({elapsed:.1f}s)")

    del model
    torch.cuda.empty_cache()
    return True


if __name__ == "__main__":
    parser = ArgumentParser(description="Stage 2: MedSAM3 tool segmentation")
    parser.add_argument(
        "--input-dir", required=True,
        help="Stage 1 output directory (contains images/)",
    )
    parser.add_argument(
        "--sam3-checkpoint", default="",
        help="Path to MedSAM3 checkpoint (.pt)",
    )
    parser.add_argument(
        "--text-prompt", default="surgical tool",
        help="Segmentation text prompt",
    )
    parser.add_argument(
        "--score-threshold", type=float, default=0.3,
        help="Minimum confidence score for mask inclusion",
    )
    parser.add_argument(
        "--max-masks", type=int, default=0,
        help="Maximum number of masks to union (0 = all)",
    )
    args = parser.parse_args()

    success = run_medsam3_segmentation(
        input_dir=args.input_dir,
        checkpoint_path=args.sam3_checkpoint,
        text_prompt=args.text_prompt,
        score_threshold=args.score_threshold,
        max_masks=args.max_masks,
    )
    sys.exit(0 if success else 1)
