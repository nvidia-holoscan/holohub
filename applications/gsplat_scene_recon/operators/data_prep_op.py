# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data Preparation Operator for GSplat Scene Reconstruction (Phase 1).

Receives frames, depth maps, and tool masks from upstream inference operators
and writes them to disk in partial EndoNeRF format. Each frame is written
immediately as it arrives — no buffering.

Output structure:
    <output_dir>/
        images/    *.png     — RGB frames
        depth_raw/ *.npy     — float32 dense depth maps from DA2
        masks/     *.png     — uint8 binary tool masks from MedSAM3

Note: poses_bounds.npy is NOT written here. That requires VGGT (Phase 2)
and format conversion (Phase 3) to complete the EndoNeRF dataset.
"""

from pathlib import Path

import cupy as cp
import cv2
import numpy as np
from holoscan.core import Operator, OperatorSpec


class DataPrepOp(Operator):
    """
    Writes per-frame images, depth maps, and masks to disk.

    Inputs:
        frame_in: dict with "frame" → RGB uint8 [H, W, 3], "frame_idx" → int32 [1]
        depth_in: dict with "depth" → float32 [H, W]
        mask_in: dict with "mask" → uint8 [H, W]

    Parameters:
        output_dir: Base directory for the partial EndoNeRF dataset
    """

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self._frame_count = 0

    def setup(self, spec: OperatorSpec):
        spec.input("frame_in")
        spec.input("depth_in")
        spec.input("mask_in")
        spec.param("output_dir", "output/phase1_data")

    def initialize(self):
        out = Path(self.output_dir)
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "depth_raw").mkdir(parents=True, exist_ok=True)
        (out / "masks").mkdir(parents=True, exist_ok=True)
        print(f"[DataPrep] Output directory: {out}")

    def compute(self, op_input, op_output, context):
        frame_msg = op_input.receive("frame_in")
        depth_msg = op_input.receive("depth_in")
        mask_msg = op_input.receive("mask_in")

        if frame_msg is None or depth_msg is None or mask_msg is None:
            return

        frame = cp.asnumpy(cp.asarray(frame_msg["frame"]))
        depth = cp.asnumpy(cp.asarray(depth_msg["depth"]))
        mask = cp.asnumpy(cp.asarray(mask_msg["mask"]))

        if frame.ndim == 4:
            frame = frame[0]

        idx = self._frame_count
        out = Path(self.output_dir)

        # Save frame as BGR PNG
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out / "images" / f"{idx:06d}.png"), frame_bgr)

        # Save depth as float32 numpy array
        np.save(str(out / "depth_raw" / f"{idx:06d}.npy"), depth.astype(np.float32))

        # Save mask as uint8 PNG
        cv2.imwrite(str(out / "masks" / f"{idx:06d}.png"), mask)

        if idx % 10 == 0:
            tool_pct = (mask > 0).sum() / max(mask.size, 1) * 100
            print(
                f"[DataPrep] Frame {idx}: saved image + depth "
                f"[{depth.min():.1f}, {depth.max():.1f}] + mask ({tool_pct:.0f}% tool)"
            )

        self._frame_count += 1

    def stop(self):
        print(f"[DataPrep] Wrote {self._frame_count} frames to {self.output_dir}")
        print(f"[DataPrep]   images/    — {self._frame_count} PNG files")
        print(f"[DataPrep]   depth_raw/ — {self._frame_count} .npy files")
        print(f"[DataPrep]   masks/     — {self._frame_count} PNG files")
