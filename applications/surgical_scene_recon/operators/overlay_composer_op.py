# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Overlay Composer Operator.

Creates a 3-panel live preview from source, depth, and mask streams:
  ┌─────────────┬─────────────┬─────────────┐
  │   Source     │    Depth    │    Mask     │
  │   Frame      │  Colormap   │  Overlay    │
  └─────────────┴─────────────┴─────────────┘

Emits a single composite tensor for HoloViz display.
"""

import cupy as cp
import cv2
import numpy as np
from holoscan.core import Operator, OperatorSpec


class OverlayComposerOp(Operator):
    """
    Composes a 3-panel live preview from source, depth, and mask streams.

    Inputs:
        source: dict with "frame" → RGB uint8 [H, W, 3]
        depth: dict with "depth" → float32 [H, W]
        masks: dict with "mask" → uint8 [H, W]

    Outputs:
        out: dict with "composite" → float32 [H, 3*W, 4] RGBA normalized [0, 1]

    Parameters:
        depth_colormap: OpenCV colormap name (default "turbo")
        mask_overlay_alpha: Blending alpha for mask overlay (default 0.4)
    """

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("source")
        spec.input("depth")
        spec.input("masks")
        spec.output("out")
        spec.param("depth_colormap", "turbo")
        spec.param("mask_overlay_alpha", 0.4)

    def compute(self, op_input, op_output, context):
        source_msg = op_input.receive("source")
        depth_msg = op_input.receive("depth")
        mask_msg = op_input.receive("masks")

        if source_msg is None or depth_msg is None or mask_msg is None:
            return

        source = cp.asnumpy(cp.asarray(source_msg["frame"]))
        depth = cp.asnumpy(cp.asarray(depth_msg["depth"]))
        mask = cp.asnumpy(cp.asarray(mask_msg["mask"]))

        if source.ndim == 4:
            source = source[0]

        H, W = source.shape[:2]

        # Resize depth/mask to match source if needed
        if depth.shape[:2] != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
        if mask.shape[:2] != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        # Panel 1: source frame (passthrough)
        panel1 = source

        # Panel 2: depth colormap
        d_min, d_max = float(depth.min()), float(depth.max())
        if d_max > d_min:
            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            depth_norm = np.zeros((H, W), dtype=np.uint8)

        cmap_attr = f"COLORMAP_{self.depth_colormap.upper()}"
        cmap_id = getattr(cv2, cmap_attr, cv2.COLORMAP_TURBO)
        depth_bgr = cv2.applyColorMap(depth_norm, cmap_id)
        panel2 = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB)

        # Panel 3: source with semi-transparent mask overlay (red for tools)
        panel3 = source.copy()
        tool_region = mask > 127
        if np.any(tool_region):
            alpha = float(self.mask_overlay_alpha)
            overlay_color = np.array([255, 50, 50], dtype=np.float32)
            for c in range(3):
                panel3[tool_region, c] = np.clip(
                    (1.0 - alpha) * panel3[tool_region, c] + alpha * overlay_color[c],
                    0,
                    255,
                ).astype(np.uint8)

        # Concatenate horizontally: [H, 3*W, 3]
        composite = np.concatenate([panel1, panel2, panel3], axis=1)

        # Convert to float32 RGBA [0, 1] for HoloViz
        rgba = np.ones((*composite.shape[:2], 4), dtype=np.float32)
        rgba[..., :3] = composite.astype(np.float32) / 255.0

        op_output.emit({"composite": cp.asarray(rgba)}, "out")
