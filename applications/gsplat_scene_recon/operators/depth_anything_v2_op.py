# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Depth Anything V2 Operator for Holoscan.

Wraps the DA2 ViT-S/B/L model to produce per-frame dense depth maps.
The model runs in PyTorch; the operator handles CuPy↔NumPy conversion
and isolates torch CUDA work on a dedicated stream.
"""

import sys

import cupy as cp
import cv2
import numpy as np
from holoscan.core import Operator, OperatorSpec

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


class DepthAnythingV2Op(Operator):
    """
    Runs Depth Anything V2 monocular depth estimation on each video frame.

    Inputs:
        frame_in: dict with "frame" key → RGB uint8 CuPy tensor [H, W, 3]

    Outputs:
        depth_out: dict with "depth" key → float32 CuPy tensor [H, W]

    Parameters:
        da2_root: Directory containing the depth_anything_v2 package. If empty,
                  auto-resolves to ``models/depth_anything_v2/`` relative
                  to the application directory.
        checkpoint: Path to .pth model weights
        encoder: Model variant — "vits", "vitb", or "vitl"
        max_depth: Maximum depth value for the metric head
        input_size: Internal processing resolution (default 518, must be multiple of 14)
    """

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self._model = None
        self._torch_stream = None
        self._frame_count = 0

    def setup(self, spec: OperatorSpec):
        spec.input("frame_in")
        spec.output("depth_out")
        spec.param("da2_root", "")
        spec.param("checkpoint", "")
        spec.param("encoder", "vits")
        spec.param("max_depth", 20.0)
        spec.param("input_size", 518)

    def _resolve_da2_root(self) -> str:
        from pathlib import Path

        if self.da2_root:
            return self.da2_root
        app_dir = Path(__file__).resolve().parent.parent
        bundled = app_dir / "models" / "depth_anything_v2"
        if bundled.is_dir():
            return str(bundled)
        return ""

    def start(self):
        import torch

        print(f"[DA2] Loading model: encoder={self.encoder}, max_depth={self.max_depth}")

        da2_root = self._resolve_da2_root()
        if da2_root and da2_root not in sys.path:
            sys.path.insert(0, da2_root)

        from depth_anything_v2.dpt import DepthAnythingV2

        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"Unknown encoder '{self.encoder}'. Use: {list(MODEL_CONFIGS)}")

        config = {**MODEL_CONFIGS[self.encoder], "max_depth": self.max_depth}
        self._model = DepthAnythingV2(**config)

        ckpt = torch.load(self.checkpoint, map_location="cuda", weights_only=False)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        self._model.load_state_dict(ckpt, strict=True)
        self._model.to("cuda").eval()

        self._torch_stream = torch.cuda.Stream()
        print("[DA2] Model loaded successfully.")

    def compute(self, op_input, op_output, context):
        import torch

        msg = op_input.receive("frame_in")
        if msg is None:
            return

        frame_cp = cp.asarray(msg["frame"])
        frame_rgb = cp.asnumpy(frame_cp)
        if frame_rgb.ndim == 4:
            frame_rgb = frame_rgb[0]

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        with torch.cuda.stream(self._torch_stream):
            with torch.no_grad():
                depth = self._model.infer_image(frame_bgr, input_size=self.input_size)
        self._torch_stream.synchronize()

        depth = np.ascontiguousarray(depth, dtype=np.float32)

        op_output.emit({"depth": cp.asarray(depth)}, "depth_out")

        if self._frame_count % 10 == 0:
            print(
                f"[DA2] Frame {self._frame_count}: "
                f"depth [{depth.min():.2f}, {depth.max():.2f}]"
            )
        self._frame_count += 1

    def stop(self):
        import torch

        print(f"[DA2] Stopped after {self._frame_count} frames.")
        self._model = None
        torch.cuda.empty_cache()
