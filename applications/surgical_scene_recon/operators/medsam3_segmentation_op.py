# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MedSAM3 Segmentation Operator for Holoscan.

Wraps the SAM3 image model to produce per-frame binary tool segmentation masks.
Supports text-prompted inference. The operator loads the model once in start()
and runs inference per frame in compute(). All PyTorch work is isolated on a
dedicated CUDA stream to avoid conflicts with other GPU operators.
"""

import cupy as cp
import cv2
import numpy as np
import torch
from holoscan.core import Operator, OperatorSpec


class MedSAM3SegmentationOp(Operator):
    """
    Holoscan operator that runs MedSAM3 tool segmentation on each video frame.

    Inputs:
        frame_in: dict with "frame" key → RGB uint8 CuPy tensor [H, W, 3]

    Outputs:
        mask_out: dict with "mask" key → uint8 CuPy tensor [H, W] (0=tissue, 255=tool)

    Parameters:
        checkpoint_path: Path to MedSAM3 checkpoint file (.pt)
        text_prompt: Text prompt for segmentation (default: "surgical tool")
        score_threshold: Minimum confidence score for mask inclusion
        max_masks: Maximum number of masks to union (0 = all)
        mask_value: Pixel value for tool regions in output mask (default: 255)
    """

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.sam3_model = None
        self._torch_stream = None
        self._frame_count = 0

    def setup(self, spec: OperatorSpec):
        spec.input("frame_in")
        spec.output("mask_out")

        spec.param("checkpoint_path", "")
        spec.param("text_prompt", "surgical tool")
        spec.param("score_threshold", 0.3)
        spec.param("max_masks", 0)
        spec.param("mask_value", 255)

    def start(self):
        print(f"[MedSAM3] Checkpoint: {self.checkpoint_path}")
        print(f"[MedSAM3] Text prompt: '{self.text_prompt}'")

        from models.medsam3.sam3_inference import SAM3Model

        ckpt = self.checkpoint_path if self.checkpoint_path else None
        self.sam3_model = SAM3Model(
            checkpoint_path=ckpt,
            confidence_threshold=0.1,
            device="cuda",
        )
        self.sam3_model.load_model()
        self._torch_stream = torch.cuda.Stream()
        print("[MedSAM3] Model loaded successfully.")

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("frame_in")
        if msg is None:
            return

        # Extract frame tensor — check common key names
        frame_tensor = None
        for key in ("frame", "source_video"):
            if key in msg:
                frame_tensor = msg[key]
                break
        if frame_tensor is None:
            for key in msg.keys():
                if "video" in key or "frame" in key or "rgb" in key:
                    frame_tensor = msg[key]
                    break

        if frame_tensor is None:
            print("[MedSAM3] Warning: no frame tensor found in input message")
            return

        frame_cp = cp.asarray(frame_tensor)

        if frame_cp.dtype == cp.float32:
            if frame_cp.max() <= 1.0:
                frame_rgb = cp.asnumpy((frame_cp * 255.0).astype(cp.uint8))
            else:
                frame_rgb = cp.asnumpy(frame_cp.astype(cp.uint8))
        else:
            frame_rgb = cp.asnumpy(frame_cp)

        if frame_rgb.ndim == 4:
            frame_rgb = frame_rgb[0]

        with torch.cuda.stream(self._torch_stream):
            inference_state = self.sam3_model.encode_image(frame_rgb)
            raw_mask = self.sam3_model.predict_text_union(
                inference_state,
                self.text_prompt,
                score_threshold=self.score_threshold,
                max_masks=self.max_masks,
            )
        self._torch_stream.synchronize()

        H, W = frame_rgb.shape[:2]
        if raw_mask is not None:
            raw_mask = np.squeeze(raw_mask)
            if raw_mask.shape != (H, W):
                raw_mask = cv2.resize(
                    raw_mask.astype(np.uint8),
                    (W, H),
                    interpolation=cv2.INTER_NEAREST,
                )
            binary_mask = (raw_mask > 0).astype(np.uint8) * int(self.mask_value)
        else:
            binary_mask = np.zeros((H, W), dtype=np.uint8)

        if self._frame_count % 10 == 0:
            tool_pct = (binary_mask > 0).sum() / binary_mask.size * 100
            print(f"[MedSAM3] Frame {self._frame_count}: tool coverage {tool_pct:.1f}%")

        op_output.emit({"mask": cp.asarray(binary_mask)}, "mask_out")
        self._frame_count += 1

    def stop(self):
        print(f"[MedSAM3] Stopped after {self._frame_count} frames.")
        self.sam3_model = None
        torch.cuda.empty_cache()
