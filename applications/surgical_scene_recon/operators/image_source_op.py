# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Image Directory Source Operator.

Reads PNG images from a directory and emits them as RGB CuPy tensors,
one frame per compute() tick. Designed as a lightweight alternative to
VideoStreamReplayerOp for directories of individual frames.
"""

import glob
import os

import cupy as cp
import cv2
from holoscan.core import Operator, OperatorSpec


class ImageDirectorySourceOp(Operator):
    """
    Reads images from a directory and emits one frame per tick.

    Outputs:
        frame_out: dict with
            "frame": RGB uint8 CuPy tensor [H, W, 3]
            "frame_idx": int32 CuPy tensor [1]

    Parameters:
        directory: Path to directory containing image files
        pattern: Glob pattern for images (default: "*.png")
    """

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self._files = []
        self._idx = 0

    def setup(self, spec: OperatorSpec):
        spec.output("frame_out")
        spec.param("directory", "")
        spec.param("pattern", "*.png")

    def initialize(self):
        self._files = sorted(glob.glob(os.path.join(self.directory, self.pattern)))
        self._idx = 0
        if not self._files:
            raise ValueError(f"No images matching '{self.pattern}' found in {self.directory}")
        sample = cv2.imread(self._files[0])
        h, w = sample.shape[:2]
        print(f"[ImageSource] {len(self._files)} images ({w}x{h}) in {self.directory}")
        # Note: frames may have different resolutions; DataPrepOp normalizes to first frame's size

    def compute(self, op_input, op_output, context):
        if self._idx >= len(self._files):
            return

        frame_bgr = cv2.imread(self._files[self._idx])
        if frame_bgr is None:
            print(f"[ImageSource] Warning: cannot read {self._files[self._idx]}")
            self._idx += 1
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        op_output.emit(
            {
                "frame": cp.asarray(frame_rgb),
                "frame_idx": cp.array([self._idx], dtype=cp.int32),
            },
            "frame_out",
        )

        if self._idx % 10 == 0 or self._idx == len(self._files) - 1:
            print(f"[ImageSource] Frame {self._idx}/{len(self._files)}")
        self._idx += 1

    def stop(self):
        print(f"[ImageSource] Emitted {self._idx} frames.")

    @property
    def frame_count(self):
        return len(self._files)
