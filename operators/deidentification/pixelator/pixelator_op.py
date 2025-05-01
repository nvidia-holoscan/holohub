# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
from holoscan.core import Operator, OperatorSpec

__all__ = ["PixelatorOp"]


class PixelatorOp(Operator):
    """
    This operator is used to deidentify the input image by pixelation.
    """

    def __init__(
        self,
        *args,
        block_size_h: int = 16,
        block_size_w: int = 16,
        tensor_name: str = "",
        **kwargs,
    ):
        self.block_size_h = block_size_h
        self.block_size_w = block_size_w
        self.tensor_name = tensor_name
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("in")
        image = cp.asarray(in_message[self.tensor_name])
        # Pixelate the image by downsampling and upsampling
        h, w = image.shape[:2]
        small_h = h // self.block_size_h
        small_w = w // self.block_size_w
        # Reshape and mean across blocks to downsample
        reshaped = image.reshape(small_h, self.block_size_h, small_w, self.block_size_w, -1)
        downsampled = cp.mean(reshaped, axis=(1, 3))
        # Repeat each pixel to upsample back to original size
        upsampled = cp.repeat(
            cp.repeat(downsampled, self.block_size_h, axis=0), self.block_size_w, axis=1
        )
        image = upsampled.astype(cp.uint8)
        # add the holoviz tensors to the output message
        out_message = in_message.copy()
        out_message[self.tensor_name] = image
        op_output.emit(out_message, "out")
