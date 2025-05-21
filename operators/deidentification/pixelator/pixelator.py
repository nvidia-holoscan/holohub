# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cupy as cp
from holoscan.core import Operator, OperatorSpec

__all__ = ["PixelatorOp"]


class PixelatorOp(Operator):
    """
    This operator is used to deidentify the input image by pixelation.

    Args:
        tensor_name (str): The name of the tensor to be pixelated.
        block_size_h (int): The height of the pixelation block. Defaults to 16.
        block_size_w (int): The width of the pixelation block. Defaults to 16.
    """

    def __init__(
        self,
        *args,
        tensor_name: str,
        block_size_h: int = 16,
        block_size_w: int = 16,
        **kwargs,
    ):
        if block_size_h <= 0:
            raise ValueError("block_size_h must be a positive integer")
        if block_size_w <= 0:
            raise ValueError("block_size_w must be a positive integer")
        self.block_size_h = block_size_h
        self.block_size_w = block_size_w
        self.tensor_name = tensor_name
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("in")
        if self.tensor_name not in in_message:
            raise KeyError(f"Tensor '{self.tensor_name}' not found in input message")
        image = cp.asarray(in_message[self.tensor_name])
        h, w = image.shape[:2]

        # Pad image to make dimensions divisible by block size
        pad_h = (self.block_size_h - h % self.block_size_h) % self.block_size_h
        pad_w = (self.block_size_w - w % self.block_size_w) % self.block_size_w
        if pad_h > 0 or pad_w > 0:
            pad_width = ((0, pad_h), (0, pad_w)) + ((0, 0),) * (image.ndim - 2)
            image_padded = cp.pad(image, pad_width, mode="constant", constant_values=0)
        else:
            image_padded = image

        H, W = image_padded.shape[:2]

        # Reshape and mean across blocks
        reshaped = image_padded.reshape(
            H // self.block_size_h, self.block_size_h, W // self.block_size_w, self.block_size_w, -1
        )
        pixel_blocks = cp.mean(reshaped, axis=(1, 3))

        # Repeat each block to create pixelated image of original size
        pixelated = cp.repeat(
            cp.repeat(pixel_blocks, self.block_size_h, axis=0), self.block_size_w, axis=1
        )

        # Crop to original size if padded
        image = pixelated[:h, :w, ...].astype(cp.uint8)

        # Create and emit the result
        out_message = in_message.copy()
        out_message[self.tensor_name] = image
        op_output.emit(out_message, "out")
