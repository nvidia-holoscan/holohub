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

"""
Image Saver Operator

Saves rendered images to disk for inspection and validation.
Useful for debugging and comparing rendered output with ground truth.
"""

import os
from pathlib import Path

import cupy as cp
import numpy as np
from holoscan.core import Operator, OperatorSpec
from PIL import Image


class ImageSaverOp(Operator):
    """
    Save rendered images to disk.

    Receives rendered RGB images (as tensors) and saves them as PNG files.
    Automatically creates output directory and numbers files sequentially.

    Useful for:
    - Debugging rendering pipeline
    - Visual inspection of output quality
    - Comparing with ground truth images
    - Creating test datasets
    """

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.frame_count = 0
        self.output_dir_created = False

    def setup(self, spec: OperatorSpec):
        """Define operator interface."""
        spec.input("input")  # Rendered RGB image

        spec.param("output_dir", "output/rendered_images")
        spec.param("prefix", "frame")
        spec.param("save_every", 1)  # Save every N frames (1 = save all)
        spec.param("verbose", True)

    def start(self):
        """Create output directory."""
        # Create output directory if it doesn't exist
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.output_dir_created = True

        if self.verbose:
            print("[ImageSaver] Initialized")
            print(f"[ImageSaver]   Output directory: {output_path.absolute()}")
            print(f"[ImageSaver]   Saving every {self.save_every} frame(s)")

    def compute(self, op_input, op_output, context):
        """Save one frame."""
        # Receive input
        in_message = op_input.receive("input")

        # Check if we should save this frame
        if self.frame_count % self.save_every != 0:
            self.frame_count += 1
            return

        # Get the image tensor
        # Try different possible tensor names
        tensor = None
        for name in ["rendered_rgb", "input", "image", "rgb"]:
            try:
                tensor = in_message.get(name)
                if tensor is not None:
                    break
            except (KeyError, AttributeError, RuntimeError):
                # KeyError: tensor name not found in message
                # AttributeError: message object doesn't have 'get' method
                # RuntimeError: GXF/Holoscan internal error accessing tensor
                continue

        if tensor is None:
            print("[ImageSaver] WARNING: No image tensor found in message")
            self.frame_count += 1
            return

        # Convert to numpy array
        # Handle both CuPy and NumPy arrays
        try:
            if hasattr(tensor, "__cuda_array_interface__"):
                # CuPy array
                img_array = cp.asnumpy(cp.asarray(tensor))
            else:
                # Already NumPy or can be converted
                img_array = np.asarray(tensor)
        except Exception as e:
            print(f"[ImageSaver] ERROR: Failed to convert tensor to array: {e}")
            self.frame_count += 1
            return

        # Ensure correct shape and type
        if img_array.ndim == 4:
            # Remove batch dimension if present [1, H, W, C] -> [H, W, C]
            img_array = img_array[0]

        if img_array.ndim != 3 or img_array.shape[2] not in [1, 3, 4]:
            print(f"[ImageSaver] WARNING: Unexpected image shape: {img_array.shape}")
            self.frame_count += 1
            return

        # Convert to uint8 if needed
        if img_array.dtype == np.float32 or img_array.dtype == np.float64:
            # Assume values are in [0, 1] range
            img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
        elif img_array.dtype != np.uint8:
            # Try to convert
            img_array = img_array.astype(np.uint8)

        # Handle grayscale
        if img_array.shape[2] == 1:
            img_array = img_array.squeeze(2)

        # Create PIL Image
        if img_array.ndim == 2:
            # Grayscale
            pil_image = Image.fromarray(img_array, mode="L")
        else:
            # RGB or RGBA
            pil_image = Image.fromarray(
                img_array, mode="RGB" if img_array.shape[2] == 3 else "RGBA"
            )

        # Generate filename
        filename = f"{self.prefix}_{self.frame_count:06d}.png"
        filepath = os.path.join(self.output_dir, filename)

        # Save image
        try:
            pil_image.save(filepath)
            if self.verbose and self.frame_count % 10 == 0:
                print(f"[ImageSaver] Saved frame {self.frame_count}: {filename}")
        except Exception as e:
            print(f"[ImageSaver] ERROR: Failed to save image: {e}")

        self.frame_count += 1

    def stop(self):
        """Cleanup and summary."""
        if self.verbose:
            print("[ImageSaver] Stopped")
            print(f"[ImageSaver]   Total frames saved: {self.frame_count}")
            print(f"[ImageSaver]   Output directory: {Path(self.output_dir).absolute()}")
