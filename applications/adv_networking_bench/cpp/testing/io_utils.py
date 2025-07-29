#!/usr/bin/env python3
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

import logging
import os

# Configure the logger
logger = logging.getLogger(__name__)

def generate_smpte2110_file(file_path: str, width: int = 1920, height: int = 1080, bit_depth: int = 10, num_frames: int = 10):
    """
    Generate a zero-filled YCbCr 4:2:2 media file with specified parameters.
    
    Args:
        file_path: Output file path
        width: Video width in pixels (default: 1920 for 1080p)
        height: Video height in pixels (default: 1080 for 1080p)  
        bit_depth: Bit depth in bits - must be 8, 10, or 12 (default: 10)
        num_frames: Number of frames to generate (default: 10)
    """
    # Validate input parameters
    if bit_depth not in [8, 10, 12]:
        raise ValueError(f"Unsupported bit depth: {bit_depth}. Supported values: 8, 10, 12")
    
    if width <= 0 or height <= 0 or num_frames <= 0:
        raise ValueError("Width, height, and num_frames must be positive integers")
    
    logger.info(f"Generating YCbCr 4:2:2 media file: {file_path} ({width}x{height}, {bit_depth}-bit, {num_frames} frames)")
    
    # SMPTE 2110-20 specification: YCbCr 4:2:2 pgroup sizes
    # Each pgroup covers 2 pixels and contains Y0, Cb, Y1, Cr samples
    SMPTE_2110_PGROUP_SIZES = {
        8:  4,  # 4 bytes per 2 pixels = 2.0 bytes/pixel
        10: 5,  # 5 bytes per 2 pixels = 2.5 bytes/pixel  
        12: 6,  # 6 bytes per 2 pixels = 3.0 bytes/pixel
    }

    # Calculate total file size
    pixels_per_frame = width * height
    pgroups_per_frame = pixels_per_frame // 2  # Each pgroup covers 2 pixels
    bytes_per_frame = pgroups_per_frame * SMPTE_2110_PGROUP_SIZES[bit_depth]
    total_size = int(bytes_per_frame * num_frames)
    
    logger.info(f"Frame size: {bytes_per_frame} bytes, Total file size: {total_size} bytes ({total_size / (1024*1024):.1f} MB)")
    
    # Create the directory if it doesn't exist
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory: {e}")
        raise
    
    # Generate the zero-filled file
    try:
        with open(file_path, 'wb') as f:
            f.write(b'\x00' * total_size)
    except OSError as e:
        logger.error(f"Failed to write file {file_path}: {e}")
        raise
    
    logger.info(f"Successfully generated YCbCr 4:2:2 media file: {file_path}")

