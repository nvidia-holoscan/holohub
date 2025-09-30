#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Script to generate golden reference frames for streaming_client_enhanced testing.
This creates synthetic golden frames for regression testing without requiring real video data.
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image


def create_test_pattern(width, height, pattern_type="gradient", frame_number=1):
    """Create a test pattern for golden frame generation."""

    if pattern_type == "gradient":
        # Create a gradient pattern with frame number influence
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                # Add frame number influence to create unique patterns
                offset = (frame_number * 10) % 256
                frame[y, x, 0] = min(255, ((x + offset) * 255) // width)  # Red channel
                frame[y, x, 1] = min(255, ((y + offset) * 255) // height)  # Green channel
                frame[y, x, 2] = min(
                    255, (((x + y) + offset) * 255) // (width + height)
                )  # Blue channel

    elif pattern_type == "checkerboard":
        # Create checkerboard pattern
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        square_size = 40
        for y in range(height):
            for x in range(width):
                if ((x // square_size) + (y // square_size) + frame_number) % 2:
                    frame[y, x] = [255, 255, 255]  # White
                else:
                    frame[y, x] = [0, 0, 0]  # Black

    elif pattern_type == "circles":
        # Create concentric circles pattern
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        center_x, center_y = width // 2, height // 2

        for y in range(height):
            for x in range(width):
                dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                # Create rings with frame number influence
                ring_value = min(255, int((dist + frame_number * 5) % 100) * 2)
                frame[y, x] = [ring_value, ring_value, max(0, 255 - ring_value)]

    elif pattern_type == "text":
        # Create text-based pattern for frame identification
        frame = np.full((height, width, 3), 64, dtype=np.uint8)  # Dark gray background

        # Add frame number as text pattern (simplified)
        text_area_h = height // 4
        text_area_w = width // 4
        start_y = (height - text_area_h) // 2
        start_x = (width - text_area_w) // 2

        # Create simple digit pattern for frame number
        digit_patterns = {
            1: [[0, 1, 0], [1, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 1]],
            2: [[1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]],
            3: [[1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
            4: [[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]],
            5: [[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
            6: [[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1]],
            7: [[1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
            8: [[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1]],
            9: [[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
            0: [[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]],
        }

        if frame_number <= 10:
            digit = frame_number % 10
            pattern = digit_patterns.get(digit, digit_patterns[0])

            for row_idx, row in enumerate(pattern):
                for col_idx, val in enumerate(row):
                    if val:
                        # Draw pixel blocks for visibility
                        for dy in range(8):
                            for dx in range(8):
                                py = start_y + row_idx * 8 + dy
                                px = start_x + col_idx * 8 + dx
                                if 0 <= py < height and 0 <= px < width:
                                    frame[py, px] = [255, 255, 255]  # White text

    return frame


def generate_golden_frames(output_dir, width=854, height=480, num_frames=10):
    """Generate a set of golden reference frames."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    patterns = ["gradient", "checkerboard", "circles", "text"]

    print(f"Generating {num_frames} golden reference frames...")

    for i in range(1, num_frames + 1):
        # Cycle through different patterns
        pattern_type = patterns[(i - 1) % len(patterns)]

        # Create the test pattern
        frame_data = create_test_pattern(width, height, pattern_type, i)

        # Convert to PIL Image and save
        img = Image.fromarray(frame_data, "RGB")
        filename = f"{i:04d}.png"
        filepath = output_path / filename
        img.save(filepath)

        print(f"  Generated {filename} ({pattern_type} pattern)")

    print(f"\nGolden frames saved to: {output_path}")
    return output_path


def create_test_config(output_dir):
    """Create a test configuration file for streaming client testing."""

    config_content = """# Golden Frame Testing Configuration for StreamingClientOp Enhanced
# This configuration is used for regression testing with golden reference frames

# Video settings (matching golden frame generation)
width: 854
height: 480
fps: 30

# Network settings (test configuration)
server_ip: "127.0.0.1"
signaling_port: 48010

# Frame handling for testing
receive_frames: false  # Disable for unit testing
send_frames: false     # Disable for unit testing

# Validation settings
min_non_zero_bytes: 100

# Test-specific settings
golden_frame_dir: "./golden_frames"
tolerance: 0.05  # 5% tolerance for frame comparison
max_test_frames: 10
"""

    config_path = Path(output_dir) / "golden_frame_test_config.yaml"
    with open(config_path, "w") as f:
        f.write(config_content)

    print(f"Test configuration saved to: {config_path}")
    return config_path


def main():
    parser = ArgumentParser(
        description="Generate golden reference frames for streaming_client_enhanced testing"
    )
    parser.add_argument(
        "--output-dir",
        default="./golden_frames",
        help="Output directory for golden reference frames",
    )
    parser.add_argument("--frames", type=int, default=10, help="Number of frames to generate")
    parser.add_argument("--width", type=int, default=854, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--config", action="store_true", help="Also generate test configuration")

    args = parser.parse_args()

    # Generate golden frames
    output_path = generate_golden_frames(args.output_dir, args.width, args.height, args.frames)

    # Generate test configuration if requested
    if args.config:
        create_test_config(args.output_dir)

    print("\n✅ Golden frame generation complete!")
    print(f"📁 Location: {output_path}")
    print(f"📊 Generated {args.frames} reference frames")
    print("🎯 Use these frames for visual regression testing")


if __name__ == "__main__":
    main()
