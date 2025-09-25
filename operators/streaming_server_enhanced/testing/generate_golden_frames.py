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

"""
Generate golden reference frames for StreamingServer visual regression testing.

This script creates synthetic reference frames with various patterns for testing
the StreamingServer operators' frame processing capabilities.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    print("PIL (Pillow) is required. Install with: pip install Pillow")
    sys.exit(1)


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
                square_x = x // square_size
                square_y = y // square_size
                is_white = (square_x + square_y + frame_number) % 2 == 0
                color_value = 255 if is_white else 0
                frame[y, x] = [color_value, color_value, color_value]

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
        # Create pattern with text overlay
        frame = np.full((height, width, 3), 128, dtype=np.uint8)  # Gray background

        # Add simple text pattern (without font rendering)
        text_y = height // 2
        text_start_x = width // 4
        text_width = width // 2
        text_height = 20

        # Create simple text rectangle
        frame[text_y : text_y + text_height, text_start_x : text_start_x + text_width] = [
            255,
            255,
            255,
        ]

        # Add frame number as pattern in corners
        corner_size = 50
        color_intensity = (frame_number * 50) % 256
        frame[:corner_size, :corner_size] = [color_intensity, 0, 0]  # Top-left
        frame[-corner_size:, -corner_size:] = [0, color_intensity, 0]  # Bottom-right

    elif pattern_type == "noise":
        # Create noise pattern (deterministic based on frame number)
        np.random.seed(frame_number)
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    elif pattern_type == "solid":
        # Create solid color based on frame number
        colors = [
            [255, 0, 0],  # Red
            [0, 255, 0],  # Green
            [0, 0, 255],  # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [255, 128, 0],  # Orange
            [128, 0, 255],  # Purple
            [255, 255, 255],  # White
            [128, 128, 128],  # Gray
        ]
        color = colors[frame_number % len(colors)]
        frame = np.full((height, width, 3), color, dtype=np.uint8)

    elif pattern_type == "border":
        # Create frame with colored border
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        border_width = 20
        border_color = [
            (frame_number * 30) % 256,
            (frame_number * 50) % 256,
            (frame_number * 70) % 256,
        ]

        # Top and bottom borders
        frame[:border_width, :] = border_color
        frame[-border_width:, :] = border_color

        # Left and right borders
        frame[:, :border_width] = border_color
        frame[:, -border_width:] = border_color

        # Fill center with complementary color
        center_color = [255 - c for c in border_color]
        frame[border_width:-border_width, border_width:-border_width] = center_color

    else:  # default pattern
        frame = np.zeros((height, width, 3), dtype=np.uint8)

    return frame


def generate_golden_frames(output_dir, width=854, height=480, count=10, patterns=None):
    """Generate golden reference frames with various patterns."""

    if patterns is None:
        patterns = ["gradient", "checkerboard", "circles", "text", "noise", "solid", "border"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating {count} golden frames in {output_path}")
    print(f"Resolution: {width}x{height}")
    print(f"Patterns: {', '.join(patterns)}")

    generated_count = 0

    for i in range(count):
        frame_number = i + 1
        pattern = patterns[i % len(patterns)]

        # Generate frame
        frame_data = create_test_pattern(width, height, pattern, frame_number)

        # Convert to PIL Image (RGB format for saving)
        # Note: OpenCV uses BGR, but PIL uses RGB, so we need to convert
        frame_rgb = frame_data[:, :, ::-1]  # BGR to RGB
        image = Image.fromarray(frame_rgb, "RGB")

        # Save frame
        filename = f"{frame_number:04d}.png"
        filepath = output_path / filename
        image.save(filepath)

        print(f"  Generated: {filename} ({pattern} pattern)")
        generated_count += 1

    print(f"\n✅ Successfully generated {generated_count} golden frames")

    # Generate metadata file
    metadata_file = output_path / "metadata.txt"
    with open(metadata_file, "w") as f:
        f.write("Golden Frames Metadata\n")
        f.write("======================\n")
        f.write(f"Generated frames: {generated_count}\n")
        f.write(f"Resolution: {width}x{height}\n")
        f.write(f"Patterns used: {', '.join(patterns)}\n")
        f.write("Frame format: PNG (RGB)\n")
        f.write("Usage: Visual regression testing for StreamingServer\n")
        f.write("\nFrame Details:\n")

        for i in range(generated_count):
            frame_number = i + 1
            pattern = patterns[i % len(patterns)]
            f.write(f"  {frame_number:04d}.png - {pattern} pattern\n")

    print(f"📝 Metadata saved to: {metadata_file}")
    return generated_count


def verify_golden_frames(frames_dir):
    """Verify that golden frames exist and are readable."""
    frames_path = Path(frames_dir)

    if not frames_path.exists():
        print(f"❌ Golden frames directory not found: {frames_path}")
        return False

    png_files = list(frames_path.glob("*.png"))

    if not png_files:
        print(f"❌ No PNG files found in: {frames_path}")
        return False

    print(f"🔍 Verifying {len(png_files)} golden frames...")

    valid_count = 0
    for png_file in sorted(png_files):
        try:
            with Image.open(png_file) as img:
                width, height = img.size
                print(f"  ✅ {png_file.name}: {width}x{height} {img.mode}")
                valid_count += 1
        except Exception as e:
            print(f"  ❌ {png_file.name}: Error - {e}")

    print(f"\n📊 Verification complete: {valid_count}/{len(png_files)} frames valid")
    return valid_count == len(png_files)


def main():
    """Main function for golden frame generation."""
    parser = argparse.ArgumentParser(
        description="Generate golden reference frames for StreamingServer testing"
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        default="golden_frames",
        help="Output directory for golden frames (default: golden_frames)",
    )

    parser.add_argument("--width", "-w", type=int, default=854, help="Frame width (default: 854)")

    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")

    parser.add_argument(
        "--count", "-c", type=int, default=10, help="Number of frames to generate (default: 10)"
    )

    parser.add_argument(
        "--patterns",
        "-p",
        nargs="+",
        choices=["gradient", "checkerboard", "circles", "text", "noise", "solid", "border"],
        help="Patterns to use (default: all patterns)",
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing golden frames instead of generating new ones",
    )

    args = parser.parse_args()

    if args.verify:
        # Verify existing frames
        success = verify_golden_frames(args.output_dir)
        sys.exit(0 if success else 1)
    else:
        # Generate new frames
        try:
            count = generate_golden_frames(
                args.output_dir, args.width, args.height, args.count, args.patterns
            )

            print("\n🎉 Golden frame generation completed successfully!")
            print(f"📁 Location: {Path(args.output_dir).absolute()}")
            print(f"🖼️  Frames: {count}")

            # Verify the generated frames
            print("\n🔍 Verifying generated frames...")
            verify_golden_frames(args.output_dir)

        except Exception as e:
            print(f"❌ Error generating golden frames: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
