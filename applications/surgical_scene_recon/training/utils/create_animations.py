# MIT License
#
# Copyright (c) 2025-2026, EndoGaussian Project
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Create GIF and MP4 Animations from Rendered Frames
===================================================

Creates animated GIF and MP4 video from comparison frames in renders_all_train directory.

Usage:
    python utils/create_animations.py --input output/tissue_only/renders_all_train
    python utils/create_animations.py --input output/full_scene/renders_all_train --fps 10
    python utils/create_animations.py --input output/tissue_only/renders_all_train --skip-gif
"""

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List

import imageio
from PIL import Image
from tqdm import tqdm


def natural_sort_key(s: str) -> List:
    """
    Sort strings naturally (handles numbers correctly).

    Example: ['frame_1.png', 'frame_2.png', 'frame_10.png']
    instead of: ['frame_1.png', 'frame_10.png', 'frame_2.png']
    """
    import re

    return [int(c) if c.isdigit() else c.lower() for c in re.split("([0-9]+)", s)]


def create_gif(image_paths: List[str], output_path: str, fps: int = 10, loop: int = 0):
    """
    Create an animated GIF from a list of image paths.

    Args:
        image_paths: List of paths to input images
        output_path: Path to save the output GIF
        fps: Frames per second (default: 10)
        loop: Number of loops (0 = infinite, default: 0)
    """
    print(f"\n{'='*70}")
    print("Creating GIF Animation")
    print(f"{'='*70}\n")
    print(f"  Input: {len(image_paths)} frames")
    print(f"  Output: {output_path}")
    print(f"  FPS: {fps}")
    print(f"  Loop: {'infinite' if loop == 0 else loop}")

    # Read images using PIL for better control
    pil_images = []
    for img_path in tqdm(image_paths, desc="Loading frames"):
        img = Image.open(img_path)
        pil_images.append(img)

    # Calculate duration per frame (in milliseconds for PIL)
    duration_ms = int(1000 / fps)

    print(f"  Duration per frame: {duration_ms} ms")

    # Save as GIF using PIL (better FPS control)
    print("\nSaving GIF...")
    pil_images[0].save(
        output_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration_ms,
        loop=loop,
        optimize=False,  # Don't optimize for faster saving
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ GIF created: {output_path} ({file_size_mb:.2f} MB)")
    print(f"  Actual FPS: {fps} ({duration_ms}ms per frame)")


def create_mp4(image_paths: List[str], output_path: str, fps: int = 10, quality: int = 8):
    """
    Create an MP4 video from a list of image paths.

    Args:
        image_paths: List of paths to input images
        output_path: Path to save the output MP4
        fps: Frames per second (default: 10)
        quality: Quality setting (1-10, 10=best, default: 8)
    """
    print(f"\n{'='*70}")
    print("Creating MP4 Video")
    print(f"{'='*70}\n")
    print(f"  Input: {len(image_paths)} frames")
    print(f"  Output: {output_path}")
    print(f"  FPS: {fps}")
    print(f"  Quality: {quality}/10")

    # Read images
    images = []
    for img_path in tqdm(image_paths, desc="Loading frames"):
        img = imageio.imread(img_path)
        images.append(img)

    # Save as MP4 using imageio-ffmpeg
    print("\nEncoding MP4...")
    writer = imageio.get_writer(
        output_path, fps=fps, quality=quality, codec="libx264", pixelformat="yuv420p"
    )

    for img in tqdm(images, desc="Writing frames"):
        writer.append_data(img)

    writer.close()

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ MP4 created: {output_path} ({file_size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Create GIF and MP4 animations from rendered frames"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input directory containing rendered frames (e.g., output/tissue_only/renders_all_train)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: same as input directory)",
    )
    parser.add_argument(
        "--pattern",
        "-p",
        type=str,
        default="*comparison*.png",
        help="Filename pattern to match (default: *comparison*.png)",
    )
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
    parser.add_argument(
        "--gif-name",
        type=str,
        default="animation.gif",
        help="Output GIF filename (default: animation.gif)",
    )
    parser.add_argument(
        "--mp4-name",
        type=str,
        default="animation.mp4",
        help="Output MP4 filename (default: animation.mp4)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=8,
        choices=range(1, 11),
        help="MP4 quality (1-10, 10=best, default: 8)",
    )
    parser.add_argument(
        "--skip-gif", action="store_true", help="Skip GIF creation (only create MP4)"
    )
    parser.add_argument(
        "--skip-mp4", action="store_true", help="Skip MP4 creation (only create GIF)"
    )
    parser.add_argument(
        "--loop", type=int, default=0, help="GIF loop count (0=infinite, default: 0)"
    )

    args = parser.parse_args()

    # Validate input directory
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Set output directory
    output_dir = Path(args.output) if args.output else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all matching images
    pattern = os.path.join(input_dir, args.pattern)
    image_paths = sorted(glob.glob(pattern), key=natural_sort_key)

    if len(image_paths) == 0:
        print(f"Error: No images found matching pattern: {pattern}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print("Animation Creator")
    print(f"{'='*70}\n")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(image_paths)} frames")
    print(f"Pattern: {args.pattern}")

    # Create GIF
    if not args.skip_gif:
        gif_path = output_dir / args.gif_name
        create_gif(image_paths, str(gif_path), fps=args.fps, loop=args.loop)

    # Create MP4
    if not args.skip_mp4:
        mp4_path = output_dir / args.mp4_name
        create_mp4(image_paths, str(mp4_path), fps=args.fps, quality=args.quality)

    print(f"\n{'='*70}")
    print("✓ Animation creation complete!")
    print(f"{'='*70}\n")
    print(f"Output files saved to: {output_dir}")
    if not args.skip_gif:
        print(f"  - {args.gif_name}")
    if not args.skip_mp4:
        print(f"  - {args.mp4_name}")
    print()


if __name__ == "__main__":
    main()
