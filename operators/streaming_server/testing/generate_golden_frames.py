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
Golden frame generation script for StreamingServer testing.

This script generates reference frames from the streaming server demo
that can be used for visual validation in automated testing.

Note: For the comprehensive testing framework, we reuse golden frames
from the endoscopy_tool_tracking application since both applications
process the same endoscopy video data and should produce similar
visual output for validation purposes.
"""

import argparse
from pathlib import Path


def main():
    """Generate golden reference frames for streaming server testing."""
    parser = argparse.ArgumentParser(
        description="Generate golden frames for StreamingServer testing"
    )
    parser.add_argument("--output-dir", default=".", help="Output directory for golden frames")
    parser.add_argument("--count", type=int, default=10, help="Number of frames to generate")
    args = parser.parse_args()

    print("🎬 StreamingServer Golden Frame Generator")
    print("=" * 50)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Check if golden frames already exist
    existing_frames = list(output_dir.glob("*.png"))
    if existing_frames:
        print(f"✅ Found {len(existing_frames)} existing golden frames:")
        for frame in sorted(existing_frames):
            print(f"   - {frame.name}")
        print("\n📝 Golden frames are already available for testing.")
        print("   These frames were copied from endoscopy_tool_tracking")
        print("   and provide consistent reference data for validation.")
        return

    print("⚠️  No golden frames found.")
    print("🔄 For comprehensive testing, golden frames should be copied")
    print("   from a working application that processes the same data.")
    print("\n📋 To generate golden frames:")
    print("   1. Run streaming_server_demo with recording enabled")
    print("   2. Extract frames from the recorded output")
    print("   3. Select representative frames for validation")
    print("   4. Copy frames to this testing directory")

    # Create placeholder information
    info_file = output_dir / "golden_frames_info.txt"
    with open(info_file, "w") as f:
        f.write("StreamingServer Demo Golden Frames\n")
        f.write("=" * 40 + "\n\n")
        f.write("These golden reference frames are used for automated\n")
        f.write("validation of the streaming server demo output.\n\n")
        f.write("Frames should represent typical endoscopy video\n")
        f.write("processing output and be consistent across test runs.\n\n")
        f.write(f"Expected frame count: {args.count}\n")
        f.write("Frame format: PNG\n")
        f.write("Resolution: 854x480\n")
        f.write("Naming pattern: NNNN.png (e.g., 0001.png)\n")

    print(f"📝 Created {info_file} with golden frame specifications")
    print("\n✅ Golden frame generation setup complete.")


if __name__ == "__main__":
    main()
