#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
Test Data Loading - Phase 1.1

Simple test application to verify EndoNeRF data loading.
Loads frames, depths, masks, and camera poses and prints debug info.

Usage:
    python test_data_loading.py --data_dir /path/to/EndoNeRF/pulling
"""

from argparse import ArgumentParser  # noqa: E402

from holoscan.conditions import CountCondition  # noqa: E402
from holoscan.core import Application  # noqa: E402

# Import our custom operators
from operators import DebugPrintOp, EndoNeRFLoaderOp  # noqa: E402


class DataLoadingTestApp(Application):
    """
    Minimal test application for data loading verification.

    Pipeline:
        EndoNeRFLoaderOp → DebugPrintOp

    The loader reads data and the debug operator prints information about it.
    """

    def __init__(self, data_dir, num_frames=20):
        super().__init__()
        self.name = "Data Loading Test"
        self.data_dir = data_dir
        self.num_frames = num_frames

    def compose(self):
        """Build the test pipeline."""
        print(f"\n{'='*60}")
        print("  Data Loading Test Application")
        print(f"{'='*60}")
        print(f"Data directory: {self.data_dir}")
        print(f"Will process {self.num_frames} frames")
        print(f"{'='*60}\n")

        # Create operators
        # Use CountCondition to limit how many times compute() is called
        count_condition = CountCondition(self, count=self.num_frames)

        loader = EndoNeRFLoaderOp(
            self,
            name="loader",
            data_dir=self.data_dir,
            loop=False,  # Don't loop for testing
            max_frames=self.num_frames,  # Limit number of frames
            count=count_condition,  # Stop after num_frames calls
        )

        debug = DebugPrintOp(
            self,
            name="debug",
            print_every=5,  # Print every 5 frames
            print_first=2,  # Always print first 2 frames in detail
        )

        # Connect pipeline
        self.add_flow(loader, debug, {("frame_data", "input")})

        print("[App] Pipeline composed successfully!")
        print("[App] Starting execution...\n")


def main():
    """Main entry point."""
    # Parse arguments
    parser = ArgumentParser(description="Test EndoNeRF data loading")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to EndoNeRF pulling directory"
    )
    parser.add_argument(
        "--num_frames", type=int, default=20, help="Number of frames to process (default: 20)"
    )
    args = parser.parse_args()

    # Validate path
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory does not exist: {args.data_dir}")
        return 1

    # Create and run application
    try:
        app = DataLoadingTestApp(data_dir=args.data_dir, num_frames=args.num_frames)
        app.run()

        print(f"\n{'='*60}")
        print("  Test completed successfully!")
        print(f"{'='*60}\n")
        return 0

    except Exception as e:
        print(f"\n{'='*60}")
        print("  ERROR: Test failed!")
        print(f"  {e}")
        print(f"{'='*60}\n")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
