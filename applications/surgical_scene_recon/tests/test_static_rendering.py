#!/usr/bin/env python3
import sys
import os
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
Test Static Rendering - Phase 1.2a

Test application for static gaussian splatting rendering.
Pipeline: EndoNeRFLoader → GsplatLoader → GsplatRender → DebugPrint

Usage:
    python test_static_rendering.py \
        --data_dir /path/to/EndoNeRF/pulling \
        --checkpoint /path/to/checkpoint.pt \
        --num_frames 5
"""

import os  # noqa: E402
from argparse import ArgumentParser  # noqa: E402

from holoscan.core import Application  # noqa: E402
from holoscan.conditions import CountCondition  # noqa: E402

# Import our custom operators
from operators import (  # noqa: E402
    EndoNeRFLoaderOp,
    GsplatLoaderOp,
    GsplatRenderOp,
    DebugPrintOp,
)


class StaticRenderingTestApp(Application):
    """
    Test application for static rendering verification.
    
    Pipeline:
        EndoNeRFLoaderOp → GsplatLoaderOp (loads once)
                 ↓              ↓
        (pose_data)      (splats)
                 ↓              ↓
              GsplatRenderOp
                 ↓
            DebugPrintOp
    """
    
    def __init__(self, data_dir, checkpoint_path, num_frames=5):
        super().__init__()
        self.name = "Static Rendering Test"
        self.data_dir = data_dir
        self.checkpoint_path = checkpoint_path
        self.num_frames = num_frames
    
    def compose(self):
        """Build the test pipeline."""
        print(f"\n{'='*70}")
        print("  Static Rendering Test Application")
        print(f"{'='*70}")
        print(f"Data directory: {self.data_dir}")
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"Will render {self.num_frames} frames")
        print(f"{'='*70}\n")
        
        # Create operators
        # Use CountCondition to limit execution
        count_condition = CountCondition(self, count=self.num_frames)
        
        # Data loader (with count limit)
        loader = EndoNeRFLoaderOp(
            self,
            name="loader",
            data_dir=self.data_dir,
            loop=False,
            max_frames=self.num_frames,
            count=count_condition
        )
        
        # Gsplat checkpoint loader (loads once, emits every frame)
        gsplat_loader = GsplatLoaderOp(
            self,
            name="gsplat_loader",
            checkpoint_path=self.checkpoint_path,
            use_deformation=False  # Static mode for Phase 1.2a
        )
        
        # Renderer
        renderer = GsplatRenderOp(
            self,
            name="renderer",
            width=640,
            height=512,
            render_mode="RGB",  # Just RGB for now
            near_plane=0.01,
            far_plane=1000.0
        )
        
        # Debug operator to inspect rendered output
        debug = DebugPrintOp(
            self,
            name="debug",
            print_every=5,
            print_first=2
        )
        
        # Connect pipeline
        # loader emits frame_data (pose + time) every frame
        # This triggers gsplat_loader to emit splats
        # Both pose_data and splats go to renderer
        self.add_flow(loader, gsplat_loader, {("frame_data", "trigger")})
        self.add_flow(loader, renderer, {("frame_data", "pose_data")})
        self.add_flow(gsplat_loader, renderer, {("splats", "splats")})
        self.add_flow(renderer, debug, {("rendered_rgb", "input")})
        
        print("[App] Pipeline composed successfully!")
        print("[App] Pipeline structure:")
        print("[App]   loader ─┬──(trigger)──> gsplat_loader ──(splats)──┐")
        print("[App]           └──(pose_data)───────────────────────────┼──> renderer ──> debug")
        print("[App] Starting execution...\n")


def main():
    """Main entry point."""
    # Parse arguments
    parser = ArgumentParser(description="Test static Gaussian Splatting rendering")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to EndoNeRF pulling directory"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to gsplat checkpoint (.pt file)"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=5,
        help="Number of frames to render (default: 5)"
    )
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory does not exist: {args.data_dir}")
        return 1
    
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint does not exist: {args.checkpoint}")
        return 1
    
    # Create and run application
    try:
        app = StaticRenderingTestApp(
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint,
            num_frames=args.num_frames
        )
        app.run()
        
        print(f"\n{'='*70}")
        print("  Static rendering test completed successfully!")
        print(f"{'='*70}\n")
        return 0
        
    except Exception as e:
        print(f"\n{'='*70}")
        print("  ERROR: Test failed!")
        print(f"  {e}")
        print(f"{'='*70}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
