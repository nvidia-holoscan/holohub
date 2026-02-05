# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Test Dynamic Rendering with Holoviz - Phase 1.2b

Test application for DYNAMIC gaussian splatting rendering with temporal deformation.
Pipeline: EndoNeRFLoader → GsplatLoader (dynamic) → GsplatRender → Holoviz + ImageSaver

Usage:
    python dynamic_rendering_viz.py \
        --data_dir /path/to/EndoNeRF/pulling \
        --output_dir /path/to/output \
        --checkpoint /path/to/checkpoint.pt \
        --num_frames 63
"""

import os
from argparse import ArgumentParser

from holoscan.conditions import CountCondition
from holoscan.core import Application
from holoscan.operators import HolovizOp
from holoscan.resources import UnboundedAllocator

# Import our custom operators
from operators import EndoNeRFLoaderOp, GsplatLoaderOp, GsplatRenderOp, ImageSaverOp


class DynamicRenderingVizApp(Application):
    """
    Test application for DYNAMIC rendering with Holoviz visualization.

    This uses temporal deformation to render accurate per-frame reconstructions,
    accounting for tissue deformation and tool movement over time.

    Pipeline:
        EndoNeRFLoaderOp → GsplatLoaderOp (loads deformation network)
                 ↓              ↓
        (pose_data + TIME)  (base splats + deform_net)
                 ↓              ↓
              GsplatRenderOp (applies deformation(time))
                 ↓
          ┌──────┴──────┐
          ↓             ↓
       Holoviz     ImageSaver
    """

    def __init__(
        self,
        data_dir,
        output_dir,
        checkpoint_path,
        num_frames=-1,
        loop=True,
        export_ply=False,
        export_ply_frame=1,
        export_ply_path="",
        depth_mode="binocular",
    ):
        super().__init__()
        self.name = "Dynamic Rendering Visualization"
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.checkpoint_path = checkpoint_path
        self.num_frames = num_frames
        self.loop = loop
        self.export_ply = export_ply
        self.export_ply_frame = export_ply_frame
        self.export_ply_path = export_ply_path
        self.depth_mode = depth_mode

    def compose(self):
        """Build the visualization pipeline."""
        print(f"\n{'='*70}")
        print("  Dynamic Rendering Visualization Application")
        print(f"{'='*70}")
        print(f"Data directory: {self.data_dir}")
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"Frames to render: {self.num_frames}")
        print(f"Loop playback: {self.loop}")
        print(f"Export PLY: {self.export_ply}")
        if self.export_ply:
            print(f"  - Export at frame: {self.export_ply_frame}")
            print(f"  - Export path: {self.export_ply_path or 'auto-generated'}")
        print(f"Depth mode: {self.depth_mode}")
        print("Mode: DYNAMIC (with temporal deformation)")
        print(f"{'='*70}\n")

        # Create operators
        # Data loader (with optional count limit)
        loader_kwargs = {
            "name": "loader",
            "data_dir": self.data_dir,
            "loop": self.loop,
            "max_frames": self.num_frames if self.num_frames > 0 else -1,
            "depth_mode": self.depth_mode,
        }

        # Only add count condition if we want to limit frames
        if self.num_frames > 0 and not self.loop:
            count_condition = CountCondition(self, count=self.num_frames)
            loader_kwargs["count"] = count_condition

        # Create allocator (like reference applications)
        host_allocator = UnboundedAllocator(self, name="host_allocator")

        loader = EndoNeRFLoaderOp(self, **loader_kwargs)

        # Gsplat checkpoint loader with DEFORMATION ENABLED
        gsplat_loader = GsplatLoaderOp(
            self,
            name="gsplat_loader",
            checkpoint_path=self.checkpoint_path,
            use_deformation=True,  # ← DYNAMIC MODE!
        )

        # Renderer (will apply temporal deformation)
        renderer = GsplatRenderOp(
            self,
            name="renderer",
            width=640,
            height=512,
            render_mode="RGB",
            near_plane=0.01,
            far_plane=1000.0,
            export_ply=self.export_ply,
            export_ply_frame=self.export_ply_frame,
            export_ply_path=self.export_ply_path,
        )

        # Holoviz operator for visualization (with allocator like reference apps)
        holoviz = HolovizOp(
            self,
            name="holoviz",
            allocator=host_allocator,
            width=640,
            height=512,
            tensors=[dict(name="rendered_rgb", type="color", opacity=1.0, priority=0)],
        )

        # Image saver for debugging/inspection
        image_saver = ImageSaverOp(
            self,
            name="image_saver",
            output_dir=os.path.join(self.output_dir, "rendered_dynamic"),
            prefix="dynamic",
            save_every=1,
            verbose=True,
        )

        # Connect pipeline
        self.add_flow(loader, gsplat_loader, {("frame_data", "trigger")})
        self.add_flow(loader, renderer, {("frame_data", "pose_data")})
        self.add_flow(gsplat_loader, renderer, {("splats", "splats")})
        self.add_flow(renderer, holoviz, {("rendered_rgb", "receivers")})
        self.add_flow(renderer, image_saver, {("rendered_rgb", "input")})

        print("[App] Pipeline composed successfully!")
        print("[App] Pipeline structure:")
        print("[App]   loader ─┬──(trigger)──> gsplat_loader ──(splats+deform_net)──┐")
        print(
            "[App]           └──(pose_data+time)─────────────────────────────────┼──> renderer ──┬──> holoviz"
        )
        print(
            "[App]                                                                                  └──> image_saver"
        )
        print("[App] ")
        print("[App] Rendering mode: DYNAMIC (temporal deformation active)")
        print("[App] Starting visualization...\n")
        print("[App] Press ESC or close window to exit.\n")


def main():
    """Main entry point."""
    # Parse arguments
    parser = ArgumentParser(description="Test DYNAMIC Gaussian Splatting rendering with Holoviz")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to EndoNeRF pulling directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for rendering (default: ./output)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to gsplat checkpoint (.pt file) with deformation network",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=-1,
        help="Number of frames to render (default: -1 = all frames)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        default=True,
        help="Loop playback continuously (default: True)",
    )
    parser.add_argument(
        "--no-loop", dest="loop", action="store_false", help="Disable loop playback"
    )
    parser.add_argument(
        "--export_ply",
        action="store_true",
        default=False,
        help="Export gaussians to PLY file",
    )
    parser.add_argument(
        "--export_ply_frame",
        type=int,
        default=1,
        help="Frame number at which to export PLY (default: 1)",
    )
    parser.add_argument(
        "--export_ply_path",
        type=str,
        default="",
        help="Output path for PLY file (default: auto-generated in output_dir)",
    )
    parser.add_argument(
        "--depth_mode",
        type=str,
        default="binocular",
        choices=["binocular", "monocular"],
        help="Depth mode: 'binocular' (stereo depth) or 'monocular' (mono depth estimation)",
    )
    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory does not exist: {args.data_dir}")
        return 1

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint does not exist: {args.checkpoint}")
        return 1

    # Handle default PLY export path
    export_ply_path = args.export_ply_path
    if args.export_ply and not export_ply_path:
        export_ply_path = os.path.join(args.output_dir, "gaussians.ply")

    # Create and run application
    try:
        app = DynamicRenderingVizApp(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint,
            num_frames=args.num_frames,
            loop=args.loop,
            export_ply=args.export_ply,
            export_ply_frame=args.export_ply_frame,
            export_ply_path=export_ply_path,
            depth_mode=args.depth_mode,
        )
        app.run()

        print(f"\n{'='*70}")
        print("  Dynamic visualization completed successfully!")
        print(f"{'='*70}\n")
        return 0

    except Exception as e:
        print(f"\n{'='*70}")
        print("  ERROR: Visualization failed!")
        print(f"  {e}")
        print(f"{'='*70}\n")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
