# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
G-SHARP Scene Reconstruction — Phase 1: Streaming Inference

Runs Depth Anything V2 and MedSAM3 in parallel on video frames via Holoscan,
writes partial EndoNeRF data (images, depth_raw, masks) to disk, and displays
a live 3-panel preview (source | depth colormap | mask overlay) via HoloViz.

This is Phase 1 of the five-phase hybrid workflow described in
architecture_analysis_v2.md. Phase 2 (VGGT batch poses) and Phase 3
(format conversion) complete the EndoNeRF dataset for GSplat training.

Data flow:

    ImageSource ─┬─→ DA2Op ─────────┬─→ DataPrepOp → images/, depth_raw/, masks/
                 ├─→ MedSAM3Op ─────┤
                 ├───────────────────┘
                 │
                 ├─→ OverlayOp ──→ HoloViz (live 3-panel preview)
                 │       ↑  ↑
                 │   DA2Op  MedSAM3Op
                 └───────────────────

Usage:
    python gsplat_scene_recon.py \\
        --images /path/to/frame/directory \\
        --output /path/to/output \\
        --da2-root /path/to/depth_anything_v2_code \\
        --da2-checkpoint /path/to/depth_anything_v2_vits.pth \\
        [--sam3-checkpoint /path/to/medsam3_checkpoint.pt] \\
        [--headless] \\
        [--config phase1_config.yaml]
"""

import glob
import os
from argparse import ArgumentParser

import cv2
from holoscan.conditions import CountCondition
from holoscan.core import Application
from holoscan.operators import HolovizOp
from holoscan.resources import UnboundedAllocator

from operators import (
    DataPrepOp,
    DepthAnythingV2Op,
    ImageDirectorySourceOp,
    MedSAM3SegmentationOp,
    OverlayComposerOp,
)


class SceneReconInferenceApp(Application):
    """
    Phase 1 streaming Holoscan application.

    Runs DA2 and MedSAM3 on each frame, saves partial EndoNeRF data to disk,
    and optionally displays a live 3-panel preview.
    """

    def __init__(self, args):
        super().__init__()
        self.name = "G-SHARP Phase 1: Streaming Inference"
        self._args = args

    def compose(self):
        args = self._args
        image_dir = args.images
        output_dir = args.output
        headless = args.headless

        # Count frames to set the source stopping condition
        frames = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        num_frames = len(frames)
        if num_frames == 0:
            raise ValueError(f"No PNG files found in {image_dir}")
        print(f"[Phase1] {num_frames} frames in {image_dir}")

        # ----------------------------------------------------------------
        # Source: read PNG frames from directory
        # ----------------------------------------------------------------
        source = ImageDirectorySourceOp(
            self,
            CountCondition(self, count=num_frames),
            name="source",
            directory=image_dir,
        )

        # ----------------------------------------------------------------
        # Inference: DA2 (dense depth) + MedSAM3 (tool segmentation)
        # ----------------------------------------------------------------
        da2 = DepthAnythingV2Op(
            self,
            name="depth_anything_v2",
            da2_root=args.da2_root,
            checkpoint=args.da2_checkpoint,
            encoder=args.da2_encoder,
        )

        medsam3 = MedSAM3SegmentationOp(
            self,
            name="medsam3",
            checkpoint_path=args.sam3_checkpoint,
        )

        # ----------------------------------------------------------------
        # Data writer: save frames, depth, masks to disk
        # ----------------------------------------------------------------
        data_prep = DataPrepOp(
            self,
            name="data_prep",
            output_dir=output_dir,
        )

        # ----------------------------------------------------------------
        # Connect source → inference operators → data writer
        # ----------------------------------------------------------------
        # Source broadcasts to DA2, MedSAM3, and DataPrepOp
        self.add_flow(source, da2, {("frame_out", "frame_in")})
        self.add_flow(source, medsam3, {("frame_out", "frame_in")})
        self.add_flow(source, data_prep, {("frame_out", "frame_in")})

        # DA2 and MedSAM3 feed depth/masks into DataPrepOp
        self.add_flow(da2, data_prep, {("depth_out", "depth_in")})
        self.add_flow(medsam3, data_prep, {("mask_out", "mask_in")})

        # ----------------------------------------------------------------
        # Visualization branch (optional)
        # ----------------------------------------------------------------
        if not headless:
            overlay = OverlayComposerOp(self, name="overlay")

            # Read one frame to get dimensions for proper aspect ratio
            sample = cv2.imread(frames[0])
            h, w = sample.shape[:2]
            win_w = min(w * 3, 1920)
            win_h = int(win_w * h / (w * 3))

            pool = UnboundedAllocator(self, name="pool")
            holoviz = HolovizOp(
                self,
                allocator=pool,
                name="holoviz",
                headless=False,
                width=win_w,
                height=win_h,
                tensors=[dict(name="composite", type="color")],
                window_title="G-SHARP: Source | Depth | Mask",
            )

            self.add_flow(source, overlay, {("frame_out", "source")})
            self.add_flow(da2, overlay, {("depth_out", "depth")})
            self.add_flow(medsam3, overlay, {("mask_out", "masks")})
            self.add_flow(overlay, holoviz, {("out", "receivers")})

        # ----------------------------------------------------------------
        print("[Phase1] Pipeline composed successfully.")
        print(f"[Phase1]   Headless: {headless}")
        print(f"[Phase1]   Output:   {output_dir}")


def main():
    parser = ArgumentParser(description="G-SHARP Phase 1: Streaming Inference")

    # Required paths
    parser.add_argument(
        "--images",
        required=True,
        help="Directory containing PNG frames",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for partial EndoNeRF data (images, depth_raw, masks)",
    )
    _app_dir = os.path.dirname(os.path.abspath(__file__))
    _models = os.path.join(_app_dir, "models")

    parser.add_argument(
        "--da2-root",
        default=os.path.join(_models, "depth_anything_v2"),
        help="DA2 model code directory (default: bundled models/)",
    )
    parser.add_argument(
        "--da2-checkpoint",
        default=os.path.join(_app_dir, "assets", "da2", "depth_anything_v2_vits.pth"),
        help="Path to Depth Anything V2 .pth checkpoint",
    )
    # Optional
    parser.add_argument(
        "--sam3-checkpoint",
        default=os.path.join(_app_dir, "assets", "medsam3", "checkpoint_8_new_best.pt"),
        help="Path to MedSAM3 checkpoint (empty = HuggingFace default)",
    )
    parser.add_argument(
        "--da2-encoder",
        default="vits",
        choices=["vits", "vitb", "vitl"],
        help="DA2 encoder variant (default: vits)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without HoloViz visualization (for Docker/CI)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config YAML for operator overrides",
    )

    args = parser.parse_args()

    # Load config if provided
    app = SceneReconInferenceApp(args)

    config_path = args.config or os.path.join(os.path.dirname(__file__), "phase1_config.yaml")
    if os.path.exists(config_path):
        app.config(config_path)
        print(f"[Phase1] Config loaded: {config_path}")

    print("\n" + "=" * 70)
    print("  G-SHARP Phase 1: Streaming Inference")
    print("=" * 70)
    print(f"  Images:         {args.images}")
    print(f"  Output:         {args.output}")
    print(f"  DA2 root:       {args.da2_root}")
    print(f"  DA2 checkpoint: {args.da2_checkpoint}")
    print(f"  DA2 encoder:    {args.da2_encoder}")
    print(f"  SAM3 checkpoint: {args.sam3_checkpoint or '(default)'}")
    print(f"  Headless:       {args.headless}")
    print("=" * 70 + "\n")

    app.run()

    print(f"\n[Phase1] Complete. Partial EndoNeRF data at: {args.output}")
    print("[Phase1]   images/    — RGB frames (PNG)")
    print("[Phase1]   depth_raw/ — dense depth maps (float32 .npy)")
    print("[Phase1]   masks/     — tool segmentation masks (uint8 PNG)")
    print("[Phase1] Next steps:")
    print("[Phase1]   Phase 2: Run VGGT batch inference for camera poses")
    print("[Phase1]   Phase 3: Format conversion → complete EndoNeRF dataset")


if __name__ == "__main__":
    main()
