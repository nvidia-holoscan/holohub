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
Gaussian Splatting Loader Operator

Loads trained 3D Gaussian Splatting checkpoint with optional temporal deformation.

Phase 1.2a: Static mode (no deformation) - simpler, like xr_gsplat
Phase 1.2b: Dynamic mode (with deformation) - full fidelity

Adapted from: holohub/applications/xr_gsplat/gsplat_loader_op.py
"""

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from holoscan.core import Operator, OperatorSpec


class GsplatLoaderOp(Operator):
    """
    Load Gaussian Splatting checkpoint and emit parameters.

    Supports two modes:
    - Static: Load and activate parameters immediately (Phase 1.2a)
    - Dynamic: Load base parameters + deformation network (Phase 1.2b)

    The mode is determined by:
    1. use_deformation parameter
    2. Presence of 'deform_net' in checkpoint
    """

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)

        # Will be set in start()
        self.mode = None
        self.num_gaussians = 0

        # Static mode attributes
        self.means = None
        self.scales = None
        self.quats = None
        self.opacities = None
        self.colors = None

        # Dynamic mode attributes (Phase 1.2b)
        self.means_base = None
        self.scales_base = None
        self.quats_base = None
        self.opacities_base = None
        self.deform_net = None

    def setup(self, spec: OperatorSpec):
        """Define operator interface."""
        spec.input("trigger")  # Dummy input to trigger on each frame
        spec.output("splats")
        spec.param("checkpoint_path", "")
        spec.param("use_deformation", False)  # False = static mode (Phase 1.2a)

    def start(self):
        """Load checkpoint and prepare gaussian parameters."""
        print("[GsplatLoader] Starting...")
        print(f"[GsplatLoader] Checkpoint: {self.checkpoint_path}")
        print(f"[GsplatLoader] Use deformation: {self.use_deformation}")

        # Validate checkpoint path
        ckpt_path = Path(self.checkpoint_path)
        if not ckpt_path.exists():
            raise ValueError(f"Checkpoint not found: {self.checkpoint_path}")

        # Load checkpoint
        print("[GsplatLoader] Loading checkpoint...")
        ckpt = torch.load(self.checkpoint_path, map_location="cuda", weights_only=False)

        # Print checkpoint info
        print("[GsplatLoader] Checkpoint info:")
        if "step" in ckpt:
            print(f"  - Training step: {ckpt['step']}")
        if "stage" in ckpt:
            print(f"  - Training stage: {ckpt['stage']}")

        # Determine mode
        has_deform_net = "deform_net" in ckpt and ckpt["deform_net"] is not None
        use_dynamic = self.use_deformation and has_deform_net

        if use_dynamic:
            print("[GsplatLoader] Deformation network found in checkpoint")
            self._load_dynamic_mode(ckpt)
        else:
            if not has_deform_net:
                print("[GsplatLoader] No deformation network in checkpoint")
            self._load_static_mode(ckpt)

        print(f"[GsplatLoader] Ready! Mode: {self.mode}, Gaussians: {self.num_gaussians}")

    def _load_static_mode(self, ckpt):
        """
        Load in static mode (like xr_gsplat).

        Apply activations immediately:
        - exp(scales) to convert from log-space
        - sigmoid(opacities) to convert from logit-space
        - normalize(quats) for valid quaternions
        """
        print("[GsplatLoader] Loading in STATIC mode")

        splats = ckpt["splats"]

        # Load and activate parameters
        self.means = splats["means"].cuda()
        self.num_gaussians = len(self.means)

        # Apply activations (convert from learned space to rendering space)
        self.scales = torch.exp(splats["scales"]).cuda()  # log → linear
        self.quats = F.normalize(splats["quats"], p=2, dim=-1).cuda()  # normalize quaternions
        self.opacities = torch.sigmoid(splats["opacities"]).cuda()  # logit → [0,1]

        # Colors (spherical harmonics) - concatenate DC and higher order
        sh0 = splats["sh0"].cuda()
        shN = splats["shN"].cuda()
        self.colors = torch.cat([sh0, shN], dim=-2)

        self.mode = "static"

        # Print statistics
        print("[GsplatLoader] Static mode loaded:")
        print(f"  - Gaussians: {self.num_gaussians}")
        print(f"  - Means range: [{self.means.min():.2f}, {self.means.max():.2f}]")
        print(f"  - Scales range: [{self.scales.min():.4f}, {self.scales.max():.4f}]")
        print(f"  - Opacities range: [{self.opacities.min():.4f}, {self.opacities.max():.4f}]")
        print(f"  - SH degree: {int(torch.sqrt(torch.tensor(self.colors.shape[-2])).item()) - 1}")

    def _load_dynamic_mode(self, ckpt):
        """
        Load in dynamic mode (Phase 1.2b).

        Keep parameters in RAW space (log/logit):
        - scales stay in log-space (will exp AFTER deformation)
        - opacities stay in logit-space (will sigmoid AFTER deformation)
        - quats are normalized

        Also load deformation network.
        """
        print("[GsplatLoader] Loading in DYNAMIC mode")

        # Add training code to Python path
        import sys

        # Get training code path from environment or use relative path
        # Priority: 1) Environment variable, 2) Local training/ directory
        training_code_path = os.environ.get(
            "GSPLAT_TRAINING_PATH", os.path.join(os.path.dirname(__file__), "..", "training")
        )
        if training_code_path not in sys.path:
            sys.path.insert(0, training_code_path)

        splats = ckpt["splats"]

        # Load BASE parameters (in RAW space, NO activations)
        self.means_base = splats["means"].cuda()
        self.num_gaussians = len(self.means_base)

        self.scales_base = splats["scales"].cuda()  # Keep in log-space!
        self.quats_base = F.normalize(splats["quats"], p=2, dim=-1).cuda()  # Normalize quats
        self.opacities_base = splats["opacities"].cuda()  # Keep in logit-space!

        # Colors (spherical harmonics) - same as static
        sh0 = splats["sh0"].cuda()
        shN = splats["shN"].cuda()
        self.colors = torch.cat([sh0, shN], dim=-2)

        # Load deformation network
        print("[GsplatLoader] Loading deformation network...")
        from argparse import Namespace

        from scene.deformation import deform_network

        # Get config from checkpoint or use defaults
        if "config" in ckpt:
            cfg = ckpt["config"]
        else:
            # Use defaults from training code (matching checkpoint)
            print("[GsplatLoader] Warning: No config in checkpoint, using defaults")
            cfg = Namespace(
                bounds=1.5,
                kplanes_config={
                    "grid_dimensions": 2,
                    "input_coordinate_dim": 4,
                    "output_coordinate_dim": 64,
                    "resolution": [64, 64, 64, 100],
                },
                multires=[1, 2, 4, 8],
                no_grid=False,
                no_dx=False,
                no_ds=False,
                no_dr=False,
                no_do=False,
                net_width=32,  # Model was trained with 32, not 64
                timebase_pe=6,
                defor_depth=0,
                posebase_pe=10,
                scale_rotation_pe=10,
                opacity_pe=10,
                timenet_width=64,
                timenet_output=32,
            )

        # Create deformation network
        self.deform_net = deform_network(cfg).cuda()
        self.deform_net.load_state_dict(ckpt["deform_net"])
        self.deform_net.eval()

        self.mode = "dynamic"

        # Print statistics
        print("[GsplatLoader] Dynamic mode loaded:")
        print(f"  - Gaussians: {self.num_gaussians}")
        print(f"  - Means range: [{self.means_base.min():.2f}, {self.means_base.max():.2f}]")
        print(
            f"  - Scales range (log): [{self.scales_base.min():.4f}, {self.scales_base.max():.4f}]"
        )
        print(
            f"  - Opacities range (logit): [{self.opacities_base.min():.4f}, {self.opacities_base.max():.4f}]"
        )
        print(f"  - SH degree: {int(torch.sqrt(torch.tensor(self.colors.shape[-2])).item()) - 1}")
        print("  - Deformation network: Ready")

    def compute(self, op_input, op_output, context):
        """Emit gaussian parameters."""
        # Receive trigger (we don't actually use it, just need it for scheduling)
        _ = op_input.receive("trigger")

        if self.mode == "static":
            # Static mode: Emit pre-activated parameters
            splats_dict = {
                "means": self.means,
                "scales": self.scales,
                "quats": self.quats,
                "opacities": self.opacities,
                "colors": self.colors,
                "mode": "static",
                "deform_net": None,
            }
        else:
            # Dynamic mode: Emit base parameters + deformation network
            splats_dict = {
                "means_base": self.means_base,
                "scales_base": self.scales_base,
                "quats_base": self.quats_base,
                "opacities_base": self.opacities_base,
                "colors": self.colors,
                "mode": "dynamic",
                "deform_net": self.deform_net,
            }

        # Emit as dictionary (not Entity, just a Python dict)
        # This is passed as-is to the render operator
        op_output.emit(splats_dict, "splats")

    def stop(self):
        """Cleanup."""
        print("[GsplatLoader] Stopped")
