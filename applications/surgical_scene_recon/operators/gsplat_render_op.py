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
Gaussian Splatting Render Operator

Renders 3D Gaussian Splatting scene from camera viewpoint.

Phase 1.2a: Static mode (no temporal deformation)
Phase 1.2b: Dynamic mode (with temporal deformation)

Adapted from: holohub/applications/xr_gsplat/xr_gsplat.py::XrGsplatOp
"""

import math
import traceback

import cupy as cp
import holoscan as hs
import torch
from gsplat.rendering import rasterization
from holoscan.core import Operator, OperatorSpec
from holoscan.gxf import Entity


class GsplatRenderOp(Operator):
    """
    Render 3D Gaussian Splatting scene.

    Receives:
    - Gaussian parameters (from GsplatLoaderOp)
    - Camera pose + time (from EndoNeRFLoaderOp)

    Outputs:
    - Rendered RGB image
    - Rendered depth map (optional)

    Supports both static and dynamic rendering modes.
    """

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.frame_count = 0

        # Initialize parameters with defaults (will be overridden by spec.param values)
        self.width = 640
        self.height = 512
        self.render_mode = "RGB"
        self.near_plane = 0.01
        self.far_plane = 1000.0

    def setup(self, spec: OperatorSpec):
        """Define operator interface."""
        spec.input("splats")
        spec.input("pose_data")
        spec.output("rendered_rgb")
        # Note: depth output removed for Phase 1.2a simplicity

        spec.param("width", 640)
        spec.param("height", 512)
        spec.param("render_mode", "RGB")  # "RGB" or "RGB+D" or "RGB+ED"
        spec.param("near_plane", 0.01)
        spec.param("far_plane", 1000.0)

    def compute(self, op_input, op_output, context):
        """Render one frame."""
        # Receive inputs
        splats = op_input.receive("splats")
        pose_data = op_input.receive("pose_data")

        # DEBUG: Print every 10 frames
        if self.frame_count % 10 == 0:
            try:
                frame_idx = cp.asnumpy(cp.asarray(pose_data.get("frame_idx")))[0]
                print(f"[GsplatRender] Rendering frame {frame_idx}")
            except (KeyError, AttributeError, IndexError, RuntimeError):
                # frame_idx might not be available in pose_data
                print(f"[GsplatRender] Rendering frame {self.frame_count}")

        # Get rendering parameters based on mode
        mode = splats.get("mode", "static")

        if mode == "dynamic":
            # Phase 1.2b - dynamic rendering with temporal deformation
            params = self._get_dynamic_params(splats, pose_data)
        else:
            # Phase 1.2a - static rendering
            params = self._get_static_params(splats)

        # Extract camera parameters
        R = torch.as_tensor(pose_data.get("R"), device="cuda")
        T = torch.as_tensor(pose_data.get("T"), device="cuda")
        K = torch.as_tensor(pose_data.get("K"), device="cuda")

        # DEBUG: First frame details
        if self.frame_count == 0:
            print("\n[GsplatRender] ===== FIRST RENDER DEBUG =====")
            print(f"  Mode: {mode}")
            print(f"  Num gaussians: {len(params['means'])}")
            print(f"  Image size: {self.width}x{self.height}")
            print(f"  Render mode: {self.render_mode}")
            print(f"  R shape: {R.shape}")
            print(f"  T shape: {T.shape}")
            print(f"  K shape: {K.shape}")
            print("[GsplatRender] ================================\n")

        # Render
        rendered = self._render_gsplat(params, R, T, K)

        # Emit outputs
        self._emit_outputs(rendered, op_output, context)

        self.frame_count += 1

    def _get_static_params(self, splats):
        """Extract parameters for static rendering (already activated)."""
        return {
            "means": torch.as_tensor(splats["means"], device="cuda"),
            "scales": torch.as_tensor(splats["scales"], device="cuda"),
            "quats": torch.as_tensor(splats["quats"], device="cuda"),
            "opacities": torch.as_tensor(splats["opacities"], device="cuda"),
            "colors": torch.as_tensor(splats["colors"], device="cuda"),
        }

    def _get_dynamic_params(self, splats, pose_data):
        """
        Extract parameters for dynamic rendering (apply temporal deformation).

        Steps:
        1. Get base parameters (in raw space: log for scales, logit for opacities)
        2. Get time value from pose_data
        3. Apply deformation network
        4. Apply activations (exp, sigmoid) AFTER deformation
        """
        # Get base parameters (still in raw space)
        means_base = torch.as_tensor(splats["means_base"], device="cuda")
        scales_base = torch.as_tensor(splats["scales_base"], device="cuda")
        quats_base = torch.as_tensor(splats["quats_base"], device="cuda")
        opacities_base = torch.as_tensor(splats["opacities_base"], device="cuda")
        colors = torch.as_tensor(splats["colors"], device="cuda")

        # Get deformation network
        deform_net = splats["deform_net"]

        # Get time value
        time_val = torch.as_tensor(pose_data.get("time"), device="cuda")

        # Expand time to match number of gaussians
        N = means_base.shape[0]
        time_expanded = time_val.view(1, 1).repeat(N, 1)

        # Apply deformation
        with torch.no_grad():
            means_def, scales_def, quats_def, opacities_def = deform_net(
                point=means_base,
                scales=scales_base,
                rotations=quats_base,
                opacity=opacities_base.unsqueeze(-1),
                times_sel=time_expanded,
            )

        # Apply activations AFTER deformation
        means = means_def
        scales = torch.exp(scales_def)  # log → linear
        quats = quats_def  # Already normalized by network
        opacities = torch.sigmoid(opacities_def.squeeze(-1))  # logit → [0,1]

        # Debug: Print deformation effect on first frame
        if self.frame_count == 0:
            print("[GsplatRender] Dynamic deformation applied:")
            print(f"  - Mean displacement: {(means - means_base).abs().mean():.6f}")
            print(f"  - Scale change: {(scales_def - scales_base).abs().mean():.6f}")
            print(f"  - Time value: {time_val.item():.4f}")

        return {
            "means": means,
            "scales": scales,
            "quats": quats,
            "opacities": opacities,
            "colors": colors,
        }

    def _render_gsplat(self, params, R, T, K):
        """
        Render using gsplat.rasterization().

        Builds view matrix from R, T and calls gsplat rendering.
        """
        # Build view matrix (world to camera transform)
        # In gsplat, this is the inverse of camera-to-world
        view_matrix = torch.eye(4, device="cuda")
        view_matrix[:3, :3] = R
        view_matrix[:3, 3] = T

        # For gsplat, we need world-to-camera, which is already what we have
        # But gsplat expects the inverse, so:
        # Actually, let's check the convention...
        # In xr_gsplat: view_matrix = view_matrix.inverse()
        # But our R, T are already world-to-camera from the poses
        # Let's follow xr_gsplat pattern for now
        view_matrix = view_matrix.inverse()  # camera-to-world

        viewmats = view_matrix.unsqueeze(0)  # [1, 4, 4] - batch dimension
        Ks = K.unsqueeze(0)  # [1, 3, 3] - batch dimension

        # Compute spherical harmonics degree
        sh_degree = int(math.sqrt(params["colors"].shape[-2]) - 1)

        # DEBUG: First render details
        if self.frame_count == 0:
            print("[GsplatRender] Rendering details:")
            print(f"  - SH degree: {sh_degree}")
            print(f"  - View matrix shape: {viewmats.shape}")
            print(f"  - K shape: {Ks.shape}")
            print(f"  - Means: {params['means'].shape}")
            print(f"  - Colors: {params['colors'].shape}")

        # Render with gsplat
        try:
            rendered_colors, rendered_alphas, meta = rasterization(
                means=params["means"],
                quats=params["quats"],
                scales=params["scales"],
                opacities=params["opacities"],
                colors=params["colors"],
                viewmats=viewmats,
                Ks=Ks,
                width=self.width,
                height=self.height,
                sh_degree=sh_degree,
                near_plane=self.near_plane,
                far_plane=self.far_plane,
                render_mode=self.render_mode,
                packed=False,
                sparse_grad=False,
                rasterize_mode="classic",
            )

            # DEBUG: First render output
            if self.frame_count == 0:
                print("[GsplatRender] Render output:")
                print(f"  - Colors shape: {rendered_colors.shape}")
                print(f"  - Alphas shape: {rendered_alphas.shape}")
                print(
                    f"  - Colors range: [{rendered_colors.min():.4f}, {rendered_colors.max():.4f}]"
                )

            # Extract RGB and depth
            # rendered_colors: [1, H, W, C] where C depends on render_mode
            # RGB: first 3 channels
            # Depth: 4th channel if render_mode includes 'D'
            rgb = rendered_colors[0, ..., :3]  # [H, W, 3]
            depth = None
            if "D" in self.render_mode and rendered_colors.shape[-1] > 3:
                depth = rendered_colors[0, ..., 3:4]  # [H, W, 1]

            return {"rgb": rgb, "depth": depth, "alphas": rendered_alphas[0]}

        except Exception as e:
            print("[GsplatRender] ERROR during rendering:")
            print(f"  {e}")
            traceback.print_exc()
            raise

    def _emit_outputs(self, rendered, op_output, context):
        """Emit rendered RGB."""
        # Convert RGB to uint8 for visualization
        rgb = rendered["rgb"]
        rgb_uint8 = torch.clamp(rgb * 255.0, 0, 255).to(torch.uint8)

        # Convert to CuPy for Holoscan (Holoviz expects CuPy or NumPy, not torch)
        rgb_cupy = cp.asarray(rgb_uint8)

        # Create and emit RGB message
        rgb_message = Entity(context)
        rgb_message.add(hs.as_tensor(rgb_cupy), "rendered_rgb")
        op_output.emit(rgb_message, "rendered_rgb")

        # Note: Depth output removed for Phase 1.2a simplicity
        # Can be added back in later phases if needed

    def stop(self):
        """Cleanup."""
        print(f"[GsplatRender] Stopped after rendering {self.frame_count} frames")
