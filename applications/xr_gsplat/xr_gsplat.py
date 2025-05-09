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
import math
import os
from argparse import ArgumentParser

import gsplat
import holoscan.core
import numpy as np
import torch
from gsplat_loader_op import GsplatLoaderOp
from scipy.spatial.transform import Rotation

import holohub.xr as xr


class XrGsplatOp(holoscan.core.Operator):
    def __init__(self, fragment, xr_session, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.xr_session = xr_session
        self.width = 1920 * 2
        self.height = 1080 * 2

    def start(self):
        self.color_swapchain = xr.XrSwapchainCuda(
            self.xr_session, xr.XrSwapchainCudaFormat.R8G8B8A8_SRGB, self.width * 2, self.height
        )
        self.depth_swapchain = xr.XrSwapchainCuda(
            self.xr_session, xr.XrSwapchainCudaFormat.D32_SFLOAT, self.width * 2, self.height
        )

    def setup(self, spec: holoscan.core.OperatorSpec):
        spec.input("splats")
        spec.input("xr_frame_state")
        spec.output("xr_composition_layer")

    def compute(self, op_input, op_output, context):
        frame_state = op_input.receive("xr_frame_state")
        cuda_stream = op_input.receive_cuda_stream("xr_frame_state")

        composition_layer = xr.XrCompositionLayerProjectionStorage.create_for_frame(
            frame_state, self.xr_session, self.color_swapchain, self.depth_swapchain
        )

        # Create view matrices for both eyes from composition layer views
        view_matrices = []
        for view in composition_layer.views:
            # Extract position and orientation from the view's pose

            rotation = torch.tensor(
                Rotation.from_quat(
                    [
                        view.pose.orientation.x,
                        view.pose.orientation.y,
                        view.pose.orientation.z,
                        view.pose.orientation.w,
                    ]
                ).as_matrix()
            )

            # Apply coordinate system transformation
            # Create view matrix
            view_matrix = torch.eye(4)
            view_matrix[:3, :3] = rotation
            view_matrix[:3, 3] = torch.tensor(
                [
                    view.pose.position.x,
                    view.pose.position.y,
                    view.pose.position.z,
                ]
            )

            # Convert from OpenGL (z-back) to OpenCV (z-forward, y-down) coordinate system
            # This requires a 180 degree rotation around X axis
            coordinate_transform = torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            view_matrix = torch.matmul(view_matrix.float(), coordinate_transform.float())

            view_matrix = view_matrix.inverse()

            # Convert from z-back to z-forward coordinate system
            z_flip = torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            view_matrix = torch.matmul(view_matrix, z_flip)

            view_matrices.append(view_matrix)

        # Stack the view matrices into a single tensor [2, 4, 4]
        view_matrices = torch.stack(view_matrices).cuda()
        # Keep only the first view matrix
        # view_matrices = view_matrices[0:1]  # Shape becomes [1, 4, 4]

        # Create intrinsics matrices for both views
        intrinsics_list = []
        for view in composition_layer.views:
            # Get FOV angles from view
            left_fov = view.fov.angleLeft
            right_fov = view.fov.angleRight
            up_fov = view.fov.angleUp
            down_fov = view.fov.angleDown

            # Create intrinsics matrix
            intrinsics_mat = np.zeros((3, 3))

            # For asymmetric FOVs, we need to handle left and right FOVs separately
            # fx calculation uses the total horizontal FOV (left + right)
            fx = self.width / (np.tan(right_fov) + np.tan(-left_fov))
            intrinsics_mat[0, 0] = fx  # fx
            # Similarly for vertical FOV
            fy = self.height / (np.tan(up_fov) + np.tan(-down_fov))
            intrinsics_mat[1, 1] = fy  # fy
            # Principal point needs to account for asymmetric FOVs
            intrinsics_mat[0, 2] = fx * np.tan(-left_fov)
            intrinsics_mat[1, 2] = fy * np.tan(-down_fov)
            intrinsics_mat[2, 2] = 1.0

            intrinsics_list.append(intrinsics_mat)

        # Stack the intrinsics matrices into a single tensor [2, 3, 3]
        intrinsics = np.stack(intrinsics_list)
        intrinsics = torch.from_numpy(intrinsics).float().cuda()
        # intrinsics = intrinsics[0:1]  # Shape becomes [1, 3, 3]

        splats = op_input.receive("splats")
        means = torch.as_tensor(splats["means"])
        quats = torch.as_tensor(splats["quats"])
        scales = torch.as_tensor(splats["scales"])
        opacities = torch.as_tensor(splats["opacities"])
        colors = torch.as_tensor(splats["colors"])
        sh_degree = int(math.sqrt(colors.shape[-2]) - 1)

        view_configuration_depth_range = self.xr_session.view_configuration_depth_range()
        near_plane = view_configuration_depth_range.recommendedNearZ
        far_plane = view_configuration_depth_range.recommendedFarZ

        # Render the gaussians using gsplat
        rendered_colors, rendered_alphas, _ = gsplat.rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=view_matrices,  # View matrix with batch dimension [C, 4, 4]
            Ks=intrinsics,  # Camera intrinsics matrix [C, 3, 3]
            near_plane=near_plane,
            far_plane=far_plane,
            width=self.width,
            height=self.height,
            sh_degree=sh_degree,
            render_mode="RGB+ED",
            # this is to speedup large-scale rendering by skipping far-away Gaussians.
            # radius_clip=3,
        )

        rendered_depth = rendered_colors[..., 3:4]
        rendered_colors = rendered_colors[..., :3]

        # convert rendered_depth from linear depth to vulkan depth buffer format
        # using the standard Vulkan depth transform: z' = (far * z - near * far)/(z * (far - near))
        rendered_depth = torch.clamp(rendered_depth, near_plane, far_plane)
        rendered_depth = (far_plane * rendered_depth - near_plane * far_plane) / (
            rendered_depth * (far_plane - near_plane)
        )

        # Scale rendered_depth to 0-255 range and convert to uint8
        # Expand depth values to all RGB channels
        # rendered_colors = (rendered_depth * 255).to(torch.uint8).repeat(1, 1, 1, 3)

        rendered_colors = (rendered_colors * 255).to(torch.uint8)
        rendered_alphas = (rendered_alphas * 255).to(torch.uint8)

        rendered_cuda_left = rendered_colors[0]
        rendered_alphas_cuda_left = rendered_alphas[0]

        rendered_cuda_right = rendered_colors[1]
        rendered_alphas_cuda_right = rendered_alphas[1]

        # Acquire swapchain images
        color_image = self.color_swapchain.acquire()
        depth_image = self.depth_swapchain.acquire()

        # Convert holoscan.Tensor to torch.Tensor
        color_tensor = torch.as_tensor(color_image)
        depth_tensor = torch.as_tensor(depth_image)

        # Copy rendered_cuda into left half of color_tensor
        color_tensor[:, : color_tensor.shape[1] // 2, :3] = (
            rendered_cuda_left  # Copy RGB channels to left half
        )
        color_tensor[:, : color_tensor.shape[1] // 2, 3:4] = (
            rendered_alphas_cuda_left  # Set alpha for left half
        )

        color_tensor[:, color_tensor.shape[1] // 2 :, :3] = (
            rendered_cuda_right  # Copy RGB channels to right half
        )
        color_tensor[:, color_tensor.shape[1] // 2 :, 3:4] = (
            rendered_alphas_cuda_right  # Set alpha for right half
        )

        depth_tensor[:, : depth_tensor.shape[1] // 2] = rendered_depth[0]
        depth_tensor[:, depth_tensor.shape[1] // 2 :] = rendered_depth[1]

        self.color_swapchain.release(cuda_stream)
        self.depth_swapchain.release(cuda_stream)

        op_output.emit(
            composition_layer, "xr_composition_layer", "XrCompositionLayerProjectionStorage"
        )


class GsplatApp(holoscan.core.Application):
    def __init__(self):
        super().__init__()

    def compose(self):
        self.xr_session = xr.XrSession(self)

        xr_begin_frame = xr.XrBeginFrameOp(
            self, xr_session=self.xr_session, name="xr_begin_frame_op"
        )
        xr_end_frame = xr.XrEndFrameOp(self, xr_session=self.xr_session, name="xr_end_frame_op")
        gsplat_source = GsplatLoaderOp(self, name="gsplat_source", **self.kwargs("gsplat_loader"))
        xr_gsplat = XrGsplatOp(self, xr_session=self.xr_session, name="xr_gsplat")

        # Connect operators
        self.add_flow(gsplat_source, xr_gsplat, {("splats", "splats")})
        self.add_flow(xr_begin_frame, xr_gsplat, {("xr_frame_state", "xr_frame_state")})
        self.add_flow(xr_begin_frame, xr_end_frame, {("xr_frame_state", "xr_frame_state")})
        self.add_flow(xr_gsplat, xr_end_frame, {("xr_composition_layer", "xr_composition_layers")})


if __name__ == "__main__":
    parser = ArgumentParser(description="XR Gsplat Application.")
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to the configuration file.",
    )

    args = parser.parse_args()

    app = GsplatApp()
    app.config(args.config)
    app.run()
