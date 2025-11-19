# MIT License
#
# Copyright (c) 2025 EndoGaussian Project
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
Camera models for EndoGaussian rendering.
MIT-licensed clean-room implementation using standard pinhole camera model.
"""

import numpy as np
import torch
from torch import nn
from utils.graphics_utils import getProjectionMatrix, getWorld2View2


class Camera(nn.Module):
    """
    Pinhole camera model with intrinsics and extrinsics.

    Attributes:
        uid: Unique identifier
        colmap_id: COLMAP camera ID
        R: 3x3 rotation matrix (world to camera)
        T: 3D translation vector
        FoVx: Horizontal field of view
        FoVy: Vertical field of view
        image_name: Name of the image
        time: Timestamp for temporal data
        mask: Binary mask for the image
        original_image: Original image tensor
        original_depth: Original depth tensor
        image_width: Image width in pixels
        image_height: Image height in pixels
        znear: Near clipping plane
        zfar: Far clipping plane
        world_view_transform: World-to-view transformation matrix
        projection_matrix: Projection matrix
        full_proj_transform: Combined projection and view transformation
        camera_center: Camera center in world coordinates
    """

    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image,
        depth,
        mask,
        gt_alpha_mask,
        image_name,
        uid,
        trans=None,
        scale=1.0,
        data_device="cuda",
        time=0,
        Znear=None,
        Zfar=None,
    ):
        """
        Initialize camera with intrinsics and extrinsics.

        Args:
            colmap_id: COLMAP camera ID
            R: Rotation matrix
            T: Translation vector
            FoVx: Horizontal field of view
            FoVy: Vertical field of view
            image: Image tensor
            depth: Depth tensor
            mask: Mask tensor
            gt_alpha_mask: Ground truth alpha mask
            image_name: Name of the image
            uid: Unique identifier
            trans: Additional translation (default: [0, 0, 0])
            scale: Scale factor
            data_device: Device to store data on
            time: Timestamp
            Znear: Near clipping plane (optional)
            Zfar: Far clipping plane (optional)
        """
        super(Camera, self).__init__()

        # Initialize default translation if not provided
        if trans is None:
            trans = np.array([0.0, 0.0, 0.0])

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.time = time
        self.mask = mask

        # Set up data device
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        # Process image
        self.original_image = image.clamp(0.0, 1.0)
        self.original_depth = depth
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        # Apply alpha mask if provided
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width))

        # Set clipping planes
        if Zfar is not None and Znear is not None:
            self.zfar = Zfar
            self.znear = Znear
        else:
            # Default values for EndoNeRF
            self.zfar = 120.0
            self.znear = 0.01

        self.trans = trans
        self.scale = scale

        # Compute transformation matrices
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)

        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
        ).transpose(0, 1)

        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)

        self.camera_center = self.world_view_transform.inverse()[3, :3]


class MiniCam:
    """
    Lightweight camera class for rendering without heavy processing.

    Attributes:
        image_width: Image width in pixels
        image_height: Image height in pixels
        FoVy: Vertical field of view
        FoVx: Horizontal field of view
        znear: Near clipping plane
        zfar: Far clipping plane
        world_view_transform: World-to-view transformation matrix
        full_proj_transform: Combined projection and view transformation
        camera_center: Camera center in world coordinates
        time: Timestamp
    """

    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
        time,
    ):
        """
        Initialize minimal camera for rendering.

        Args:
            width: Image width
            height: Image height
            fovy: Vertical field of view
            fovx: Horizontal field of view
            znear: Near clipping plane
            zfar: Far clipping plane
            world_view_transform: World-to-view matrix
            full_proj_transform: Full projection matrix
            time: Timestamp
        """
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform

        # Compute camera center from view matrix
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

        self.time = time
