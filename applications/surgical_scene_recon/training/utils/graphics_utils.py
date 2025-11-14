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
Computer graphics utilities for camera transformations and projections.
MIT-licensed clean-room implementation using standard computer graphics algorithms.
"""

import torch
import math
import numpy as np
from typing import NamedTuple, Optional


class BasicPointCloud(NamedTuple):
    """
    Simple container for point cloud data.
    
    Attributes:
        points: Nx3 array of 3D point positions
        colors: Nx3 array of RGB colors
        normals: Nx3 array of surface normals
    """
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray


def geom_transform_points(points: torch.Tensor, transf_matrix: torch.Tensor) -> torch.Tensor:
    """
    Apply homogeneous transformation to 3D points.
    
    Args:
        points: Points tensor of shape [P, 3]
        transf_matrix: Transformation matrix of shape [4, 4]
        
    Returns:
        Transformed points of shape [P, 3]
    """
    P, _ = points.shape
    
    # Convert to homogeneous coordinates [P, 4]
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    
    # Apply transformation: points_hom @ transf_matrix.T
    # Transpose is needed because PyTorch uses row vectors [P, 4] @ [4, 4] = [P, 4]
    points_out = torch.matmul(points_hom, transf_matrix.T)
    
    # Convert back to 3D coordinates (divide by w)
    denom = points_out[:, 3:] + 0.0000001  # Add epsilon for numerical stability
    return points_out[:, :3] / denom


def getWorld2View(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Construct world-to-view transformation matrix.
    
    Args:
        R: 3x3 rotation matrix
        t: 3D translation vector
        
    Returns:
        4x4 transformation matrix
    """
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def getWorld2View2(R: np.ndarray, t: np.ndarray, 
                   translate: Optional[np.ndarray] = None, 
                   scale: float = 1.0) -> np.ndarray:
    """
    Construct world-to-view transformation matrix with optional translation and scaling.
    
    This function first applies the camera extrinsics (R, t), then applies additional
    translation and scaling in world space.
    
    Args:
        R: 3x3 rotation matrix (camera orientation)
        t: 3D translation vector (camera position)
        translate: Additional translation to apply in world space (default: [0, 0, 0])
        scale: Scale factor to apply
        
    Returns:
        4x4 world-to-view transformation matrix
    """
    # Initialize default translation if not provided
    if translate is None:
        translate = np.array([0.0, 0.0, 0.0])
    
    # Build initial camera-to-world matrix
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    
    # Convert to world-to-camera (invert)
    C2W = np.linalg.inv(Rt)
    
    # Apply translation and scaling
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    
    # Convert back to camera-to-world
    Rt = np.linalg.inv(C2W)
    
    return np.float32(Rt)


def getProjectionMatrix(znear: float, zfar: float, fovX: float, fovY: float) -> torch.Tensor:
    """
    Construct OpenGL-style projection matrix from field of view parameters.
    
    Args:
        znear: Near clipping plane distance
        zfar: Far clipping plane distance
        fovX: Horizontal field of view in radians
        fovY: Vertical field of view in radians
        
    Returns:
        4x4 projection matrix
    """
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)
    
    # Compute frustum bounds
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    
    # Initialize projection matrix
    P = torch.zeros(4, 4)
    
    z_sign = 1.0
    
    # Fill projection matrix (OpenGL convention)
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    
    return P


def fov2focal(fov: float, pixels: int) -> float:
    """
    Convert field of view to focal length in pixels.
    
    Args:
        fov: Field of view in radians (must be in range (0, π))
        pixels: Image dimension in pixels (width or height)
        
    Returns:
        Focal length in pixels
        
    Raises:
        ValueError: If fov is not in valid range (0, π)
        
    Note:
        Based on the pinhole camera model:
        focal = pixels / (2 * tan(fov / 2))
    """
    # Validate FOV is in valid range
    if fov <= 0 or fov >= math.pi:
        raise ValueError(f"Field of view must be in range (0, π), got {fov:.4f} radians")
    
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal: float, pixels: int) -> float:
    """
    Convert focal length to field of view.
    
    Args:
        focal: Focal length in pixels (must be positive)
        pixels: Image dimension in pixels (width or height)
        
    Returns:
        Field of view in radians
        
    Raises:
        ValueError: If focal length is not positive
        
    Note:
        Based on the pinhole camera model:
        fov = 2 * atan(pixels / (2 * focal))
    """
    # Validate focal length is positive
    if focal <= 0:
        raise ValueError(f"Focal length must be positive, got {focal:.4f} pixels")
    
    return 2 * math.atan(pixels / (2 * focal))
