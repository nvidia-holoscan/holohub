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
Image quality metrics and utilities for EndoGaussian.
MIT-licensed clean-room implementation.
"""

import numpy as np
import torch
from typing import Optional
from PIL import Image


def tensor2array(tensor):
    """
    Convert a PyTorch tensor to numpy array.
    
    Args:
        tensor: PyTorch tensor or numpy array
        
    Returns:
        Numpy array
    """
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    else:
        return tensor


def mse(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Compute mean squared error between two images.
    
    Args:
        img1: First image tensor of shape [B, C, H, W]
        img2: Second image tensor of shape [B, C, H, W]
        
    Returns:
        MSE per image of shape [B, 1]
    """
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


@torch.no_grad()
def psnr(img1: torch.Tensor, img2: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Supports optional masking for surgical tool exclusion in endoscopic images.
    
    Args:
        img1: First image tensor (predicted)
        img2: Second image tensor (ground truth)
        mask: Optional binary mask. If provided, PSNR is computed only on masked regions.
              Can have 3 channels (RGB mask) or 1 channel (binary mask).
              
    Returns:
        PSNR value in dB (scalar)
        
    Note:
        - Assumes pixel values are in range [0, 1]
        - PSNR = 20 * log10(MAX_I / sqrt(MSE))
        - Higher PSNR = better quality
    """
    if mask is None:
        # Compute MSE without masking
        mse_val = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    else:
        # Compute MSE with masking
        # Validate mask has at least 2 dimensions
        if mask.ndim < 2:
            raise ValueError(f"Mask must have at least 2 dimensions, got shape {mask.shape}")
        
        if mask.shape[1] == 3:
            # RGB mask - compute weighted MSE across all channels
            mse_val = (((img1 - img2) ** 2) * mask).sum() / (mask.sum() + 1e-10)
        else:
            # Single channel mask - compute MSE and normalize by mask sum
            # Divide by 3.0 to account for 3 color channels
            mse_val = (((img1 - img2) ** 2) * mask).sum() / ((mask.sum() + 1e-10) * 3.0)
    
    # PSNR = 20 * log10(1.0 / sqrt(MSE))
    return 20 * torch.log10(1.0 / torch.sqrt(mse_val))


def rmse(a, b, mask: Optional[np.ndarray] = None) -> float:
    """
    Compute Root Mean Squared Error between two images.
    
    Args:
        a: First image (tensor or numpy array)
        b: Second image (tensor or numpy array)
        mask: Optional binary mask
        
    Returns:
        RMSE value (scalar)
    """
    # Convert tensors to numpy arrays if needed
    if torch.is_tensor(a):
        a = tensor2array(a)
    if torch.is_tensor(b):
        b = tensor2array(b)
    if torch.is_tensor(mask):
        mask = tensor2array(mask)
    
    # Validate inputs are at least 2D (images should have H, W dimensions)
    if a.ndim < 2:
        raise ValueError(f"Input 'a' must have at least 2 dimensions (H, W), got shape {a.shape}")
    if b.ndim < 2:
        raise ValueError(f"Input 'b' must have at least 2 dimensions (H, W), got shape {b.shape}")
    
    if mask is None:
        # Simple RMSE without masking
        rmse_val = (((a - b) ** 2).sum() / (a.shape[-1] * a.shape[-2])) ** 0.5
    else:
        # RMSE with masking
        # Expand mask to match dimensions if needed
        if len(mask.shape) == len(a.shape) - 1:
            mask = mask[..., None]
        
        mask_sum = np.sum(mask) + 1e-10
        rmse_val = (((a - b) ** 2 * mask).sum() / mask_sum) ** 0.5
    
    return rmse_val


def write_png(path: str, data: np.ndarray):
    """
    Write a numpy array as a PNG image.
    
    Args:
        path: Output file path
        data: Image data as numpy array [H, W, C]
        
    Returns:
        None (writes to disk)
    """
    Image.fromarray(data).save(path)
