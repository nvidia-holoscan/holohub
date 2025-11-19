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
General utility functions for EndoGaussian.
MIT-licensed clean-room implementation.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute inverse sigmoid (logit) function with numerical stability.
    
    Args:
        x: Input tensor with values in (0, 1)
        eps: Epsilon for clamping to prevent inf/nan at boundaries (default: 1e-7)
        
    Returns:
        Inverse sigmoid: log(x / (1 - x))
        
    Note:
        Input is clamped to [eps, 1-eps] to prevent numerical instabilities
        when x approaches 0 or 1.
    """
    # Clamp input to valid range to prevent inf/nan
    x_clamped = torch.clamp(x, min=eps, max=1.0 - eps)
    return torch.log(x_clamped / (1.0 - x_clamped))


def PILtoTorch(pil_image: Image.Image, resolution: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """
    Convert a PIL image to a PyTorch tensor.
    
    Args:
        pil_image: Input PIL image
        resolution: Optional target resolution (width, height). If None, keeps original size.
        
    Returns:
        PyTorch tensor of shape [C, H, W] with values in [0, 1]
    """
    # Resize if resolution is specified
    if resolution is not None:
        resized_image_PIL = pil_image.resize(resolution)
    else:
        resized_image_PIL = pil_image
    
    # Convert to numpy array and normalize to [0, 1]
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    
    # Handle grayscale vs RGB
    if len(resized_image.shape) == 3:
        # RGB image: [H, W, C] -> [C, H, W]
        return resized_image.permute(2, 0, 1)
    else:
        # Grayscale image: [H, W] -> [1, H, W]
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


@torch.jit.script
def percentile_torch(t: torch.Tensor, q: float) -> float:
    """
    Compute the q-th percentile of a flattened tensor.
    
    This is a PyTorch implementation equivalent to numpy.percentile(..., interpolation="nearest").
    
    Args:
        t: Input tensor
        q: Percentile to compute, must be between 0 and 100 inclusive
        
    Returns:
        The q-th percentile value (scalar)
        
    Note:
        Uses torch.kthvalue for efficient computation without full sorting.
    """
    # Compute the k-th value index (1-based)
    # Note: kthvalue is 1-based, so k=1 gives the minimum value
    k = 1 + int(round(0.01 * float(q) * (t.numel() - 1)))
    
    # Get the k-th smallest value
    result = t.view(-1).kthvalue(k).values.item()
    
    return result
