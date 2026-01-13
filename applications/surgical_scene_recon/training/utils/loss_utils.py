# MIT License
#
# Copyright (c) 2025-2026, EndoGaussian Project
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
Loss functions for EndoGaussian training.
MIT-licensed clean-room implementation.
"""

from math import exp

import torch
import torch.nn.functional as F


def TV_loss(x: torch.Tensor) -> torch.Tensor:
    """
    Compute Total Variation loss for image smoothness regularization.

    Args:
        x: Input tensor of shape [B, C, H, W]

    Returns:
        Normalized TV loss (scalar)
    """
    B, C, H, W = x.shape

    # Compute horizontal and vertical differences
    tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
    tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()

    # Normalize by total number of elements
    return (tv_h + tv_w) / (B * C * H * W)


def lpips_loss(img1: torch.Tensor, img2: torch.Tensor, lpips_model) -> torch.Tensor:
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity) loss.

    Args:
        img1: First image tensor
        img2: Second image tensor
        lpips_model: Pretrained LPIPS model

    Returns:
        Mean LPIPS loss (scalar)
    """
    loss = lpips_model(img1, img2)
    return loss.mean()


def l1_loss(
    network_output: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute masked L1 loss between network output and ground truth.

    This function supports optional masking for surgical tool exclusion in endoscopic images.

    Args:
        network_output: Predicted image/depth tensor
        gt: Ground truth tensor
        mask: Optional binary mask (0=ignore, non-zero=use). Can be 2D, 3D, or 4D.
              If provided, only computes loss on masked regions.
              Automatically handles shape broadcasting to match loss dimensions.

    Returns:
        Mean L1 loss (scalar)

    Note:
        Supports various mask shapes: [H, W], [C, H, W], [B, C, H, W].
        The function automatically expands masks to match the loss tensor shape.
    """
    # Compute absolute difference
    loss = torch.abs(network_output - gt)

    # Apply mask if provided
    if mask is not None:
        # If shapes already match, no expansion needed
        if mask.shape == loss.shape:
            loss = loss[mask != 0]
        else:
            # Need to expand mask to match loss shape
            if mask.ndim == 4 and loss.ndim == 4:
                # Both [B, C, H, W] - expand channels if needed
                if mask.shape[1] == 1 and loss.shape[1] > 1:
                    mask = mask.repeat(1, loss.shape[1], 1, 1)
            elif mask.ndim == 3 and loss.ndim == 4:
                # Mask is [C, H, W] or [H, W, C], loss is [B, C, H, W]
                # Assume mask is [H, W, 1] or [1, H, W]
                if mask.shape[0] == 1:
                    # Mask is [1, H, W], expand to [B, C, H, W]
                    mask = mask.unsqueeze(0).repeat(loss.shape[0], loss.shape[1], 1, 1)
                elif mask.shape[2] == 1:
                    # Mask is [H, W, 1], transpose and expand
                    mask = (
                        mask.permute(2, 0, 1)
                        .unsqueeze(0)
                        .repeat(loss.shape[0], loss.shape[1], 1, 1)
                    )
            elif mask.ndim == 3 and loss.ndim == 3:
                # Both [B/C, H, W] - check if already compatible
                if mask.shape[0] == 1 and loss.shape[0] > 1:
                    # Mask is [1, H, W], loss is [C, H, W] - repeat along first dim
                    mask = mask.repeat(loss.shape[0], 1, 1)
            elif mask.ndim == 2 and loss.ndim == 4:
                # Mask is [H, W], loss is [B, C, H, W]
                mask = mask.unsqueeze(0).unsqueeze(0).repeat(loss.shape[0], loss.shape[1], 1, 1)
            elif mask.ndim == 2 and loss.ndim == 3:
                # Mask is [H, W], loss is [C/B, H, W]
                mask = mask.unsqueeze(0).repeat(loss.shape[0], 1, 1)

            # Extract only masked regions
            loss = loss[mask != 0]

    return loss.mean()


def l2_loss(network_output: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Compute L2 (MSE) loss between network output and ground truth.

    Args:
        network_output: Predicted tensor
        gt: Ground truth tensor

    Returns:
        Mean squared error (scalar)
    """
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """
    Create a 1D Gaussian kernel.

    Args:
        window_size: Size of the Gaussian window
        sigma: Standard deviation of the Gaussian

    Returns:
        Normalized 1D Gaussian kernel
    """
    gauss = torch.Tensor(
        [exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)]
    )
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> torch.Tensor:
    """
    Create a 2D Gaussian window for SSIM computation.

    Args:
        window_size: Size of the window (e.g., 11)
        channel: Number of channels

    Returns:
        2D Gaussian window of shape [channel, 1, window_size, window_size]
    """
    # Create 1D Gaussian window
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)

    # Create 2D window by outer product
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)

    # Expand to all channels
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()

    return window


def _ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    channel: int,
    size_average: bool = True,
) -> torch.Tensor:
    """
    Internal SSIM computation function.

    Args:
        img1: First image
        img2: Second image
        window: Gaussian window
        window_size: Size of window
        channel: Number of channels
        size_average: Whether to average the result

    Returns:
        SSIM value(s)
    """
    # Compute local means
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Compute local variances and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # SSIM constants (from Wang et al. 2004)
    C1 = 0.01**2
    C2 = 0.03**2

    # Compute SSIM map
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(
    img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True
) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM) between two images.

    Based on "Image Quality Assessment: From Error Visibility to Structural Similarity"
    by Wang et al., IEEE TIP 2004.

    Args:
        img1: First image tensor
        img2: Second image tensor
        window_size: Size of Gaussian window (default: 11)
        size_average: Whether to average the result (default: True)

    Returns:
        SSIM value (scalar if size_average=True, otherwise per-image)
    """
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    # Move window to same device as images
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
