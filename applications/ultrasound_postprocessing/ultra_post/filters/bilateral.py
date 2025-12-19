# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

import cupy as cp


def _positive_radius(value: int) -> int:
    return max(1, int(value))


def _bilateral_gray(image: cp.ndarray, radius: int, spatial_sigma: float, range_sigma: float) -> cp.ndarray:
    r = _positive_radius(radius)
    spatial_sig, range_sig = float(spatial_sigma), float(range_sigma)

    ax = cp.arange(-r, r + 1, dtype=cp.float32)
    xx, yy = cp.meshgrid(ax, ax)
    spatial = cp.exp(-(xx**2 + yy**2) / (2.0 * spatial_sig**2))
    padded = cp.pad(image, r, mode="reflect")

    result, norm = cp.zeros_like(image), cp.zeros_like(image)
    h, w = image.shape

    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            w_spatial = spatial[dy + r, dx + r]
            patch = padded[dy + r : dy + r + h, dx + r : dx + r + w]
            w_range = cp.exp(-((patch - image) ** 2) / (2.0 * range_sig**2))
            weight = w_spatial * w_range
            result += weight * patch
            norm += weight

    return result / cp.maximum(norm, 1e-8)


def bilateral_filter(
    image: cp.ndarray,
    radius: int = 3,
    spatial_sigma: float = 2.0,
    range_sigma: float = 0.1,
    per_channel: bool = True,
) -> cp.ndarray:
    """Edge-preserving bilateral smoothing."""

    img = cp.asarray(image, dtype=cp.float32, order="C")
    img = cp.clip(img, 0.0, 1.0)

    if img.ndim == 2:
        return cp.clip(_bilateral_gray(img, radius, spatial_sigma, range_sigma), 0.0, 1.0)

    if img.ndim == 3:
        if per_channel:
            channels = [_bilateral_gray(img[..., i], radius, spatial_sigma, range_sigma) for i in range(img.shape[-1])]
        else:
            # Luminance guidance (simple RGB approx)
            guidance = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
            filtered_guidance = _bilateral_gray(guidance, radius, spatial_sigma, range_sigma)
            gain = filtered_guidance / cp.maximum(guidance, 1e-6)
            channels = [img[..., i] * gain for i in range(img.shape[-1])]
        return cp.clip(cp.stack(channels, axis=-1), 0.0, 1.0)

    return img  # Fallback


__all__ = ["bilateral_filter"]
