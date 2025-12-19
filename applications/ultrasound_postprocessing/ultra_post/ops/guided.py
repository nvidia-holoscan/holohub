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
from cupyx.scipy.ndimage import uniform_filter


def _positive_radius(value: int) -> int:
    return max(1, int(value))


def _box_filter(img: cp.ndarray, radius: int) -> cp.ndarray:
    k = 2 * radius + 1
    return uniform_filter(img, size=(k, k), mode="reflect")


def _guided_filter_channel(p: cp.ndarray, I: cp.ndarray, radius: int, eps: float) -> cp.ndarray:
    mean_I, mean_p = _box_filter(I, radius), _box_filter(p, radius)
    mean_Ip, mean_II = _box_filter(I * p, radius), _box_filter(I * I, radius)
    
    a = (mean_Ip - mean_I * mean_p) / (mean_II - mean_I * mean_I + eps)
    b = mean_p - a * mean_I
    return _box_filter(a, radius) * I + _box_filter(b, radius)


def guided_filter(
    image: cp.ndarray,
    radius: int = 5,
    eps: float = 0.01,
    use_luminance: bool = True,
) -> cp.ndarray:
    """Edge-aware smoothing guided by image luminance or channels."""

    img = cp.asarray(image, dtype=cp.float32, order="C")
    r, eps = _positive_radius(radius), float(eps)

    if img.ndim == 2:
        return cp.clip(_guided_filter_channel(img, img, r, eps), 0.0, 1.0)

    if img.ndim == 3:
        guidance = img
        if use_luminance and img.shape[-1] >= 3:
            lum = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
            guidance = cp.stack([lum] * img.shape[-1], axis=-1)
        
        channels = [_guided_filter_channel(img[..., i], guidance[..., i], r, eps) for i in range(img.shape[-1])]
        return cp.clip(cp.stack(channels, axis=-1), 0.0, 1.0)

    return img


__all__ = ["guided_filter"]
