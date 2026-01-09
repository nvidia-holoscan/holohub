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

from __future__ import annotations

import cupy as cp
from cupyx.scipy import ndimage as cndimage


def _positive_odd(value: int) -> int:
    """Clamp to 1+ and coerce to the nearest odd value."""

    value = int(value)
    if value <= 0:
        return 1
    return value if value % 2 else value + 1


def non_local_means(
    image: cp.ndarray,
    h: float = 0.1,
    patch_size: int = 7,
    patch_distance: int = 11,
    fast_mode: bool = True,
) -> cp.ndarray:
    """Classic Non-Local Means denoiser implemented with CuPy."""

    data = cp.asarray(image, dtype=cp.float32, order="C")
    p_size = _positive_odd(patch_size)
    r = _positive_odd(patch_distance)
    h = float(h)

    if r <= 0 or h <= 0.0:
        return data

    padded = cp.pad(data, r, mode="reflect")
    height, width = data.shape
    accum, weights = cp.zeros_like(data), cp.zeros_like(data)
    h2, k_area = h**2 + 1e-8, float(p_size**2)

    offsets = range(-r, r + 1)
    if fast_mode and r > 1:
        offsets = [o for o in offsets if o % 2 == 0]

    center = padded[r : r + height, r : r + width]

    for dy in offsets:
        sy, ey = r + dy, r + dy + height
        for dx in offsets:
            shifted = padded[sy:ey, r + dx : r + dx + width]
            diff2 = (center - shifted) ** 2
            patch_mse = cndimage.uniform_filter(diff2, size=p_size, mode="reflect")
            weight = cp.exp(-(patch_mse * k_area) / h2)
            accum += weight * shifted
            weights += weight

    return cp.where(weights > 1e-8, accum / weights, data)


__all__ = ["non_local_means"]
