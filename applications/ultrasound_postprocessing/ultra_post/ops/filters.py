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
from cupyx.scipy import ndimage as cndimage


def _ensure_odd(value: int) -> int:
    value = int(value)
    return value + 1 if value % 2 == 0 else max(1, value)


def median_filter(image: cp.ndarray, size: int = 3) -> cp.ndarray:
    """Median filter for impulse/speckle reduction."""

    return cndimage.median_filter(cp.asarray(image, dtype=cp.float32), size=_ensure_odd(size), mode="reflect")


def gaussian_filter(image: cp.ndarray, sigma: float = 1.0) -> cp.ndarray:
    """Gaussian blur with configurable sigma."""

    return cndimage.gaussian_filter(cp.asarray(image, dtype=cp.float32), sigma=float(sigma), mode="reflect")


def unsharp_mask(image: cp.ndarray, sigma: float = 1.0, amount: float = 1.5, threshold: float = 0.0) -> cp.ndarray:
    """Sharpening via blurred subtraction."""

    data = cp.asarray(image, dtype=cp.float32)
    blurred = cndimage.gaussian_filter(data, sigma=float(sigma), mode="reflect")
    mask = data - blurred
    if threshold > 0.0:
        mask = cp.where(cp.abs(mask) < float(threshold), 0.0, mask)
    return data + float(amount) * mask


__all__ = ["median_filter", "gaussian_filter", "unsharp_mask"]
