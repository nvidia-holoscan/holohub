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


def get_matte_mask(image: cp.ndarray) -> cp.ndarray:
    """Return a boolean mask where ``True`` indicates valid tissue pixels.

    Background is detected via connected components seeded from the image
    corners, preventing inner anechoic regions from being treated as matte.
    """
    data = cp.asarray(image)
    background = (data == 0) if data.ndim == 2 else cp.all(data == 0, axis=-1)

    # Label connected components of the background
    labels, _ = cndimage.label(background, structure=cp.ones((3, 3)))

    # Identify background labels from corners (corners > 0 avoids tissue/label 0)
    corners = labels[[0, 0, -1, -1], [0, -1, 0, -1]]
    return ~cp.isin(labels, cp.unique(corners[corners > 0]))


def get_fill_indices(mask: cp.ndarray) -> cp.ndarray:
    """Compute nearest-valid pixel indices for each location."""
    if not (mask := cp.asarray(mask, bool)).any():
        return cp.zeros((mask.ndim, *mask.shape), dtype=cp.int32)

    # edt returns indices to nearest zero (tissue) in ~mask
    return cndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)


def apply_fill(image: cp.ndarray, indices: cp.ndarray) -> cp.ndarray:
    """Fill invalid pixels using nearest valid neighbors via ``indices``."""
    return cp.asarray(image)[tuple(indices)]


__all__ = ["apply_fill", "get_fill_indices", "get_matte_mask"]
