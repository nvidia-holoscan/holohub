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
from cucim.skimage import exposure as cucim_exposure


def clahe(
    image: cp.ndarray,
    tiles: tuple[int, int] = (8, 8),
    clip_limit: float = 0.01,
    nbins: int = 256,
) -> cp.ndarray:
    """Contrast Limited Adaptive Histogram Equalization."""

    data = cp.asarray(image, dtype=cp.float32, order="C")
    
    # 'tiles' param implies grid count (e.g., 8x8 grid).
    # cucim/skimage expects 'kernel_size' in pixels (e.g., 512/8 = 64px).
    h, w = data.shape[:2]
    grid_y = max(1, int(tiles[0]))
    grid_x = max(1, int(tiles[1]))
    kernel_size = (h // grid_y, w // grid_x)

    result = cucim_exposure.equalize_adapthist(
        cp.clip(data, 0.0, 1.0),
        kernel_size=kernel_size,
        clip_limit=float(clip_limit),
        nbins=int(nbins),
    )
    return cp.asarray(result, dtype=cp.float32, order="C")


__all__ = ["clahe"]
