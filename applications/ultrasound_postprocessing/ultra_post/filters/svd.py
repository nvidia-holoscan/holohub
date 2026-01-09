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

from typing import Literal

import cupy as cp


def svd_denoise(
    image: cp.ndarray,
    rank: int = 32,
    suppress: Literal["low", "high"] = "low",
    shrink: float = 0.0,
) -> cp.ndarray:
    """Low-rank or high-rank suppression via singular value truncation."""

    img = cp.asarray(image, dtype=cp.float32, order="C")
    U, s, Vt = cp.linalg.svd(img, full_matrices=False)
    
    n = s.shape[0]
    k = min(max(1, int(rank)), n)
    
    if n > 0:
        indices = cp.arange(n)
        keep = indices < k if suppress == "low" else indices >= max(0, n - k)
        
        if not keep.any():
            keep[0 if suppress != "high" else -1] = True
            
        filtered_s = cp.where(keep, s, float(shrink) * s)
        return cp.matmul(U * filtered_s[None, :], Vt)
        
    return img


__all__ = ["svd_denoise"]
