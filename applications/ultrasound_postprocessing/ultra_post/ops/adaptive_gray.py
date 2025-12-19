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


def adaptive_gray_map(
    image: cp.ndarray,
    radius: int = 5,
    beta: float = 1.0,
    ref_level: float = 0.5,
    auto_ref: bool = True,
    preserve_mean: bool = True,
) -> cp.ndarray:
    """Locally suppress minority polarity to enhance contrast."""

    data = cp.asarray(image, dtype=cp.float32, order="C")
    radius = _positive_radius(radius)
    beta = float(beta)
    ksize = 2 * radius + 1
    eps = 1e-7

    ref = cp.median(data) if auto_ref else cp.asarray(float(ref_level), dtype=cp.float32)

    pos = cp.maximum(data - ref, 0.0)
    neg = cp.maximum(ref - data, 0.0)

    area = float(ksize * ksize)
    sum_pos = uniform_filter(pos, size=ksize, mode="reflect") * area
    sum_neg = uniform_filter(neg, size=ksize, mode="reflect") * area

    sign_pos = (data >= ref).astype(cp.float32)
    same = sign_pos * sum_pos + (1.0 - sign_pos) * sum_neg
    opp = sign_pos * sum_neg + (1.0 - sign_pos) * sum_pos

    minority_ratio = cp.clip((opp - same) / (opp + same + eps), 0.0, 1.0)
    mapped = ref + (1.0 - beta * minority_ratio) * (data - ref)

    if preserve_mean:
        mapped = mapped - cp.mean(mapped) + cp.mean(data)

    return mapped


__all__ = ["adaptive_gray_map"]
