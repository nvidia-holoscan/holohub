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


def gamma_compression(image: cp.ndarray, gamma: float = 1.0) -> cp.ndarray:
    """Power-law gamma correction."""

    gamma = float(gamma)
    if gamma <= 0.0:
        raise ValueError("Gamma must be positive.")
    return cp.power(cp.asarray(image, dtype=cp.float32), 1.0 / gamma)


__all__ = ["gamma_compression"]
