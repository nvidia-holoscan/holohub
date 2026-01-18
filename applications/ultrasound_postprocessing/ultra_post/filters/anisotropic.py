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


def anisotropic_diffusion(
    image: cp.ndarray,
    num_iter: int = 10,
    kappa: float = 0.1,
    gamma: float = 0.1,
    function: str = "exponential",
) -> cp.ndarray:
    """Perona–Malik anisotropic diffusion for edge-preserving smoothing."""

    u = cp.asarray(image, dtype=cp.float32, order="C")
    kappa = float(kappa)
    gamma = float(gamma)
    func = str(function)

    for _ in range(int(num_iter)):
        n = cp.roll(u, -1, axis=0)
        s = cp.roll(u, 1, axis=0)
        e = cp.roll(u, -1, axis=1)
        w = cp.roll(u, 1, axis=1)

        dN, dS, dE, dW = n - u, s - u, e - u, w - u

        if func == "reciprocal":
            cN = 1.0 / (1.0 + (dN / kappa) ** 2)
            cS = 1.0 / (1.0 + (dS / kappa) ** 2)
            cE = 1.0 / (1.0 + (dE / kappa) ** 2)
            cW = 1.0 / (1.0 + (dW / kappa) ** 2)
        else:
            cN = cp.exp(-((dN / kappa) ** 2))
            cS = cp.exp(-((dS / kappa) ** 2))
            cE = cp.exp(-((dE / kappa) ** 2))
            cW = cp.exp(-((dW / kappa) ** 2))

        u += gamma * (cN * dN + cS * dS + cE * dE + cW * dW)
        u = cp.clip(u, 0.0, 1.0)

    return u


__all__ = ["anisotropic_diffusion"]
