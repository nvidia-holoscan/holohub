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

from enum import Enum

import cupy as cp


class MapChoice(str, Enum):
    blue = "Blue"
    orange = "Orange"


def _to_luminance(image: cp.ndarray) -> cp.ndarray:
    # Accept 2D or 3D (H, W, C). Convert to grayscale in [0,1].
    img = cp.asarray(image, dtype=cp.float32, order="C")
    img = cp.clip(img, 0.0, 1.0)
    luminance = cp.clip(img[..., 0], 0.0, 1.0)
    if img.ndim == 2:
        luminance = img
    elif img.ndim == 3 and img.shape[-1] >= 3:
        r = img[..., 0]
        g = img[..., 1]
        b = img[..., 2]
        # ITU-R BT.601 luma
        y = 0.299 * r + 0.587 * g + 0.114 * b
        luminance = cp.clip(y, 0.0, 1.0)
    return luminance


def _apply_blue_hue(x: cp.ndarray) -> cp.ndarray:
    # Blue-dominant mapping with minimal green, near-zero red
    # Shapes intensities slightly to maintain brightness perception.
    r = 0.05 * cp.power(x, 1.0)
    g = 0.15 * cp.sqrt(x)
    b = cp.power(x, 0.85)
    return cp.stack((r, g, b), axis=-1)


def _apply_orange_hue(x: cp.ndarray) -> cp.ndarray:
    # Fixed-hue orange in HSV space (H≈30°), strong saturation.
    # Use V shaped slightly to preserve perceived brightness.
    v = cp.power(x, 0.9)
    s = 0.85
    # For H=30° (sector 0): (r1, g1, b1) = (c, x, 0), with x=c*(1 - |(h*6 mod 2)-1|)=0.5*c
    c = v * s
    g1 = 0.5 * c
    r1 = c
    b1 = cp.zeros_like(c)
    m = v - c
    r = r1 + m  # = v
    g = g1 + m  # = v - 0.5*c
    b = b1 + m  # = v - c
    return cp.stack((r, g, b), axis=-1)


def color_map(
    image: cp.ndarray,
    mode: MapChoice | str = MapChoice.blue,
    blend: float = 1.0,
    enable: bool = True,
) -> cp.ndarray:
    """Apply a simple orange/blue colorization."""

    if not enable:
        return cp.asarray(image, dtype=cp.float32, order="C")

    y = _to_luminance(cp.asarray(image, dtype=cp.float32, order="C"))

    if isinstance(mode, str):
        token = mode.split(".")[-1]
        selected = MapChoice[token] if token in MapChoice.__members__ else MapChoice(token.capitalize())
    else:
        selected = MapChoice(mode)
    colored = _apply_blue_hue(y) if selected == MapChoice.blue else _apply_orange_hue(y)

    luma_out = 0.299 * colored[..., 0] + 0.587 * colored[..., 1] + 0.114 * colored[..., 2]
    eps = 1e-6
    scale = (y + eps) / (luma_out + eps)
    scale = cp.clip(scale, 0.0, 4.0)[..., None]
    colored = colored * scale

    blend = float(blend)
    if blend < 1.0:
        gray_rgb = cp.stack((y, y, y), axis=-1)
        colored = blend * colored + (1.0 - blend) * gray_rgb

    return cp.clip(colored, 0.0, 1.0).astype(cp.float32, copy=False)


__all__ = ["color_map", "MapChoice"]
