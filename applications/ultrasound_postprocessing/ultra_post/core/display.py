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

from dataclasses import dataclass

import cupy as cp

from ultra_post.core.pipeline import Pipeline, run_pipeline


@dataclass
class DisplayCompressionSettings:
    mode: str
    dynamic_range_db: float
    gamma: float
    partial_log_mix: float


def _luminance(rgb: cp.ndarray) -> cp.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _match_luminance(rgb: cp.ndarray, target: cp.ndarray) -> cp.ndarray:
    eps = 1e-8
    target = cp.clip(target, 0.0, 1.0)
    current = cp.clip(_luminance(rgb), eps, 1.0)
    scale = (target + eps) / (current + eps)
    scaled = rgb * scale[..., None]
    return cp.clip(scaled, 0.0, 1.0).astype(cp.float32, copy=False)


def _power_compress(values: cp.ndarray, dynamic_range_db: float) -> cp.ndarray:
    eps = 1e-8
    normalized = (20.0 * cp.log10(cp.clip(values, eps, 1.0)) + dynamic_range_db) / dynamic_range_db
    return cp.clip(normalized, 0.0, 1.0).astype(cp.float32, copy=False)


def _gamma_compress(values: cp.ndarray, gamma: float) -> cp.ndarray:
    gamma = float(max(0.2, min(gamma, 5.0)))
    exponent = 1.0 / gamma
    return cp.power(cp.clip(values, 0.0, 1.0), exponent, dtype=cp.float32)


def _compress(values: cp.ndarray, settings: DisplayCompressionSettings) -> cp.ndarray:
    if settings.mode == "gamma":
        return _gamma_compress(values, settings.gamma)

    log_comp = _power_compress(values, settings.dynamic_range_db)
    if settings.mode != "partial_log":
        return log_comp

    mix = float(min(max(settings.partial_log_mix, 0.0), 1.0))
    linear = cp.clip(values, 0.0, 1.0).astype(cp.float32, copy=False)
    blended = (1.0 - mix) * linear + mix * log_comp
    return cp.clip(blended, 0.0, 1.0).astype(cp.float32, copy=False)


def compress_grayscale(image: cp.ndarray, settings: DisplayCompressionSettings) -> cp.ndarray:
    return _compress(image, settings)


def compress_color(rgb: cp.ndarray, settings: DisplayCompressionSettings) -> cp.ndarray:
    rgb = cp.clip(rgb, 0.0, 1.0).astype(cp.float32, copy=False)
    if settings.mode == "gamma":
        return _compress(rgb, settings)

    luminance = _compress(_luminance(rgb), settings)
    return _match_luminance(rgb, luminance)


def tensor_to_display(
    image: cp.ndarray, settings: DisplayCompressionSettings, apply_compression: bool
) -> cp.ndarray:
    """Convert tensor to display image with selectable compression."""

    result: cp.ndarray
    if image.ndim == 3 and image.shape[-1] >= 3:
        rgb = cp.clip(image[..., :3], 0.0, 1.0).astype(cp.float32, copy=False)
        if not apply_compression:
            result = rgb
        else:
            result = compress_color(rgb, settings)
    else:
        scalar = cp.asarray(image, dtype=cp.float32)
        if not apply_compression:
            result = cp.clip(scalar, 0.0, 1.0)
        else:
            result = compress_grayscale(scalar, settings)
    return result


def run_pipeline_colormap_last(pipeline: Pipeline, tensor: cp.ndarray) -> cp.ndarray:
    """Apply pipeline while ensuring color_map runs last."""

    non_cm = [node for node in pipeline if node.get("op") != "color_map"]
    cm_nodes = [node for node in pipeline if node.get("op") == "color_map"]
    ordered_pipeline = non_cm + cm_nodes
    return run_pipeline(ordered_pipeline, tensor)


def ensure_rgba(tensor: cp.ndarray) -> cp.ndarray:
    """Convert a tensor to contiguous RGBA float32."""

    arr = cp.clip(cp.asarray(tensor, dtype=cp.float32).squeeze(), 0.0, 1.0)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        return cp.zeros((1, 1, 4), dtype=cp.float32)

    h, w, c = arr.shape
    if c == 4:
        return cp.ascontiguousarray(arr[..., :4])

    base = cp.broadcast_to(arr[..., :1], (h, w, 3))
    rgb = arr if c == 3 else base
    alpha = cp.ones((h, w, 1), dtype=cp.float32)
    return cp.ascontiguousarray(cp.concatenate((rgb, alpha), axis=-1))


__all__ = [
    "DisplayCompressionSettings",
    "compress_color",
    "compress_grayscale",
    "tensor_to_display",
    "run_pipeline_colormap_last",
    "ensure_rgba",
]
