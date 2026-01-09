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

from typing import Any

import cupy as cp

from ultra_post.filters.utils import apply_fill, get_fill_indices, get_matte_mask


class AutoMatte:
    """Wrapper that hides matte regions before running a sub-pipeline."""

    def __init__(self, filters: list[dict[str, Any]] | None = None) -> None:
        self._mask: cp.ndarray | None = None
        self._indices: cp.ndarray | None = None
        self._pipe: list = []

        if filters:
            from ultra_post.core.pipeline import CONFIG_VERSION, pipeline_from_dict

            self._pipe = pipeline_from_dict({"version": CONFIG_VERSION, "graph": list(filters)})

    def __call__(self, image: cp.ndarray, **kwargs: Any) -> cp.ndarray:
        img = cp.asarray(image)

        # Update Cache (Mask/Fill) if geometry changed
        if self._mask is None or self._mask.shape != img.shape[:2]:
            self._mask = get_matte_mask(img)
            self._indices = get_fill_indices(self._mask)

        if self._indices is None:
            return img

        # Fill -> Run -> Mask
        filled = apply_fill(img, self._indices)

        from ultra_post.core.pipeline import run_pipeline

        out = run_pipeline(self._pipe, filled)

        # Apply mask (handle broadcasting for 3D/Color images)
        mask = self._mask if out.ndim == 2 else self._mask[..., None]
        return out * mask


__all__ = ["AutoMatte"]
