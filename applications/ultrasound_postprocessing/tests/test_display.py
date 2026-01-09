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

import unittest

try:
    import cupy as cp  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cp = None  # type: ignore

from ultra_post.core.display import (
    DisplayCompressionSettings,
    compress_color,
    compress_grayscale,
    ensure_rgba,
    tensor_to_display,
)


@unittest.skipIf(cp is None, "CuPy is required for display tests.")
class DisplayTests(unittest.TestCase):
    def test_ensure_rgba_shapes(self) -> None:
        gray = cp.ones((2, 3), dtype=cp.float32)
        rgba = ensure_rgba(gray)
        self.assertEqual(rgba.shape, (2, 3, 4))

        rgb = cp.ones((2, 3, 3), dtype=cp.float32)
        rgba_rgb = ensure_rgba(rgb)
        self.assertEqual(rgba_rgb.shape, (2, 3, 4))

        already_rgba = cp.ones((2, 3, 4), dtype=cp.float32)
        out = ensure_rgba(already_rgba)
        self.assertEqual(out.shape, (2, 3, 4))

    def test_compression_gamma(self) -> None:
        settings = DisplayCompressionSettings(mode="gamma", dynamic_range_db=40.0, gamma=2.0, partial_log_mix=0.5)
        data = cp.asarray([0.25, 1.0], dtype=cp.float32)
        out = compress_grayscale(data, settings)
        expected = cp.asarray([0.5, 1.0], dtype=cp.float32)
        self.assertTrue(cp.allclose(out, expected))

    def test_tensor_to_display_modes(self) -> None:
        settings = DisplayCompressionSettings(mode="power", dynamic_range_db=20.0, gamma=1.0, partial_log_mix=0.5)
        img = cp.asarray([[1.0, 0.1]], dtype=cp.float32)
        out = tensor_to_display(img, settings, apply_compression=True)
        self.assertEqual(out.shape, img.shape)

        rgb = cp.ones((1, 2, 3), dtype=cp.float32)
        colored = compress_color(rgb, settings)
        self.assertEqual(colored.shape, rgb.shape)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
