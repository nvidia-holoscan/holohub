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
except ImportError:  # pragma: no cover - optional GPU dependency
    cp = None  # type: ignore

from ultra_post.core.pipeline import (
    Pipeline,
    create_node,
    pipeline_from_yaml,
    pipeline_to_yaml,
    run_pipeline,
)
from ultra_post.filters.registry import FILTERS


@unittest.skipIf(cp is None, "CuPy is required for pipeline tests.")
class PipelineCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.filters = FILTERS

    def test_registry_includes_gamma(self) -> None:
        self.assertIn("gamma_compression", self.filters)

    def test_pipeline_yaml_round_trip(self) -> None:
        pipeline: Pipeline = [create_node("gamma_compression", {"gamma": 1.2})]
        yaml_text = pipeline_to_yaml(pipeline)
        loaded = pipeline_from_yaml(yaml_text, filters=self.filters)
        self.assertEqual(len(loaded), 1)
        self.assertAlmostEqual(loaded[0]["params"]["gamma"], 1.2)

    def test_pipeline_execution(self) -> None:
        pipeline: Pipeline = [create_node("gamma_compression", {"gamma": 2.0})]
        data = cp.asarray([0.0, 0.25, 1.0], dtype=cp.float32)
        out = run_pipeline(pipeline, data)
        expected = cp.asarray([0.0, 0.5, 1.0], dtype=cp.float32)
        self.assertTrue(cp.allclose(out, expected))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
