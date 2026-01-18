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

import importlib.util
import unittest

try:
    import cupy as cp  # type: ignore
except ImportError:  # pragma: no cover
    cp = None  # type: ignore

from ultra_post.app.holoscan_operators import FuncOp
from ultra_post.filters.registry import FILTERS

_HOLOSCAN_AVAILABLE = importlib.util.find_spec("holoscan") is not None


@unittest.skipIf(cp is None or not _HOLOSCAN_AVAILABLE, "CuPy/Holoscan required for FuncOp test.")
class FuncOpTests(unittest.TestCase):
    def test_func_op_gamma(self) -> None:
        from holoscan.core import Application, Operator

        app = Application()
        op_instance = FuncOp(
            app, name="gamma_op", fn=FILTERS["gamma_compression"], params={"gamma": 2.0}
        )

        self.assertIsInstance(op_instance, Operator)

        # Verify the underlying logic independently
        data = cp.asarray([0.25, 1.0], dtype=cp.float32)
        # We can't easily call op_instance compute logic without running the graph,
        # but we can verify the function it wraps works as expected.
        fn = FILTERS["gamma_compression"]
        out = fn(data, gamma=2.0)
        expected = cp.asarray([0.5, 1.0], dtype=cp.float32)
        self.assertTrue(cp.allclose(out, expected))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
