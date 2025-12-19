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

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING

import cupy as cp
from holoscan.core import Operator
from holoscan.decorator import Input, create_op  # type: ignore

from ultra_post.core.display import ensure_rgba
from ultra_post.core.loader import load_uff_frame

if TYPE_CHECKING:  # pragma: no cover - optional dependency
    from ultra_post.sim.raysim_source import RaysimFrameGenerator


@dataclass
class UffSourceConfig:
    path: Path
    dataset: Optional[str] = None
    frame_index: int = 0


class _UffSourceState:
    def __init__(self, config: UffSourceConfig) -> None:
        self.config = config
        self._frame = None

    def next_frame(self) -> cp.ndarray:
        if self._frame is None:
            loaded = load_uff_frame(
                self.config.path, dataset=self.config.dataset, frame_index=self.config.frame_index
            )
            self._frame = loaded["data"]
        return self._frame


def make_uff_source_op(config: UffSourceConfig):
    """Create a HoloScan operator that emits a GPU tensor from a UFF file."""

    state = _UffSourceState(config)

    return create_op(outputs="out")(lambda: state.next_frame())


class FuncOp(Operator):
    """Generic operator that runs a callable on the input."""

    def __init__(self, fragment, *args, fn: Callable, params: dict = None, **kwargs):
        self.fn = fn
        self.params = params or {}
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        data = op_input.receive("in")

        # Holoscan's create_op may deliver a dict when there is a single output;
        # unwrap common single-item containers so the wrapped function gets the tensor.
        if isinstance(data, dict) and len(data) == 1:
            data = next(iter(data.values()))
        elif isinstance(data, (list, tuple)) and len(data) == 1:
            data = data[0]

        op_output.emit(self.fn(data, **self.params), "out")


def make_raysim_source_op(generator: "RaysimFrameGenerator"):
    """Create a HoloScan operator that emits frames from a Raysim generator."""

    return create_op(outputs="out")(lambda: generator.next_frame())


def make_rgba_formatter_op():
    """Convert grayscale/RGB CuPy tensors into RGBA for Holoviz."""

    return create_op(inputs=Input("in", arg_map="in_"), outputs="out", op_param="op")(_to_rgba)


def _to_rgba(in_: object, *, op=None) -> object:
    return cp.ascontiguousarray(ensure_rgba(in_))


__all__ = [
    "UffSourceConfig",
    "make_uff_source_op",
    "FuncOp",
    "make_raysim_source_op",
    "make_rgba_formatter_op",
]
