"""
SPDX-FileCopyrightText: Copyright (c) 2026 Kernel.
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

import numpy as np
from holoscan.core import ExecutionContext, InputContext, Operator, OperatorSpec, OutputContext
from streams.base_nirs import BaseNirsStream, ChannelInfo

logger = logging.getLogger(__name__)


class SampleOutput(NamedTuple):
    data: np.ndarray
    channels: ChannelInfo


class StreamOperator(Operator):
    def __init__(
        self,
        stream: BaseNirsStream,
        *,
        fragment: Any | None = None,
    ) -> None:
        super().__init__(fragment, name=self.__class__.__name__)
        self._stream = stream
        self._channels: ChannelInfo

    def setup(self, spec: OperatorSpec) -> None:
        spec.output("samples")

    def start(self) -> None:
        self._stream.start()
        self._channels = self._stream.get_channels()
        self._iter = self._stream.stream_nirs()

    def compute(
        self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext
    ) -> None:

        sample = next(self._iter, None)
        if sample is None:
            raise StopIteration("No more samples available in the stream.")

        op_output.emit(SampleOutput(sample, self._channels), "samples")
