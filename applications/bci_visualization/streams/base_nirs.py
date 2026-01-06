"""
SPDX-FileCopyrightText: Copyright (c) 2026 Kernel.
SPDX-License-Identifier: Apache-2.0
"""

import abc
from typing import Iterator, NamedTuple

import numpy as np
from numpy.typing import NDArray


class ChannelInfo(NamedTuple):
    detector_module: NDArray[np.int_]
    detector_number: NDArray[np.int_]
    source_module: NDArray[np.int_]
    source_number: NDArray[np.int_]

    def __len__(self) -> int:
        return len(self.source_module)


class BaseNirsStream(abc.ABC):
    def start(self) -> None:
        pass

    @abc.abstractmethod
    def get_channels(self) -> ChannelInfo:
        pass

    @abc.abstractmethod
    def stream_nirs(self) -> Iterator[np.ndarray]:
        pass
