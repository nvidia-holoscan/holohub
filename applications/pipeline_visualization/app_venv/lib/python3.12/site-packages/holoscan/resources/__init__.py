# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""This module provides a Python API to underlying C++ API Resources.

.. autosummary::

    holoscan.resources.Allocator
    holoscan.resources.AsyncBufferReceiver
    holoscan.resources.AsyncBufferTransmitter
    holoscan.resources.BlockMemoryPool
    holoscan.resources.CudaAllocator
    holoscan.resources.CudaGreenContext
    holoscan.resources.CudaGreenContextPool
    holoscan.resources.CudaStreamPool
    holoscan.resources.DoubleBufferReceiver
    holoscan.resources.DoubleBufferTransmitter
    holoscan.resources.GXFClock
    holoscan.resources.GXFComponentResource
    holoscan.resources.ManualClock
    holoscan.resources.MemoryStorageType
    holoscan.resources.OrConditionCombiner
    holoscan.resources.RealtimeClock
    holoscan.resources.Receiver
    holoscan.resources.RMMAllocator
    holoscan.resources.SchedulingPolicy
    holoscan.resources.SerializationBuffer
    holoscan.resources.StdComponentSerializer
    holoscan.resources.StdEntitySerializer
    holoscan.resources.StreamOrderedAllocator
    holoscan.resources.SyntheticClock
    holoscan.resources.ThreadPool
    holoscan.resources.Transmitter
    holoscan.resources.UnboundedAllocator
    holoscan.resources.UcxComponentSerializer
    holoscan.resources.UcxEntitySerializer
    holoscan.resources.UcxHoloscanComponentSerializer
    holoscan.resources.UcxReceiver
    holoscan.resources.UcxSerializationBuffer
    holoscan.resources.UcxTransmitter
"""

from ._resources import (
    Allocator,
    AsyncBufferReceiver,
    AsyncBufferTransmitter,
    BlockMemoryPool,
    CudaAllocator,
    CudaGreenContext,
    CudaGreenContextPool,
    CudaStreamPool,
    DoubleBufferReceiver,
    DoubleBufferTransmitter,
    GXFClock,
    ManualClock,
    MemoryStorageType,
    OrConditionCombiner,
    RealtimeClock,
    Receiver,
    RMMAllocator,
    SchedulingPolicy,
    SerializationBuffer,
    StdComponentSerializer,
    StdEntitySerializer,
    StreamOrderedAllocator,
    SyntheticClock,
    ThreadPool,
    Transmitter,
    UcxComponentSerializer,
    UcxEntitySerializer,
    UcxHoloscanComponentSerializer,
    UcxReceiver,
    UcxSerializationBuffer,
    UcxTransmitter,
    UnboundedAllocator,
)
from ._resources import (
    GXFComponentResource as _GXFComponentResource,
)

__all__ = [
    "Allocator",
    "AsyncBufferReceiver",
    "AsyncBufferTransmitter",
    "BlockMemoryPool",
    "CudaAllocator",
    "CudaGreenContext",
    "CudaGreenContextPool",
    "CudaStreamPool",
    "DoubleBufferReceiver",
    "DoubleBufferTransmitter",
    "GXFClock",
    "GXFComponentResource",
    "ManualClock",
    "MemoryStorageType",
    "OrConditionCombiner",
    "RealtimeClock",
    "Receiver",
    "RMMAllocator",
    "SchedulingPolicy",
    "SerializationBuffer",
    "StdComponentSerializer",
    "StdEntitySerializer",
    "StreamOrderedAllocator",
    "SyntheticClock",
    "ThreadPool",
    "Transmitter",
    "UcxComponentSerializer",
    "UcxEntitySerializer",
    "UcxHoloscanComponentSerializer",
    "UcxReceiver",
    "UcxSerializationBuffer",
    "UcxTransmitter",
    "UnboundedAllocator",
]


class GXFComponentResource(_GXFComponentResource):
    def __setattr__(self, name, value):
        readonly_attributes = [
            "fragment",
            "gxf_typename",
            "conditions",
            "resources",
            "operator_type",
            "description",
        ]
        if name in readonly_attributes:
            raise AttributeError(f'cannot override read-only property "{name}"')
        super().__setattr__(name, value)

    def __init__(self, fragment, *args, **kwargs):
        from holoscan.core import ComponentSpec, _Fragment  # noqa: PLC0415

        if not isinstance(fragment, _Fragment):
            raise ValueError(
                "The first argument to an GXFComponentResource's constructor must be the Fragment "
                "(Application) to which it belongs."
            )
        # It is recommended to not use super()
        # (https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python)
        _GXFComponentResource.__init__(self, self, fragment, *args, **kwargs)
        # Create a PyGXFComponentResourceSpec object and pass it to the C++ API
        spec = ComponentSpec(fragment=self.fragment, component=self)
        self.spec = spec
        # Call setup method in the derived class
        self.setup(spec)

    def setup(self, spec):
        # This method is invoked by the derived class to set up the resource.
        super().setup(spec)

    def initialize(self):
        # Place holder for initialize method
        pass


# copy docstrings defined in core_pydoc.hpp
GXFComponentResource.__doc__ = _GXFComponentResource.__doc__
GXFComponentResource.__init__.__doc__ = _GXFComponentResource.__init__.__doc__
