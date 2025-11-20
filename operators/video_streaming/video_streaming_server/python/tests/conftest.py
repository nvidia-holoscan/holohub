#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Pytest fixtures for StreamingServer operators Python binding tests.
"""

import os
import sys

import cupy as cp
import numpy as np
import pytest
from holoscan.core import Application, Fragment


@pytest.fixture(scope="session")
def streaming_server_module():
    """Import the video_streaming_server Python module."""
    try:
        # Try multiple possible paths for the Python module
        possible_paths = [
            "/workspace/holohub/build-video_streaming/python/lib",
            os.path.join(
                os.path.dirname(__file__), "../../../..", "build-video_streaming", "python", "lib"
            ),
            os.path.join(os.path.dirname(__file__), "../../build/python/lib"),
        ]

        for path in possible_paths:
            if os.path.exists(path) and path not in sys.path:
                sys.path.insert(0, path)

        import holohub.video_streaming_server as ss_module

        return ss_module
    except ImportError as e:
        pytest.skip(f"video_streaming_server module not available: {e}")


@pytest.fixture(scope="session")
def streaming_server_classes(streaming_server_module):
    """Provide the StreamingServer operator classes."""
    try:
        return {
            "Resource": streaming_server_module.StreamingServerResource,
            "Upstream": streaming_server_module.StreamingServerUpstreamOp,
            "Downstream": streaming_server_module.StreamingServerDownstreamOp,
        }
    except AttributeError as e:
        pytest.skip(f"StreamingServer classes not found: {e}")


@pytest.fixture
def resource_factory(streaming_server_classes, fragment):
    """Factory fixture for creating StreamingServerResource instances."""
    ResourceClass = streaming_server_classes["Resource"]

    def _create_resource(
        name="test_server_resource",
        port=48010,
        width=854,
        height=480,
        fps=30,
        enable_upstream=True,
        enable_downstream=True,
        server_name="TestServer",
        fragment=fragment,  # Allow override but default to fixture fragment
    ):
        return ResourceClass(
            fragment,
            name=name,
            port=port,
            width=width,
            height=height,
            fps=fps,
            enable_upstream=enable_upstream,
            enable_downstream=enable_downstream,
            server_name=server_name,
        )

    return _create_resource


@pytest.fixture
def default_resource(resource_factory):
    """Create a StreamingServerResource with default parameters."""
    return resource_factory()


@pytest.fixture
def upstream_operator_factory(streaming_server_classes, fragment):
    """Factory fixture for creating StreamingServerUpstreamOp instances."""
    UpstreamClass = streaming_server_classes["Upstream"]

    def _create_operator(name="test_upstream", resource=None):
        if resource is None:
            pytest.skip("Resource required for operator creation")
        return UpstreamClass(fragment, name=name, video_streaming_server_resource=resource)

    return _create_operator


@pytest.fixture
def downstream_operator_factory(streaming_server_classes, fragment):
    """Factory fixture for creating StreamingServerDownstreamOp instances."""
    DownstreamClass = streaming_server_classes["Downstream"]

    def _create_operator(name="test_downstream", resource=None):
        if resource is None:
            pytest.skip("Resource required for operator creation")
        return DownstreamClass(fragment, name=name, video_streaming_server_resource=resource)

    return _create_operator


# ============================================================================
# Common fixtures (copied from root conftest.py due to --confcutdir isolation)
# ============================================================================


@pytest.fixture
def app():
    """Provide a Holoscan Application instance."""
    return Application()


@pytest.fixture
def fragment():
    """Provide a Holoscan Fragment instance."""
    return Fragment()


@pytest.fixture
def mock_image():
    """Factory fixture for creating mock image tensors."""

    def _factory(shape, dtype=cp.uint8, backend="cupy", seed=None):
        if backend == "cupy":
            xp = cp
        elif backend == "numpy":
            xp = np
        else:
            raise ValueError(f"Unknown backend: {backend}")
        rng = xp.random.default_rng(seed)
        dtype = xp.dtype(dtype)
        if dtype.kind in "ui":
            img = rng.integers(0, 256, size=shape, dtype=dtype, endpoint=False)
        elif dtype.kind == "f":
            img = rng.uniform(0.0, 1.0, size=shape)
            if img.dtype != dtype:
                img = img.astype(dtype, copy=False)
        else:
            raise ValueError(f"{dtype=} unsupported")
        return img

    return _factory


class MockOpInput:
    """Mock operator input for testing compute() methods."""

    def __init__(self, tensor, tensor_name="", port=""):
        self._tensor = tensor
        self._tensor_name = tensor_name
        self._port = port

    def receive(self, port):
        assert port == self._port
        return {self._tensor_name: self._tensor}


class MockOpOutput:
    """Mock operator output for testing compute() methods."""

    def __init__(self):
        self.emitted = None

    def emit(self, msg, port):
        self.emitted = (msg, port)


@pytest.fixture
def op_input_factory():
    """Factory fixture for creating mock operator inputs."""

    def _factory(tensor, tensor_name="", port=""):
        return MockOpInput(tensor, tensor_name=tensor_name, port=port)

    return _factory


@pytest.fixture
def op_output():
    """Provide a mock operator output."""
    return MockOpOutput()


@pytest.fixture
def execution_context():
    """Provide a mock execution context (None for Python bindings)."""
    return None
