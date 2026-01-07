#!/usr/bin/env python3
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

"""
Pytest fixtures for StreamingClientOp Python binding tests.
"""

import os
import sys

import cupy as cp
import numpy as np
import pytest
from holoscan.core import Application, Fragment


@pytest.fixture(scope="session")
def streaming_client_module():
    """Import the video_streaming_client Python module."""
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

        import holohub.video_streaming_client as sc_module

        return sc_module
    except ImportError as e:
        pytest.skip(f"video_streaming_client module not available: {e}")


@pytest.fixture(scope="session")
def streaming_client_op_class(streaming_client_module):
    """Provide the VideoStreamingClientOp class."""
    try:
        return streaming_client_module.VideoStreamingClientOp
    except AttributeError as e:
        pytest.skip(f"VideoStreamingClientOp class not found: {e}")


@pytest.fixture
def operator_factory(streaming_client_op_class, fragment):
    """Factory fixture for creating VideoStreamingClientOp instances."""

    def _create_operator(
        name="test_client",
        width=640,
        height=480,
        fps=30,
        server_ip="127.0.0.1",
        signaling_port=48010,
        send_frames=False,  # Disabled for unit testing
        receive_frames=False,  # Disabled for unit testing
        min_non_zero_bytes=100,
    ):
        return streaming_client_op_class(
            fragment,
            name=name,
            width=width,
            height=height,
            fps=fps,
            server_ip=server_ip,
            signaling_port=signaling_port,
            send_frames=send_frames,
            receive_frames=receive_frames,
            min_non_zero_bytes=min_non_zero_bytes,
        )

    return _create_operator


@pytest.fixture
def default_operator(operator_factory):
    """Create a VideoStreamingClientOp with default parameters."""
    return operator_factory()


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
