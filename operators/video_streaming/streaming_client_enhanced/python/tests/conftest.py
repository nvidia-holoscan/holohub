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
Pytest fixtures for StreamingClientOp Python binding tests.
"""

import os
import sys

import pytest


@pytest.fixture(scope="session")
def holoscan_modules():
    """Import and provide Holoscan SDK modules."""
    try:
        from holoscan.core import Application, Fragment, Operator, OperatorSpec
        return {
            'Application': Application,
            'Fragment': Fragment,
            'Operator': Operator,
            'OperatorSpec': OperatorSpec,
        }
    except ImportError as e:
        pytest.skip(f"Holoscan SDK not available: {e}")


@pytest.fixture(scope="session")
def streaming_client_module():
    """Import the streaming_client_enhanced Python module."""
    try:
        # Try multiple possible paths for the Python module
        possible_paths = [
            '/workspace/holohub/build-video_streaming/python/lib',
            os.path.join(os.path.dirname(__file__), '../../../..', 'build-video_streaming', 'python', 'lib'),
            os.path.join(os.path.dirname(__file__), '../../build/python/lib'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and path not in sys.path:
                sys.path.insert(0, path)
        
        import holohub.streaming_client_enhanced as sc_module
        return sc_module
    except ImportError as e:
        pytest.skip(f"streaming_client_enhanced module not available: {e}")


@pytest.fixture(scope="session")
def streaming_client_op_class(streaming_client_module):
    """Provide the StreamingClientOp class."""
    try:
        return streaming_client_module.StreamingClientOp
    except AttributeError as e:
        pytest.skip(f"StreamingClientOp class not found: {e}")


@pytest.fixture
def fragment(holoscan_modules):
    """Create a Holoscan Fragment for testing."""
    Fragment = holoscan_modules['Fragment']
    return Fragment()


@pytest.fixture
def operator_factory(streaming_client_op_class, fragment):
    """Factory fixture for creating StreamingClientOp instances."""
    def _create_operator(
        name="test_client",
        width=640,
        height=480,
        fps=30,
        server_ip="127.0.0.1",
        signaling_port=48010,
        send_frames=False,  # Disabled for unit testing
        receive_frames=False,  # Disabled for unit testing
        min_non_zero_bytes=100
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
            min_non_zero_bytes=min_non_zero_bytes
        )
    
    return _create_operator


@pytest.fixture
def default_operator(operator_factory):
    """Create a StreamingClientOp with default parameters."""
    return operator_factory()

