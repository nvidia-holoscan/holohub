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

import pytest
import sys
import os


@pytest.fixture(scope="session")
def holoscan_modules():
    """Import and provide Holoscan SDK modules."""
    try:
        from holoscan.core import Application, Fragment, Operator, OperatorSpec, Resource
        from holoscan.resources import Allocator
        return {
            'Application': Application,
            'Fragment': Fragment,
            'Operator': Operator,
            'OperatorSpec': OperatorSpec,
            'Resource': Resource,
            'Allocator': Allocator,
        }
    except ImportError as e:
        pytest.skip(f"Holoscan SDK not available: {e}")


@pytest.fixture(scope="session")
def streaming_server_module():
    """Import the streaming_server_enhanced Python module."""
    try:
        # Add the build directory to path if needed
        build_path = os.path.join(os.path.dirname(__file__), '../../build')
        if os.path.exists(build_path) and build_path not in sys.path:
            sys.path.insert(0, build_path)
        
        import holohub.streaming_server_enhanced as ss_module
        return ss_module
    except ImportError as e:
        pytest.skip(f"streaming_server_enhanced module not available: {e}")


@pytest.fixture(scope="session")
def streaming_server_classes(streaming_server_module):
    """Provide the StreamingServer operator classes."""
    try:
        return {
            'Resource': streaming_server_module.StreamingServerResource,
            'Upstream': streaming_server_module.StreamingServerUpstreamOp,
            'Downstream': streaming_server_module.StreamingServerDownstreamOp,
        }
    except AttributeError as e:
        pytest.skip(f"StreamingServer classes not found: {e}")


@pytest.fixture
def fragment(holoscan_modules):
    """Create a Holoscan Fragment for testing."""
    Fragment = holoscan_modules['Fragment']
    return Fragment()


@pytest.fixture
def resource_factory(streaming_server_classes, fragment):
    """Factory fixture for creating StreamingServerResource instances."""
    ResourceClass = streaming_server_classes['Resource']
    
    def _create_resource(
        name="test_server_resource",
        port=48010,
        width=854,
        height=480,
        fps=30,
        enable_upstream=True,
        enable_downstream=True,
        server_name="TestServer"
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
            server_name=server_name
        )
    
    return _create_resource


@pytest.fixture
def default_resource(resource_factory):
    """Create a StreamingServerResource with default parameters."""
    return resource_factory()


@pytest.fixture
def upstream_operator_factory(streaming_server_classes, fragment):
    """Factory fixture for creating StreamingServerUpstreamOp instances."""
    UpstreamClass = streaming_server_classes['Upstream']
    
    def _create_operator(name="test_upstream", resource=None):
        if resource is None:
            pytest.skip("Resource required for operator creation")
        return UpstreamClass(fragment, name=name, streaming_server_resource=resource)
    
    return _create_operator


@pytest.fixture
def downstream_operator_factory(streaming_server_classes, fragment):
    """Factory fixture for creating StreamingServerDownstreamOp instances."""
    DownstreamClass = streaming_server_classes['Downstream']
    
    def _create_operator(name="test_downstream", resource=None):
        if resource is None:
            pytest.skip("Resource required for operator creation")
        return DownstreamClass(fragment, name=name, streaming_server_resource=resource)
    
    return _create_operator

