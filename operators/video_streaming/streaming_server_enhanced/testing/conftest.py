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

"""Pytest configuration and fixtures for StreamingServerEnhanced testing."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

# Add the current directory to sys.path to enable imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from mock_holoscan_framework import (  # noqa: E402
    MockAllocator,
    MockExecutionContext,
    MockFragment,
    MockInputContext,
    MockOutputContext,
    MockTensor,
    MockTensorMap,
)


@pytest.fixture
def fragment():
    """Provide a mock Holoscan Fragment for testing."""
    return MockFragment()


@pytest.fixture
def mock_allocator():
    """Provide a mock allocator for testing."""
    return MockAllocator()


@pytest.fixture
def mock_tensor():
    """Provide a mock tensor for testing."""
    return MockTensor()


@pytest.fixture
def mock_tensor_map():
    """Provide a mock tensor map for testing."""
    return MockTensorMap()


@pytest.fixture
def mock_input_context(mock_tensor_map):
    """Provide a mock input context for testing."""
    return MockInputContext(mock_tensor_map)


@pytest.fixture
def mock_output_context():
    """Provide a mock output context for testing."""
    return MockOutputContext()


@pytest.fixture
def mock_execution_context():
    """Provide a mock execution context for testing."""
    return MockExecutionContext()


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def golden_frames_dir():
    """Provide the golden frames directory path."""
    testing_dir = Path(__file__).parent
    golden_dir = testing_dir / "golden_frames"
    golden_dir.mkdir(exist_ok=True)
    return golden_dir


@pytest.fixture
def sample_video_config():
    """Provide sample video configuration for testing."""
    return {
        "width": 854,
        "height": 480,
        "fps": 30,
        "port": 48010,
        "server_name": "TestStreamingServer",
        "enable_upstream": True,
        "enable_downstream": True,
        "multi_instance": False,
    }


@pytest.fixture
def mock_streaming_server_config():
    """Provide mock StreamingServer config for testing."""
    mock_config = Mock()
    mock_config.port = 48010
    mock_config.server_name = "TestStreamingServer"
    mock_config.width = 854
    mock_config.height = 480
    mock_config.fps = 30
    mock_config.enable_upstream = True
    mock_config.enable_downstream = True
    mock_config.multi_instance = False
    return mock_config


@pytest.fixture
def mock_streaming_server():
    """Provide a mock StreamingServer for testing."""
    mock_server = Mock()
    mock_server.is_running.return_value = False
    mock_server.has_connected_clients.return_value = False
    mock_server.start.return_value = None
    mock_server.stop.return_value = None
    mock_server.send_frame.return_value = None
    mock_server.receive_frame.return_value = None
    mock_server.try_receive_frame.return_value = False
    return mock_server


@pytest.fixture
def mock_streaming_server_resource(mock_streaming_server, mock_streaming_server_config):
    """Provide a mock StreamingServerResource for testing."""
    from test_utils import MockStreamingServerResource

    return MockStreamingServerResource(mock_streaming_server, mock_streaming_server_config)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests (isolated components)")
    config.addinivalue_line("markers", "functional: Functional tests (component interactions)")
    config.addinivalue_line("markers", "golden_frame: Golden frame visual regression tests")
    config.addinivalue_line("markers", "hardware: Hardware-dependent tests")
    config.addinivalue_line("markers", "slow: Slow tests (> 5 seconds)")
    config.addinivalue_line("markers", "parametrized: Parametrized tests (multiple scenarios)")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add unit marker to all tests by default
        if not any(
            marker.name in ["functional", "golden_frame", "hardware", "slow"]
            for marker in item.iter_markers()
        ):
            item.add_marker(pytest.mark.unit)

        # Add slow marker to tests that typically take longer
        if any(
            keyword in item.name.lower()
            for keyword in ["functional", "pipeline", "server", "network"]
        ):
            item.add_marker(pytest.mark.slow)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--skip-hardware-tests",
        action="store_true",
        default=False,
        help="Skip hardware-dependent tests",
    )
    parser.addoption(
        "--skip-slow-tests",
        action="store_true",
        default=False,
        help="Skip slow tests (> 5 seconds)",
    )


def pytest_runtest_setup(item):
    """Setup function called before each test."""
    # Skip hardware tests if requested
    if item.config.getoption("--skip-hardware-tests") and any(
        marker.name == "hardware" for marker in item.iter_markers()
    ):
        pytest.skip("Skipping hardware-dependent test")

    # Skip slow tests if requested
    if item.config.getoption("--skip-slow-tests") and any(
        marker.name == "slow" for marker in item.iter_markers()
    ):
        pytest.skip("Skipping slow test")
