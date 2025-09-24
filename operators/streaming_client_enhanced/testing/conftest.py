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
pytest configuration for StreamingClientOp Python bindings unit tests.

This file provides fixtures and configuration for testing the pybind11 Python
bindings of the StreamingClientOp operator in isolation.
"""

import os
import sys
import pytest
import numpy as np
from typing import Dict, Any, Optional
from unittest.mock import Mock, MagicMock

# Add the build directory to the Python path for imports
import pathlib
current_dir = pathlib.Path(__file__).parent.absolute()
build_dir = current_dir.parent.parent.parent.parent / "build"
if build_dir.exists():
    sys.path.insert(0, str(build_dir / "python" / "lib"))

def pytest_addoption(parser):
    """Add command line options to pytest."""
    parser.addoption(
        "--build-dir",
        action="store",
        default=str(build_dir),
        help="Build directory containing compiled Python bindings",
    )
    parser.addoption(
        "--skip-hardware-tests",
        action="store_true",
        default=False,
        help="Skip tests that require actual hardware connections",
    )


@pytest.fixture(scope="session")
def build_directory(request):
    """Return the build directory containing Python bindings."""
    return request.config.getoption("--build-dir")


@pytest.fixture(scope="session") 
def skip_hardware_tests(request):
    """Return whether to skip hardware-dependent tests."""
    return request.config.getoption("--skip-hardware-tests")


@pytest.fixture(scope="session")
def holoscan_modules(build_directory):
    """Import and return Holoscan core modules."""
    try:
        # Ensure the build directory is in the path
        if build_directory not in sys.path:
            sys.path.insert(0, os.path.join(build_directory, "python", "lib"))
        
        from holoscan.core import Application, Fragment, Operator
        from holoscan.resources import UnboundedAllocator
        
        return {
            'Application': Application,
            'Fragment': Fragment, 
            'Operator': Operator,
            'UnboundedAllocator': UnboundedAllocator
        }
    except ImportError as e:
        pytest.skip(f"Cannot import Holoscan modules: {e}")


@pytest.fixture(scope="session")
def streaming_client_op_class(holoscan_modules):
    """Import and return the StreamingClientOp class using app1_testing pattern."""
    # Use the same robust import pattern that worked in app1_testing
    try:
        from holohub.streaming_client import StreamingClientOp
        return StreamingClientOp
    except ImportError:
        try:
            # Try alternative import path
            from holohub.streaming_client_enhanced import StreamingClientOp
            return StreamingClientOp
        except ImportError as e:
            pytest.skip(f"Cannot import StreamingClientOp: {e}. Make sure the operator is built with Python bindings.")


@pytest.fixture
def mock_fragment(holoscan_modules):
    """Create a mock Holoscan Fragment for testing."""
    Fragment = holoscan_modules['Fragment']
    fragment = Fragment()
    return fragment


@pytest.fixture
def default_operator_params():
    """Return default parameters for StreamingClientOp creation."""
    return {
        'width': 640,
        'height': 480,
        'fps': 30,
        'server_ip': '127.0.0.1',
        'signaling_port': 48010,
        'receive_frames': False,
        'send_frames': False,
        'min_non_zero_bytes': 100,
        'name': 'test_streaming_client'
    }


@pytest.fixture
def test_video_params():
    """Return various video parameter combinations for parametrized tests."""
    return [
        {'width': 640, 'height': 480, 'fps': 30},    # Standard definition
        {'width': 1280, 'height': 720, 'fps': 60},   # HD 60fps  
        {'width': 1920, 'height': 1080, 'fps': 30},  # Full HD
        {'width': 3840, 'height': 2160, 'fps': 24},  # 4K Cinema
    ]


@pytest.fixture
def network_configs():
    """Return various network configuration combinations."""
    return [
        {'server_ip': '127.0.0.1', 'signaling_port': 48010},
        {'server_ip': '192.168.1.100', 'signaling_port': 8080},
        {'server_ip': '10.0.0.1', 'signaling_port': 9999},
    ]


@pytest.fixture
def mock_frame_data():
    """Generate mock frame data for testing."""
    def _generate_frame(width=640, height=480, channels=4, dtype=np.uint8):
        """Generate synthetic frame data."""
        # Create a simple gradient pattern
        frame = np.zeros((height, width, channels), dtype=dtype)
        
        # Create a gradient pattern (BGRA format)
        for y in range(height):
            for x in range(width):
                frame[y, x, 0] = (x * 255) // width      # Blue channel
                frame[y, x, 1] = (y * 255) // height     # Green channel  
                frame[y, x, 2] = ((x + y) * 255) // (width + height)  # Red channel
                frame[y, x, 3] = 255                     # Alpha channel (fully opaque)
        
        return frame
    
    return _generate_frame


@pytest.fixture
def mock_holoscan_framework():
    """Create a complete mock Holoscan framework for testing."""
    from .mock_holoscan_framework import MockHoloscanFramework
    return MockHoloscanFramework()


@pytest.fixture
def bgr_test_data():
    """Create BGR test data for video frame testing."""
    from .mock_holoscan_framework import create_test_bgr_frame
    
    return {
        'gradient': create_test_bgr_frame(640, 480, 'gradient'),
        'solid': create_test_bgr_frame(640, 480, 'solid'),
        'checkerboard': create_test_bgr_frame(640, 480, 'checkerboard'),
        'noise': create_test_bgr_frame(640, 480, 'noise'),
        'empty': create_test_bgr_frame(640, 480, 'empty'),
        'hd_gradient': create_test_bgr_frame(1280, 720, 'gradient'),
        'minimal': create_minimal_bgr_frame(640, 480)
    }


def create_minimal_bgr_frame(width: int, height: int, non_zero_pixels: int = 20) -> np.ndarray:
    """Create BGR frame with minimal content for validation testing."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add minimal non-zero content
    for i in range(min(non_zero_pixels, width * height)):
        y = i // width
        x = i % width
        frame[y, x] = [128, 128, 128]  # Gray pixel
    
    return frame


@pytest.fixture
def mock_streaming_client():
    """Create a mock StreamingClient for testing without actual network connections."""
    mock_client = MagicMock()
    
    # Mock basic streaming state
    mock_client.isStreaming.return_value = False
    mock_client.isUpstreamReady.return_value = False
    mock_client.startStreaming.return_value = None
    mock_client.stopStreaming.return_value = None
    mock_client.sendFrame.return_value = None
    mock_client.setFrameReceivedCallback.return_value = None
    
    # Mock connection establishment
    def mock_start_streaming(server_ip, port):
        mock_client.isStreaming.return_value = True
        mock_client.isUpstreamReady.return_value = True
    
    mock_client.startStreaming.side_effect = mock_start_streaming
    
    return mock_client


@pytest.fixture
def operator_factory(mock_fragment, streaming_client_op_class, default_operator_params):
    """Factory fixture to create StreamingClientOp instances with custom parameters."""
    def _create_operator(**kwargs):
        # Merge with defaults
        params = {**default_operator_params, **kwargs}
        
        # Create operator instance
        return streaming_client_op_class(
            mock_fragment,
            **params
        )
    
    return _create_operator


@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for tests."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


# Pytest markers for test categorization
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test (testing individual components in isolation)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test (testing component interactions)"
    )
    config.addinivalue_line(
        "markers", "hardware: mark test as requiring actual hardware (network connections, etc.)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running (> 5 seconds)"
    )
    config.addinivalue_line(
        "markers", "parametrized: mark test as parametrized (multiple parameter combinations)"
    )


def pytest_collection_modifyitems(config, items):
    """Modify collected test items based on command line options."""
    if config.getoption("--skip-hardware-tests"):
        skip_hardware = pytest.mark.skip(reason="Skipping hardware tests (--skip-hardware-tests option)")
        for item in items:
            if "hardware" in item.keywords:
                item.add_marker(skip_hardware)
