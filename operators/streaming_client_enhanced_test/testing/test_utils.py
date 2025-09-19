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
Test utilities for StreamingClientOp Python binding tests.

This module provides helper functions and utilities for testing the
StreamingClientOp Python bindings, including mock data generation,
validation helpers, and test configuration utilities.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from unittest.mock import MagicMock
import logging

logger = logging.getLogger(__name__)


class FrameDataGenerator:
    """Utility class for generating synthetic frame data for testing."""
    
    @staticmethod
    def create_solid_color_frame(width: int, height: int, 
                                color: Tuple[int, int, int, int] = (255, 0, 0, 255),
                                dtype: np.dtype = np.uint8) -> np.ndarray:
        """
        Create a solid color frame.
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels  
            color: BGRA color tuple (Blue, Green, Red, Alpha)
            dtype: Data type for the frame
            
        Returns:
            NumPy array representing the frame data
        """
        frame = np.full((height, width, 4), color, dtype=dtype)
        return frame
    
    @staticmethod
    def create_gradient_frame(width: int, height: int, 
                             dtype: np.dtype = np.uint8) -> np.ndarray:
        """
        Create a gradient frame for visual testing.
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            dtype: Data type for the frame
            
        Returns:
            NumPy array with gradient pattern
        """
        frame = np.zeros((height, width, 4), dtype=dtype)
        
        # Create gradient patterns
        for y in range(height):
            for x in range(width):
                frame[y, x, 0] = (x * 255) // width      # Blue gradient (horizontal)
                frame[y, x, 1] = (y * 255) // height     # Green gradient (vertical)
                frame[y, x, 2] = ((x + y) * 255) // (width + height)  # Red diagonal
                frame[y, x, 3] = 255                     # Full alpha
        
        return frame
    
    @staticmethod
    def create_checkerboard_frame(width: int, height: int,
                                 square_size: int = 32,
                                 color1: Tuple[int, int, int, int] = (255, 255, 255, 255),
                                 color2: Tuple[int, int, int, int] = (0, 0, 0, 255),
                                 dtype: np.dtype = np.uint8) -> np.ndarray:
        """
        Create a checkerboard pattern frame.
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            square_size: Size of each checkerboard square
            color1: First color (BGRA)
            color2: Second color (BGRA)
            dtype: Data type for the frame
            
        Returns:
            NumPy array with checkerboard pattern
        """
        frame = np.zeros((height, width, 4), dtype=dtype)
        
        for y in range(height):
            for x in range(width):
                # Determine which color to use based on checkerboard pattern
                square_x = x // square_size
                square_y = y // square_size
                color = color1 if (square_x + square_y) % 2 == 0 else color2
                frame[y, x] = color
        
        return frame
    
    @staticmethod
    def create_noise_frame(width: int, height: int,
                          noise_level: float = 0.1,
                          base_color: Tuple[int, int, int, int] = (128, 128, 128, 255),
                          dtype: np.dtype = np.uint8) -> np.ndarray:
        """
        Create a frame with random noise.
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            noise_level: Amount of noise (0.0 to 1.0)
            base_color: Base color before adding noise
            dtype: Data type for the frame
            
        Returns:
            NumPy array with noisy frame data
        """
        frame = np.full((height, width, 4), base_color, dtype=dtype)
        
        # Add random noise
        noise = np.random.randint(-int(255 * noise_level), int(255 * noise_level),
                                 size=(height, width, 3), dtype=np.int16)
        
        # Apply noise to RGB channels (keep alpha unchanged)
        frame[:, :, :3] = np.clip(frame[:, :, :3].astype(np.int16) + noise, 0, 255)
        
        return frame.astype(dtype)


class ParameterValidator:
    """Utility class for validating operator parameters."""
    
    @staticmethod
    def validate_video_params(width: int, height: int, fps: int) -> Dict[str, bool]:
        """
        Validate video parameters.
        
        Args:
            width: Video width
            height: Video height
            fps: Frame rate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'width_valid': 0 < width <= 7680,      # Up to 8K width
            'height_valid': 0 < height <= 4320,    # Up to 8K height  
            'fps_valid': 0 < fps <= 120,           # Reasonable FPS range
            'aspect_ratio_reasonable': 0.1 <= (width / height) <= 10.0
        }
        results['all_valid'] = all(results.values())
        return results
    
    @staticmethod
    def validate_network_params(server_ip: str, port: int) -> Dict[str, bool]:
        """
        Validate network parameters.
        
        Args:
            server_ip: Server IP address
            port: Port number
            
        Returns:
            Dictionary with validation results
        """
        import ipaddress
        
        results = {
            'port_valid': 1 <= port <= 65535,
            'ip_format_valid': False
        }
        
        try:
            ipaddress.ip_address(server_ip)
            results['ip_format_valid'] = True
        except ValueError:
            results['ip_format_valid'] = False
        
        results['all_valid'] = all(results.values())
        return results


class MockStreamingClientFactory:
    """Factory for creating mock StreamingClient objects."""
    
    @staticmethod
    def create_successful_client() -> MagicMock:
        """Create a mock client that simulates successful operations."""
        mock_client = MagicMock()
        
        # Set up successful behaviors
        mock_client.isStreaming.return_value = True
        mock_client.isUpstreamReady.return_value = True
        mock_client.startStreaming.return_value = None
        mock_client.stopStreaming.return_value = None
        mock_client.sendFrame.return_value = None
        mock_client.setFrameReceivedCallback.return_value = None
        
        return mock_client
    
    @staticmethod
    def create_failing_client() -> MagicMock:
        """Create a mock client that simulates failures."""
        mock_client = MagicMock()
        
        # Set up failing behaviors
        mock_client.isStreaming.return_value = False
        mock_client.isUpstreamReady.return_value = False
        mock_client.startStreaming.side_effect = RuntimeError("Connection failed")
        mock_client.stopStreaming.return_value = None
        mock_client.sendFrame.side_effect = RuntimeError("Send failed")
        mock_client.setFrameReceivedCallback.return_value = None
        
        return mock_client
    
    @staticmethod
    def create_slow_client(delay_seconds: float = 1.0) -> MagicMock:
        """Create a mock client that simulates slow operations."""
        import time
        mock_client = MagicMock()
        
        def slow_start_streaming(server_ip, port):
            time.sleep(delay_seconds)
            mock_client.isStreaming.return_value = True
            mock_client.isUpstreamReady.return_value = True
        
        mock_client.isStreaming.return_value = False
        mock_client.isUpstreamReady.return_value = False
        mock_client.startStreaming.side_effect = slow_start_streaming
        mock_client.stopStreaming.return_value = None
        mock_client.sendFrame.return_value = None
        mock_client.setFrameReceivedCallback.return_value = None
        
        return mock_client


class TestDataSets:
    """Predefined test data sets for parametrized testing."""
    
    # Standard video resolutions
    VIDEO_RESOLUTIONS = [
        (640, 480, 30),      # VGA
        (800, 600, 30),      # SVGA
        (1024, 768, 30),     # XGA
        (1280, 720, 60),     # HD 720p
        (1920, 1080, 30),    # Full HD 1080p
        (2560, 1440, 30),    # QHD 1440p
        (3840, 2160, 24),    # 4K UHD
    ]
    
    # Network configurations
    NETWORK_CONFIGS = [
        ("127.0.0.1", 48010),         # Localhost default
        ("192.168.1.100", 8080),      # Private network
        ("10.0.0.1", 9999),           # Different private range
        ("172.16.0.1", 12345),        # Another private range
    ]
    
    # Streaming modes
    STREAMING_MODES = [
        (True, False),   # Receive only
        (False, True),   # Send only
        (True, True),    # Bidirectional
        (False, False),  # Configuration only
    ]
    
    # Invalid parameter combinations for negative testing
    INVALID_VIDEO_PARAMS = [
        (-1, 480, 30),      # Negative width
        (640, -1, 30),      # Negative height
        (640, 480, -1),     # Negative FPS
        (0, 480, 30),       # Zero width
        (640, 0, 30),       # Zero height
        (640, 480, 0),      # Zero FPS
        (10000, 480, 30),   # Unreasonably large width
        (640, 10000, 30),   # Unreasonably large height
        (640, 480, 1000),   # Unreasonably high FPS
    ]
    
    INVALID_NETWORK_PARAMS = [
        ("", 48010),            # Empty IP
        ("127.0.0.1", -1),      # Negative port
        ("127.0.0.1", 0),       # Zero port
        ("127.0.0.1", 70000),   # Port too high
        ("999.999.999.999", 48010),  # Invalid IP format
        ("not_an_ip", 48010),   # Non-IP string
    ]


class TestAssertions:
    """Custom assertion helpers for StreamingClientOp testing."""
    
    @staticmethod
    def assert_frame_properties(frame_data: np.ndarray, 
                               expected_width: int, 
                               expected_height: int,
                               expected_channels: int = 4):
        """
        Assert that frame data has expected properties.
        
        Args:
            frame_data: Frame data array
            expected_width: Expected width
            expected_height: Expected height
            expected_channels: Expected number of channels
        """
        assert frame_data is not None, "Frame data should not be None"
        assert isinstance(frame_data, np.ndarray), "Frame data should be NumPy array"
        
        expected_shape = (expected_height, expected_width, expected_channels)
        assert frame_data.shape == expected_shape, \
            f"Frame shape {frame_data.shape} != expected {expected_shape}"
        
        assert frame_data.dtype in [np.uint8, np.uint16, np.float32], \
            f"Unsupported frame data type: {frame_data.dtype}"
    
    @staticmethod
    def assert_operator_properties(operator, expected_params: Dict[str, Any]):
        """
        Assert that operator has expected properties.
        
        Args:
            operator: StreamingClientOp instance
            expected_params: Expected parameter values
        """
        assert operator is not None, "Operator should not be None"
        
        # Check that required methods exist
        required_methods = ['initialize', 'setup']
        for method in required_methods:
            assert hasattr(operator, method), f"Operator missing method: {method}"
            assert callable(getattr(operator, method)), f"Method {method} not callable"
    
    @staticmethod
    def assert_parameter_validation(validation_results: Dict[str, bool], 
                                  should_be_valid: bool = True):
        """
        Assert parameter validation results.
        
        Args:
            validation_results: Results from parameter validation
            should_be_valid: Whether parameters should be valid
        """
        if should_be_valid:
            assert validation_results.get('all_valid', False), \
                f"Parameters should be valid but validation failed: {validation_results}"
        else:
            assert not validation_results.get('all_valid', True), \
                f"Parameters should be invalid but validation passed: {validation_results}"


def log_test_info(test_name: str, parameters: Dict[str, Any]):
    """
    Log test information for debugging.
    
    Args:
        test_name: Name of the test
        parameters: Test parameters
    """
    logger.info(f"Running test: {test_name}")
    for key, value in parameters.items():
        logger.debug(f"  {key}: {value}")


def cleanup_test_resources(*resources):
    """
    Clean up test resources.
    
    Args:
        *resources: Resources to clean up
    """
    for resource in resources:
        if resource is not None:
            try:
                if hasattr(resource, 'cleanup'):
                    resource.cleanup()
                elif hasattr(resource, 'close'):
                    resource.close()
                del resource
            except Exception as e:
                logger.warning(f"Failed to cleanup resource: {e}")
