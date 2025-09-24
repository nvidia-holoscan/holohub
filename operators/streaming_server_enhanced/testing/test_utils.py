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

"""Test utilities and helper functions for StreamingServer testing."""

import numpy as np
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, Optional, List, Union

from mock_holoscan_framework import (
    MockHoloscanOperator,
    MockHoloscanResource,
    MockStreamingServer,
    MockFrame,
    MockTensor,
    create_mock_bgr_frame,
    create_mock_tensor_from_frame,
    create_mock_frame_from_tensor,
)


class MockStreamingServerUpstreamOp(MockHoloscanOperator):
    """Mock StreamingServerUpstreamOp for testing."""
    
    def __init__(self, fragment=None, name="streaming_server_upstream", **kwargs):
        super().__init__(fragment, name, **kwargs)
        self.width = kwargs.get("width", 854)
        self.height = kwargs.get("height", 480)
        self.fps = kwargs.get("fps", 30)
        self.allocator = kwargs.get("allocator", Mock())
        self.streaming_server_resource = kwargs.get("streaming_server_resource", Mock())
        
        # Internal state
        self.is_shutting_down = False
        self.upstream_connected = False
        self.frames_received = 0
        self.start_time_ticks = 0
        self.last_processed_timestamp = 0
        self.duplicate_frames_detected = 0
        self.unique_frames_processed = 0
        self.processed_frame_timestamps = set()
        
    def setup(self, spec):
        """Mock setup method."""
        spec.output("output")
        
    def compute(self, op_input, op_output, context):
        """Mock compute method that simulates receiving frames."""
        if self.streaming_server_resource and hasattr(self.streaming_server_resource, 'server'):
            frame = self.streaming_server_resource.server.receive_frame()
            if frame:
                tensor = create_mock_tensor_from_frame(frame)
                op_output.emit(tensor, "output")
                self.frames_received += 1
                
                # Simulate duplicate detection
                if frame.timestamp not in self.processed_frame_timestamps:
                    self.processed_frame_timestamps.add(frame.timestamp)
                    self.unique_frames_processed += 1
                else:
                    self.duplicate_frames_detected += 1
                    
                self.last_processed_timestamp = frame.timestamp
                
    def is_duplicate_frame(self, frame):
        """Mock duplicate frame detection."""
        return frame.timestamp in self.processed_frame_timestamps
        
    def convert_frame_to_tensor(self, frame):
        """Mock frame to tensor conversion."""
        return create_mock_tensor_from_frame(frame)


class MockStreamingServerDownstreamOp(MockHoloscanOperator):
    """Mock StreamingServerDownstreamOp for testing."""
    
    def __init__(self, fragment=None, name="streaming_server_downstream", **kwargs):
        super().__init__(fragment, name, **kwargs)
        self.width = kwargs.get("width", 854)
        self.height = kwargs.get("height", 480)
        self.fps = kwargs.get("fps", 30)
        self.enable_processing = kwargs.get("enable_processing", True)
        self.processing_type = kwargs.get("processing_type", "mirror")
        self.allocator = kwargs.get("allocator", Mock())
        self.streaming_server_resource = kwargs.get("streaming_server_resource", Mock())
        
        # Internal state
        self.is_shutting_down = False
        self.downstream_connected = False
        self.frames_processed = 0
        self.frames_sent = 0
        self.start_time_ticks = 0
        
    def setup(self, spec):
        """Mock setup method."""
        spec.input("input")
        
    def compute(self, op_input, op_output, context):
        """Mock compute method that simulates processing and sending frames."""
        tensor_map = op_input.receive()
        if tensor_map and "input" in tensor_map:
            input_tensor = tensor_map["input"]
            
            # Process the frame if processing is enabled
            if self.enable_processing:
                processed_tensor = self.process_frame(input_tensor)
            else:
                processed_tensor = input_tensor
                
            # Convert to frame and send
            frame = self.convert_tensor_to_frame(processed_tensor)
            if self.streaming_server_resource and hasattr(self.streaming_server_resource, 'server'):
                self.streaming_server_resource.server.send_frame(frame)
                self.frames_sent += 1
                
            self.frames_processed += 1
            
    def process_frame(self, input_tensor):
        """Mock frame processing."""
        if self.processing_type == "mirror":
            return self.mirror_horizontally(input_tensor)
        return input_tensor
        
    def mirror_horizontally(self, input_tensor):
        """Mock horizontal mirroring."""
        data = np.array(input_tensor.data)
        mirrored_data = np.fliplr(data)
        return MockTensor(data=mirrored_data)
        
    def convert_tensor_to_frame(self, tensor):
        """Mock tensor to frame conversion."""
        return create_mock_frame_from_tensor(tensor)


class MockStreamingServerResource(MockHoloscanResource):
    """Mock StreamingServerResource for testing."""
    
    def __init__(self, server=None, config=None, fragment=None, name="streaming_server_resource", **kwargs):
        super().__init__(fragment, name, **kwargs)
        self.server = server or MockStreamingServer(config)
        self.config = config or {}
        self._event_callbacks = []
        
    def setup(self, spec):
        """Mock setup method."""
        pass
        
    def start(self):
        """Mock start method."""
        self.server.start()
        
    def stop(self):
        """Mock stop method."""
        self.server.stop()
        
    def is_running(self):
        """Check if server is running."""
        return self.server.is_running()
        
    def has_connected_clients(self):
        """Check if clients are connected."""
        return self.server.has_connected_clients()
        
    def send_frame(self, frame):
        """Send frame through server."""
        self.server.send_frame(frame)
        
    def receive_frame(self):
        """Receive frame from server."""
        return self.server.receive_frame()
        
    def try_receive_frame(self, frame):
        """Try to receive frame from server."""
        return self.server.try_receive_frame(frame)
        
    def get_config(self):
        """Get server configuration."""
        return self.config
        
    def set_event_callback(self, callback):
        """Set event callback."""
        self._event_callbacks.append(callback)
        self.server.set_event_callback(callback)
        
    def simulate_event(self, event_type, event_data=None):
        """Simulate an event for testing."""
        event = Mock()
        event.type = event_type
        event.data = event_data or {}
        
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                print(f"Error in event callback: {e}")


def create_test_frame_sequence(count=10, width=854, height=480, pattern="gradient"):
    """Create a sequence of test frames for testing."""
    frames = []
    for i in range(count):
        frame = create_mock_bgr_frame(width, height, pattern, i + 1)
        frames.append(frame)
    return frames


def create_test_tensor_sequence(count=10, width=854, height=480, pattern="gradient"):
    """Create a sequence of test tensors for testing."""
    frames = create_test_frame_sequence(count, width, height, pattern)
    tensors = [create_mock_tensor_from_frame(frame) for frame in frames]
    return tensors


def assert_frame_properties(frame, expected_width=854, expected_height=480, expected_channels=3):
    """Assert frame has expected properties."""
    assert frame.width == expected_width, f"Expected width {expected_width}, got {frame.width}"
    assert frame.height == expected_height, f"Expected height {expected_height}, got {frame.height}"
    assert frame.channels == expected_channels, f"Expected channels {expected_channels}, got {frame.channels}"
    assert frame.data.shape == (expected_height, expected_width, expected_channels), \
        f"Expected shape {(expected_height, expected_width, expected_channels)}, got {frame.data.shape}"


def assert_tensor_properties(tensor, expected_shape=(480, 854, 3), expected_dtype=np.uint8):
    """Assert tensor has expected properties."""
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"
    assert tensor.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {tensor.dtype}"


def calculate_frame_difference(frame1, frame2):
    """Calculate pixel-wise difference between two frames."""
    diff = np.abs(frame1.data.astype(np.float32) - frame2.data.astype(np.float32))
    return {
        "mean_diff": np.mean(diff),
        "max_diff": np.max(diff),
        "total_diff": np.sum(diff),
        "mse": np.mean(diff ** 2)
    }


def simulate_streaming_pipeline(upstream_op, downstream_op, frames, process_frames=True):
    """Simulate a complete streaming pipeline for testing."""
    results = {
        "frames_processed": 0,
        "frames_sent": 0,
        "frames_received": 0,
        "processing_errors": [],
        "network_errors": []
    }
    
    try:
        # Add frames to upstream for receiving
        if hasattr(upstream_op.streaming_server_resource, 'server'):
            for frame in frames:
                upstream_op.streaming_server_resource.server.add_mock_received_frame(frame)
        
        # Simulate upstream receiving and downstream processing
        for _ in range(len(frames)):
            try:
                # Upstream receive
                mock_input = Mock()
                mock_output = Mock()
                mock_context = Mock()
                
                upstream_op.compute(mock_input, mock_output, mock_context)
                results["frames_received"] += 1
                
                if process_frames and mock_output.emitted_data:
                    # Downstream process
                    downstream_input = Mock()
                    downstream_input.receive.return_value = mock_output.emitted_data
                    downstream_output = Mock()
                    
                    downstream_op.compute(downstream_input, downstream_output, mock_context)
                    results["frames_processed"] += 1
                    results["frames_sent"] += 1
                    
            except Exception as e:
                results["processing_errors"].append(str(e))
                
    except Exception as e:
        results["network_errors"].append(str(e))
        
    return results


class FrameValidator:
    """Helper class for validating frame processing results."""
    
    def __init__(self, tolerance=1.0):
        self.tolerance = tolerance
        
    def validate_frame_integrity(self, original_frame, processed_frame):
        """Validate that processed frame maintains integrity."""
        assert_frame_properties(processed_frame, original_frame.width, 
                               original_frame.height, original_frame.channels)
        
        # Check that processing didn't corrupt the frame
        assert processed_frame.data.dtype == original_frame.data.dtype
        assert processed_frame.size == original_frame.size
        
    def validate_mirroring(self, original_frame, mirrored_frame):
        """Validate horizontal mirroring operation."""
        self.validate_frame_integrity(original_frame, mirrored_frame)
        
        # Check that mirroring was applied correctly
        expected_mirrored = np.fliplr(original_frame.data)
        diff = calculate_frame_difference(
            MockFrame(mirrored_frame.width, mirrored_frame.height),
            MockFrame(mirrored_frame.width, mirrored_frame.height)
        )
        
        # Set the data for comparison
        expected_frame = MockFrame(mirrored_frame.width, mirrored_frame.height)
        expected_frame.data = expected_mirrored
        
        actual_diff = calculate_frame_difference(expected_frame, mirrored_frame)
        assert actual_diff["mean_diff"] <= self.tolerance, \
            f"Mirroring validation failed: mean diff {actual_diff['mean_diff']} > tolerance {self.tolerance}"
            
    def validate_processing_chain(self, input_frames, output_frames, processing_type="mirror"):
        """Validate an entire processing chain."""
        assert len(input_frames) == len(output_frames), \
            f"Input/output frame count mismatch: {len(input_frames)} vs {len(output_frames)}"
            
        for i, (input_frame, output_frame) in enumerate(zip(input_frames, output_frames)):
            if processing_type == "mirror":
                self.validate_mirroring(input_frame, output_frame)
            else:
                self.validate_frame_integrity(input_frame, output_frame)


def get_test_configuration(test_type="default"):
    """Get test configuration for different scenarios."""
    configs = {
        "default": {
            "width": 854,
            "height": 480,
            "fps": 30,
            "port": 48010,
            "server_name": "TestStreamingServer",
            "enable_upstream": True,
            "enable_downstream": True,
            "multi_instance": False
        },
        "high_res": {
            "width": 1920,
            "height": 1080,
            "fps": 60,
            "port": 48011,
            "server_name": "HighResTestServer",
            "enable_upstream": True,
            "enable_downstream": True,
            "multi_instance": False
        },
        "low_res": {
            "width": 320,
            "height": 240,
            "fps": 15,
            "port": 48012,
            "server_name": "LowResTestServer",
            "enable_upstream": True,
            "enable_downstream": True,
            "multi_instance": False
        },
        "upstream_only": {
            "width": 854,
            "height": 480,
            "fps": 30,
            "port": 48013,
            "server_name": "UpstreamOnlyServer",
            "enable_upstream": True,
            "enable_downstream": False,
            "multi_instance": False
        },
        "downstream_only": {
            "width": 854,
            "height": 480,
            "fps": 30,
            "port": 48014,
            "server_name": "DownstreamOnlyServer",
            "enable_upstream": False,
            "enable_downstream": True,
            "multi_instance": False
        }
    }
    
    return configs.get(test_type, configs["default"])
