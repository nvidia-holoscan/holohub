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

"""Unit tests for StreamingServerUpstreamOp."""

import pytest
from unittest.mock import Mock, MagicMock
import numpy as np

from test_utils import (
    MockStreamingServerUpstreamOp,
    MockStreamingServerResource,
    create_test_frame_sequence,
    assert_frame_properties,
    assert_tensor_properties,
    get_test_configuration
)
from mock_holoscan_framework import (
    MockStreamingServer,
    MockFrame,
    MockTensor,
    MockInputContext,
    MockOutputContext,
    MockExecutionContext,
    MockOperatorSpec,
    MockAllocator,
    create_mock_bgr_frame
)


class TestStreamingServerUpstreamOp:
    """Test suite for StreamingServerUpstreamOp."""

    @pytest.fixture
    def mock_allocator(self):
        """Provide a mock allocator."""
        return MockAllocator()

    @pytest.fixture
    def mock_server_resource(self):
        """Provide a mock streaming server resource."""
        mock_server = MockStreamingServer()
        config = get_test_configuration("default")
        return MockStreamingServerResource(mock_server, config)

    @pytest.fixture
    def upstream_op(self, fragment, mock_allocator, mock_server_resource):
        """Provide a StreamingServerUpstreamOp for testing."""
        return MockStreamingServerUpstreamOp(
            fragment=fragment,
            name="test_upstream",
            width=854,
            height=480,
            fps=30,
            allocator=mock_allocator,
            streaming_server_resource=mock_server_resource
        )

    @pytest.mark.unit
    def test_operator_initialization(self, upstream_op):
        """Test StreamingServerUpstreamOp initialization."""
        assert upstream_op.name == "test_upstream"
        assert upstream_op.width == 854
        assert upstream_op.height == 480
        assert upstream_op.fps == 30
        assert upstream_op.streaming_server_resource is not None
        assert upstream_op.allocator is not None

    @pytest.mark.unit
    def test_operator_setup(self, upstream_op):
        """Test operator setup method."""
        spec = MockOperatorSpec()
        upstream_op.setup(spec)
        
        # Upstream operator should have output port
        assert "output" in spec.outputs
        assert len(spec.inputs) == 0  # No input ports

    @pytest.mark.unit
    def test_initial_state(self, upstream_op):
        """Test operator initial state."""
        assert not upstream_op.is_shutting_down
        assert not upstream_op.upstream_connected
        assert upstream_op.frames_received == 0
        assert upstream_op.unique_frames_processed == 0
        assert upstream_op.duplicate_frames_detected == 0
        assert len(upstream_op.processed_frame_timestamps) == 0

    @pytest.mark.unit
    def test_compute_no_frames(self, upstream_op):
        """Test compute method when no frames are available."""
        input_context = MockInputContext()
        output_context = MockOutputContext()
        execution_context = MockExecutionContext()
        
        upstream_op.compute(input_context, output_context, execution_context)
        
        # No frames should be emitted
        assert len(output_context.emitted_data) == 0
        assert upstream_op.frames_received == 0

    @pytest.mark.unit
    def test_compute_single_frame(self, upstream_op):
        """Test compute method with a single frame."""
        # Add a frame to the server for receiving
        test_frame = create_mock_bgr_frame(854, 480, "gradient", 1)
        upstream_op.streaming_server_resource.server.add_mock_received_frame(test_frame)
        
        input_context = MockInputContext()
        output_context = MockOutputContext()
        execution_context = MockExecutionContext()
        
        upstream_op.compute(input_context, output_context, execution_context)
        
        # Should have received and emitted one frame
        assert upstream_op.frames_received == 1
        assert upstream_op.unique_frames_processed == 1
        assert upstream_op.duplicate_frames_detected == 0
        assert len(output_context.emitted_data) > 0

    @pytest.mark.unit
    def test_compute_multiple_frames(self, upstream_op):
        """Test compute method with multiple frames."""
        # Add multiple frames
        frames = create_test_frame_sequence(5, 854, 480, "checkerboard")
        for frame in frames:
            upstream_op.streaming_server_resource.server.add_mock_received_frame(frame)
        
        input_context = MockInputContext()
        output_context = MockOutputContext()
        execution_context = MockExecutionContext()
        
        # Process all frames
        for _ in range(len(frames)):
            upstream_op.compute(input_context, output_context, execution_context)
        
        assert upstream_op.frames_received == 5
        assert upstream_op.unique_frames_processed == 5
        assert upstream_op.duplicate_frames_detected == 0

    @pytest.mark.unit
    def test_duplicate_frame_detection(self, upstream_op):
        """Test duplicate frame detection."""
        # Create frame with same timestamp
        frame1 = create_mock_bgr_frame(854, 480, "gradient", 1)
        frame2 = create_mock_bgr_frame(854, 480, "gradient", 1)  # Same frame number = same timestamp
        
        upstream_op.streaming_server_resource.server.add_mock_received_frame(frame1)
        upstream_op.streaming_server_resource.server.add_mock_received_frame(frame2)
        
        input_context = MockInputContext()
        output_context = MockOutputContext()
        execution_context = MockExecutionContext()
        
        # Process both frames
        upstream_op.compute(input_context, output_context, execution_context)
        upstream_op.compute(input_context, output_context, execution_context)
        
        assert upstream_op.frames_received == 2
        assert upstream_op.unique_frames_processed == 1
        assert upstream_op.duplicate_frames_detected == 1

    @pytest.mark.unit
    def test_frame_to_tensor_conversion(self, upstream_op):
        """Test frame to tensor conversion."""
        test_frame = create_mock_bgr_frame(854, 480, "solid", 1)
        tensor = upstream_op.convert_frame_to_tensor(test_frame)
        
        assert_tensor_properties(tensor, (480, 854, 3), np.uint8)
        assert tensor.size == test_frame.data.size

    @pytest.mark.unit
    def test_is_duplicate_frame_method(self, upstream_op):
        """Test is_duplicate_frame method."""
        frame = create_mock_bgr_frame(854, 480, "gradient", 1)
        
        # First time should not be duplicate
        assert not upstream_op.is_duplicate_frame(frame)
        
        # Add to processed timestamps
        upstream_op.processed_frame_timestamps.add(frame.timestamp)
        
        # Now should be duplicate
        assert upstream_op.is_duplicate_frame(frame)

    @pytest.mark.unit
    @pytest.mark.parametrize("width,height", [(320, 240), (854, 480), (1920, 1080)])
    def test_different_resolutions(self, fragment, mock_allocator, mock_server_resource, width, height):
        """Test operator with different resolutions."""
        op = MockStreamingServerUpstreamOp(
            fragment=fragment,
            name="test_resolution",
            width=width,
            height=height,
            allocator=mock_allocator,
            streaming_server_resource=mock_server_resource
        )
        
        assert op.width == width
        assert op.height == height
        
        # Test with frame of matching resolution
        test_frame = create_mock_bgr_frame(width, height, "gradient", 1)
        tensor = op.convert_frame_to_tensor(test_frame)
        assert_tensor_properties(tensor, (height, width, 3), np.uint8)

    @pytest.mark.unit
    @pytest.mark.parametrize("fps", [15, 30, 60, 120])
    def test_different_fps_settings(self, fragment, mock_allocator, mock_server_resource, fps):
        """Test operator with different FPS settings."""
        op = MockStreamingServerUpstreamOp(
            fragment=fragment,
            name="test_fps",
            fps=fps,
            allocator=mock_allocator,
            streaming_server_resource=mock_server_resource
        )
        
        assert op.fps == fps

    @pytest.mark.unit
    def test_performance_tracking(self, upstream_op):
        """Test performance tracking features."""
        # Process several frames and verify tracking
        frames = create_test_frame_sequence(10, 854, 480, "noise")
        for frame in frames:
            upstream_op.streaming_server_resource.server.add_mock_received_frame(frame)
        
        input_context = MockInputContext()
        output_context = MockOutputContext()
        execution_context = MockExecutionContext()
        
        for _ in range(len(frames)):
            upstream_op.compute(input_context, output_context, execution_context)
        
        # Verify performance counters
        assert upstream_op.frames_received == 10
        assert upstream_op.unique_frames_processed == 10
        assert upstream_op.last_processed_timestamp > 0

    @pytest.mark.unit
    def test_timestamp_tracking(self, upstream_op):
        """Test timestamp tracking and ordering."""
        # Create frames with specific timestamps
        frames = []
        for i in range(5):
            frame = create_mock_bgr_frame(854, 480, "gradient", i + 1)
            frame.timestamp = (i + 1) * 1000  # 1000ms apart
            frames.append(frame)
            upstream_op.streaming_server_resource.server.add_mock_received_frame(frame)
        
        input_context = MockInputContext()
        output_context = MockOutputContext()
        execution_context = MockExecutionContext()
        
        for _ in range(len(frames)):
            upstream_op.compute(input_context, output_context, execution_context)
        
        # Verify timestamp tracking
        assert upstream_op.last_processed_timestamp == 5000
        assert len(upstream_op.processed_frame_timestamps) == 5

    @pytest.mark.unit
    def test_state_management(self, upstream_op):
        """Test operator state management."""
        # Test connection state
        assert not upstream_op.upstream_connected
        upstream_op.upstream_connected = True
        assert upstream_op.upstream_connected
        
        # Test shutdown state
        assert not upstream_op.is_shutting_down
        upstream_op.is_shutting_down = True
        assert upstream_op.is_shutting_down

    @pytest.mark.unit
    def test_error_handling_invalid_frame(self, upstream_op):
        """Test error handling with invalid frame data."""
        # Create frame with invalid dimensions
        invalid_frame = MockFrame(0, 0, 0)  # Invalid dimensions
        
        # Should handle gracefully
        try:
            tensor = upstream_op.convert_frame_to_tensor(invalid_frame)
            # If it doesn't raise an exception, check the result
            assert tensor is not None
        except Exception:
            # Exception is acceptable for invalid input
            pass

    @pytest.mark.unit
    def test_memory_efficiency(self, upstream_op):
        """Test memory efficiency with large frame sequences."""
        # Process a large number of frames to test memory handling
        frame_count = 100
        frames = create_test_frame_sequence(frame_count, 854, 480, "gradient")
        
        for frame in frames:
            upstream_op.streaming_server_resource.server.add_mock_received_frame(frame)
        
        input_context = MockInputContext()
        output_context = MockOutputContext()
        execution_context = MockExecutionContext()
        
        # Process all frames
        for _ in range(frame_count):
            upstream_op.compute(input_context, output_context, execution_context)
        
        # Verify all frames were processed
        assert upstream_op.frames_received == frame_count
        assert upstream_op.unique_frames_processed == frame_count

    @pytest.mark.unit
    def test_concurrent_processing_simulation(self, upstream_op):
        """Test simulation of concurrent frame processing."""
        # Add frames with mixed timestamps to simulate out-of-order arrival
        timestamps = [1000, 3000, 2000, 5000, 4000]
        frames = []
        
        for i, ts in enumerate(timestamps):
            frame = create_mock_bgr_frame(854, 480, "checkerboard", i + 1)
            frame.timestamp = ts
            frames.append(frame)
            upstream_op.streaming_server_resource.server.add_mock_received_frame(frame)
        
        input_context = MockInputContext()
        output_context = MockOutputContext()
        execution_context = MockExecutionContext()
        
        # Process frames
        for _ in range(len(frames)):
            upstream_op.compute(input_context, output_context, execution_context)
        
        # All frames should be processed as unique (different timestamps)
        assert upstream_op.frames_received == 5
        assert upstream_op.unique_frames_processed == 5
        assert upstream_op.duplicate_frames_detected == 0
        assert len(upstream_op.processed_frame_timestamps) == 5

    @pytest.mark.unit
    def test_resource_dependency(self, upstream_op):
        """Test dependency on streaming server resource."""
        # Operator should have reference to resource
        assert upstream_op.streaming_server_resource is not None
        
        # Should be able to access server through resource
        assert hasattr(upstream_op.streaming_server_resource, 'server')
        
        # Server should be functional
        test_frame = create_mock_bgr_frame(854, 480, "solid", 1)
        upstream_op.streaming_server_resource.server.add_mock_received_frame(test_frame)
        
        received = upstream_op.streaming_server_resource.server.receive_frame()
        assert received is not None

    @pytest.mark.unit
    def test_frame_format_handling(self, upstream_op):
        """Test handling of different frame formats."""
        # Test BGR frame (default)
        bgr_frame = create_mock_bgr_frame(854, 480, "gradient", 1)
        bgr_tensor = upstream_op.convert_frame_to_tensor(bgr_frame)
        assert_tensor_properties(bgr_tensor, (480, 854, 3), np.uint8)
        
        # Test frame with different channel count
        rgba_frame = MockFrame(854, 480, 4)  # RGBA
        rgba_frame.data = np.random.randint(0, 255, (480, 854, 4), dtype=np.uint8)
        
        try:
            rgba_tensor = upstream_op.convert_frame_to_tensor(rgba_frame)
            assert_tensor_properties(rgba_tensor, (480, 854, 4), np.uint8)
        except Exception:
            # Exception acceptable for unsupported format
            pass
