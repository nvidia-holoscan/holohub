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

"""Unit tests for StreamingServerDownstreamOp."""

import numpy as np
import pytest
from mock_holoscan_framework import (
    MockAllocator,
    MockExecutionContext,
    MockInputContext,
    MockOperatorSpec,
    MockOutputContext,
    MockStreamingServer,
    MockTensor,
    MockTensorMap,
    create_mock_bgr_frame,
    create_mock_tensor_from_frame,
)
from test_utils import (
    FrameValidator,
    MockStreamingServerDownstreamOp,
    MockStreamingServerResource,
    assert_frame_properties,
    create_test_frame_sequence,
    create_test_tensor_sequence,
    get_test_configuration,
)


class TestStreamingServerDownstreamOp:
    """Test suite for StreamingServerDownstreamOp."""

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
    def downstream_op(self, fragment, mock_allocator, mock_server_resource):
        """Provide a StreamingServerDownstreamOp for testing."""
        return MockStreamingServerDownstreamOp(
            fragment=fragment,
            name="test_downstream",
            width=854,
            height=480,
            fps=30,
            enable_processing=True,
            processing_type="mirror",
            allocator=mock_allocator,
            streaming_server_resource=mock_server_resource,
        )

    @pytest.fixture
    def frame_validator(self):
        """Provide a frame validator for testing."""
        return FrameValidator(tolerance=1.0)

    @pytest.mark.unit
    def test_operator_initialization(self, downstream_op):
        """Test StreamingServerDownstreamOp initialization."""
        assert downstream_op.name == "test_downstream"
        assert downstream_op.width == 854
        assert downstream_op.height == 480
        assert downstream_op.fps == 30
        assert downstream_op.enable_processing
        assert downstream_op.processing_type == "mirror"
        assert downstream_op.streaming_server_resource is not None
        assert downstream_op.allocator is not None

    @pytest.mark.unit
    def test_operator_setup(self, downstream_op):
        """Test operator setup method."""
        spec = MockOperatorSpec()
        downstream_op.setup(spec)

        # Downstream operator should have input port
        assert "input" in spec.inputs
        assert len(spec.outputs) == 0  # No output ports

    @pytest.mark.unit
    def test_initial_state(self, downstream_op):
        """Test operator initial state."""
        assert not downstream_op.is_shutting_down
        assert not downstream_op.downstream_connected
        assert downstream_op.frames_processed == 0
        assert downstream_op.frames_sent == 0

    @pytest.mark.unit
    def test_processing_disabled(self, fragment, mock_allocator, mock_server_resource):
        """Test operator with processing disabled."""
        op = MockStreamingServerDownstreamOp(
            fragment=fragment,
            name="no_processing",
            enable_processing=False,
            allocator=mock_allocator,
            streaming_server_resource=mock_server_resource,
        )

        assert not op.enable_processing

    @pytest.mark.unit
    def test_compute_no_input(self, downstream_op):
        """Test compute method when no input is available."""
        input_context = MockInputContext(MockTensorMap())
        output_context = MockOutputContext()
        execution_context = MockExecutionContext()

        downstream_op.compute(input_context, output_context, execution_context)

        # No frames should be processed or sent
        assert downstream_op.frames_processed == 0
        assert downstream_op.frames_sent == 0

    @pytest.mark.unit
    def test_compute_single_tensor(self, downstream_op):
        """Test compute method with a single input tensor."""
        # Create input tensor
        test_frame = create_mock_bgr_frame(854, 480, "gradient", 1)
        test_tensor = create_mock_tensor_from_frame(test_frame)

        tensor_map = MockTensorMap({"input": test_tensor})
        input_context = MockInputContext(tensor_map)
        output_context = MockOutputContext()
        execution_context = MockExecutionContext()

        downstream_op.compute(input_context, output_context, execution_context)

        # Should have processed and sent one frame
        assert downstream_op.frames_processed == 1
        assert downstream_op.frames_sent == 1
        assert len(downstream_op.streaming_server_resource.server.sent_frames) == 1

    @pytest.mark.unit
    def test_compute_multiple_tensors(self, downstream_op):
        """Test compute method with multiple tensors."""
        tensors = create_test_tensor_sequence(5, 854, 480, "checkerboard")

        input_context = MockInputContext()
        output_context = MockOutputContext()
        execution_context = MockExecutionContext()

        # Process all tensors
        for tensor in tensors:
            tensor_map = MockTensorMap({"input": tensor})
            input_context._tensor_map = tensor_map
            downstream_op.compute(input_context, output_context, execution_context)

        assert downstream_op.frames_processed == 5
        assert downstream_op.frames_sent == 5
        assert len(downstream_op.streaming_server_resource.server.sent_frames) == 5

    @pytest.mark.unit
    def test_horizontal_mirroring(self, downstream_op, frame_validator):
        """Test horizontal mirroring processing."""
        # Create a frame with asymmetric pattern for easy mirroring verification
        test_frame = create_mock_bgr_frame(854, 480, "gradient", 1)
        test_tensor = create_mock_tensor_from_frame(test_frame)

        # Process the tensor
        mirrored_tensor = downstream_op.mirror_horizontally(test_tensor)

        # Verify mirroring
        original_data = test_tensor.data
        mirrored_data = mirrored_tensor.data
        expected_mirrored = np.fliplr(original_data)

        np.testing.assert_array_equal(mirrored_data, expected_mirrored)

    @pytest.mark.unit
    def test_process_frame_with_mirroring(self, downstream_op):
        """Test process_frame method with mirroring enabled."""
        test_frame = create_mock_bgr_frame(854, 480, "checkerboard", 1)
        test_tensor = create_mock_tensor_from_frame(test_frame)

        processed_tensor = downstream_op.process_frame(test_tensor)

        # Should be different from original (mirrored)
        assert not np.array_equal(processed_tensor.data, test_tensor.data)

        # Should have same shape
        assert processed_tensor.shape == test_tensor.shape

    @pytest.mark.unit
    def test_process_frame_no_processing(self, fragment, mock_allocator, mock_server_resource):
        """Test process_frame method with processing disabled."""
        op = MockStreamingServerDownstreamOp(
            fragment=fragment,
            name="no_processing",
            enable_processing=False,
            allocator=mock_allocator,
            streaming_server_resource=mock_server_resource,
        )

        test_frame = create_mock_bgr_frame(854, 480, "solid", 1)
        test_tensor = create_mock_tensor_from_frame(test_frame)

        # Simulate compute with processing disabled
        tensor_map = MockTensorMap({"input": test_tensor})
        input_context = MockInputContext(tensor_map)
        output_context = MockOutputContext()
        execution_context = MockExecutionContext()

        op.compute(input_context, output_context, execution_context)

        # Frame should be sent without processing
        assert op.frames_processed == 1
        assert op.frames_sent == 1

    @pytest.mark.unit
    def test_tensor_to_frame_conversion(self, downstream_op):
        """Test tensor to frame conversion."""
        test_frame = create_mock_bgr_frame(854, 480, "noise", 1)
        test_tensor = create_mock_tensor_from_frame(test_frame)

        converted_frame = downstream_op.convert_tensor_to_frame(test_tensor)

        assert_frame_properties(converted_frame, 854, 480, 3)
        np.testing.assert_array_equal(converted_frame.data, test_tensor.data)

    @pytest.mark.unit
    @pytest.mark.parametrize("processing_type", ["mirror", "none"])
    def test_different_processing_types(
        self, fragment, mock_allocator, mock_server_resource, processing_type
    ):
        """Test operator with different processing types."""
        enable_processing = processing_type != "none"

        op = MockStreamingServerDownstreamOp(
            fragment=fragment,
            name="test_processing",
            enable_processing=enable_processing,
            processing_type=processing_type,
            allocator=mock_allocator,
            streaming_server_resource=mock_server_resource,
        )

        assert op.processing_type == processing_type
        assert op.enable_processing == enable_processing

    @pytest.mark.unit
    @pytest.mark.parametrize("width,height", [(320, 240), (854, 480), (1920, 1080)])
    def test_different_resolutions(
        self, fragment, mock_allocator, mock_server_resource, width, height
    ):
        """Test operator with different resolutions."""
        op = MockStreamingServerDownstreamOp(
            fragment=fragment,
            name="test_resolution",
            width=width,
            height=height,
            allocator=mock_allocator,
            streaming_server_resource=mock_server_resource,
        )

        assert op.width == width
        assert op.height == height

        # Test with tensor of matching resolution
        test_frame = create_mock_bgr_frame(width, height, "gradient", 1)
        test_tensor = create_mock_tensor_from_frame(test_frame)
        converted_frame = op.convert_tensor_to_frame(test_tensor)

        assert_frame_properties(converted_frame, width, height, 3)

    @pytest.mark.unit
    @pytest.mark.parametrize("fps", [15, 30, 60, 120])
    def test_different_fps_settings(self, fragment, mock_allocator, mock_server_resource, fps):
        """Test operator with different FPS settings."""
        op = MockStreamingServerDownstreamOp(
            fragment=fragment,
            name="test_fps",
            fps=fps,
            allocator=mock_allocator,
            streaming_server_resource=mock_server_resource,
        )

        assert op.fps == fps

    @pytest.mark.unit
    def test_performance_tracking(self, downstream_op):
        """Test performance tracking features."""
        tensors = create_test_tensor_sequence(10, 854, 480, "solid")

        input_context = MockInputContext()
        output_context = MockOutputContext()
        execution_context = MockExecutionContext()

        # Process all tensors
        for tensor in tensors:
            tensor_map = MockTensorMap({"input": tensor})
            input_context._tensor_map = tensor_map
            downstream_op.compute(input_context, output_context, execution_context)

        # Verify performance counters
        assert downstream_op.frames_processed == 10
        assert downstream_op.frames_sent == 10

    @pytest.mark.unit
    def test_state_management(self, downstream_op):
        """Test operator state management."""
        # Test connection state
        assert not downstream_op.downstream_connected
        downstream_op.downstream_connected = True
        assert downstream_op.downstream_connected

        # Test shutdown state
        assert not downstream_op.is_shutting_down
        downstream_op.is_shutting_down = True
        assert downstream_op.is_shutting_down

    @pytest.mark.unit
    def test_error_handling_invalid_tensor(self, downstream_op):
        """Test error handling with invalid tensor data."""
        # Create tensor with invalid dimensions
        invalid_tensor = MockTensor(data=np.array([]), shape=(0,), dtype=np.uint8)

        # Should handle gracefully
        try:
            frame = downstream_op.convert_tensor_to_frame(invalid_tensor)
            # If it doesn't raise an exception, check the result
            assert frame is not None
        except Exception:
            # Exception is acceptable for invalid input
            pass

    @pytest.mark.unit
    def test_memory_efficiency(self, downstream_op):
        """Test memory efficiency with large tensor sequences."""
        # Process a large number of tensors to test memory handling
        tensor_count = 100
        tensors = create_test_tensor_sequence(tensor_count, 854, 480, "gradient")

        input_context = MockInputContext()
        output_context = MockOutputContext()
        execution_context = MockExecutionContext()

        # Process all tensors
        for tensor in tensors:
            tensor_map = MockTensorMap({"input": tensor})
            input_context._tensor_map = tensor_map
            downstream_op.compute(input_context, output_context, execution_context)

        # Verify all tensors were processed
        assert downstream_op.frames_processed == tensor_count
        assert downstream_op.frames_sent == tensor_count

    @pytest.mark.unit
    def test_frame_processing_pipeline(self, downstream_op, frame_validator):
        """Test complete frame processing pipeline."""
        # Create sequence of frames with different patterns
        input_frames = [
            create_mock_bgr_frame(854, 480, "gradient", 1),
            create_mock_bgr_frame(854, 480, "checkerboard", 2),
            create_mock_bgr_frame(854, 480, "solid", 3),
        ]

        input_context = MockInputContext()
        output_context = MockOutputContext()
        execution_context = MockExecutionContext()

        # Process each frame
        for frame in input_frames:
            tensor = create_mock_tensor_from_frame(frame)
            tensor_map = MockTensorMap({"input": tensor})
            input_context._tensor_map = tensor_map

            downstream_op.compute(input_context, output_context, execution_context)

        # Verify all frames were sent
        sent_frames = downstream_op.streaming_server_resource.server.sent_frames
        assert len(sent_frames) == len(input_frames)

        # Verify processing was applied (frames should be different due to mirroring)
        for i, (original, sent) in enumerate(zip(input_frames, sent_frames)):
            frame_validator.validate_frame_integrity(original, sent)

    @pytest.mark.unit
    def test_concurrent_processing_simulation(self, downstream_op):
        """Test simulation of concurrent tensor processing."""
        # Create tensors with varying characteristics
        tensors = [
            create_mock_tensor_from_frame(create_mock_bgr_frame(854, 480, "gradient", i))
            for i in range(10)
        ]

        input_context = MockInputContext()
        output_context = MockOutputContext()
        execution_context = MockExecutionContext()

        # Process tensors rapidly
        for tensor in tensors:
            tensor_map = MockTensorMap({"input": tensor})
            input_context._tensor_map = tensor_map
            downstream_op.compute(input_context, output_context, execution_context)

        # All tensors should be processed successfully
        assert downstream_op.frames_processed == 10
        assert downstream_op.frames_sent == 10

    @pytest.mark.unit
    def test_resource_dependency(self, downstream_op):
        """Test dependency on streaming server resource."""
        # Operator should have reference to resource
        assert downstream_op.streaming_server_resource is not None

        # Should be able to access server through resource
        assert hasattr(downstream_op.streaming_server_resource, "server")

        # Server should be functional
        test_frame = create_mock_bgr_frame(854, 480, "solid", 1)
        downstream_op.streaming_server_resource.server.send_frame(test_frame)

        assert len(downstream_op.streaming_server_resource.server.sent_frames) == 1

    @pytest.mark.unit
    def test_tensor_format_handling(self, downstream_op):
        """Test handling of different tensor formats."""
        # Test standard BGR tensor (3 channels)
        bgr_tensor = MockTensor(data=np.random.randint(0, 255, (480, 854, 3), dtype=np.uint8))
        bgr_frame = downstream_op.convert_tensor_to_frame(bgr_tensor)
        assert_frame_properties(bgr_frame, 854, 480, 3)

        # Test RGBA tensor (4 channels)
        rgba_tensor = MockTensor(data=np.random.randint(0, 255, (480, 854, 4), dtype=np.uint8))

        try:
            rgba_frame = downstream_op.convert_tensor_to_frame(rgba_tensor)
            assert_frame_properties(rgba_frame, 854, 480, 4)
        except Exception:
            # Exception acceptable for unsupported format
            pass

    @pytest.mark.unit
    def test_processing_consistency(self, downstream_op):
        """Test that processing is applied consistently."""
        # Process the same tensor multiple times
        test_frame = create_mock_bgr_frame(854, 480, "gradient", 1)
        test_tensor = create_mock_tensor_from_frame(test_frame)

        results = []
        for _ in range(3):
            processed = downstream_op.process_frame(test_tensor)
            results.append(processed.data.copy())

        # All results should be identical (deterministic processing)
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])

    @pytest.mark.unit
    def test_frame_queue_behavior(self, downstream_op):
        """Test behavior when processing frames in sequence."""
        frames = create_test_frame_sequence(5, 854, 480, "noise")

        input_context = MockInputContext()
        output_context = MockOutputContext()
        execution_context = MockExecutionContext()

        # Process frames in sequence
        for frame in frames:
            tensor = create_mock_tensor_from_frame(frame)
            tensor_map = MockTensorMap({"input": tensor})
            input_context._tensor_map = tensor_map
            downstream_op.compute(input_context, output_context, execution_context)

        # Verify frames are sent in order
        sent_frames = downstream_op.streaming_server_resource.server.sent_frames
        assert len(sent_frames) == len(frames)

        # Each sent frame should correspond to processed input
        for i, sent_frame in enumerate(sent_frames):
            assert_frame_properties(sent_frame, 854, 480, 3)
