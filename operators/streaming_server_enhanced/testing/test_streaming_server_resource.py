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

"""Unit tests for StreamingServerResource."""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from test_utils import (
    MockStreamingServerResource,
    create_test_frame_sequence,
    assert_frame_properties,
    get_test_configuration
)
from mock_holoscan_framework import MockStreamingServer, MockFrame


class TestStreamingServerResource:
    """Test suite for StreamingServerResource."""

    @pytest.fixture
    def mock_server(self):
        """Provide a mock streaming server."""
        return MockStreamingServer()

    @pytest.fixture
    def default_config(self):
        """Provide default configuration."""
        return get_test_configuration("default")

    @pytest.fixture
    def streaming_server_resource(self, mock_server, default_config):
        """Provide a mock streaming server resource."""
        return MockStreamingServerResource(mock_server, default_config)

    @pytest.mark.unit
    def test_resource_initialization(self, streaming_server_resource, default_config):
        """Test StreamingServerResource initialization."""
        assert streaming_server_resource.config == default_config
        assert streaming_server_resource.server is not None
        assert hasattr(streaming_server_resource, '_event_callbacks')

    @pytest.mark.unit
    def test_resource_setup(self, streaming_server_resource):
        """Test StreamingServerResource setup method."""
        mock_spec = Mock()
        streaming_server_resource.setup(mock_spec)
        # Setup should complete without errors
        assert True

    @pytest.mark.unit
    def test_server_lifecycle(self, streaming_server_resource):
        """Test server start/stop lifecycle."""
        # Initially not running
        assert not streaming_server_resource.is_running()
        
        # Start server
        streaming_server_resource.start()
        assert streaming_server_resource.is_running()
        
        # Stop server
        streaming_server_resource.stop()
        assert not streaming_server_resource.is_running()

    @pytest.mark.unit
    def test_client_connection_status(self, streaming_server_resource):
        """Test client connection status monitoring."""
        # Initially no clients
        assert not streaming_server_resource.has_connected_clients()
        
        # Simulate client connection
        streaming_server_resource.server.simulate_client_connection()
        assert streaming_server_resource.has_connected_clients()
        
        # Simulate client disconnection
        streaming_server_resource.server.simulate_client_disconnection()
        assert not streaming_server_resource.has_connected_clients()

    @pytest.mark.unit
    def test_frame_sending(self, streaming_server_resource):
        """Test frame sending functionality."""
        frame = MockFrame(854, 480, 3)
        frame.data = np.random.randint(0, 255, (480, 854, 3), dtype=np.uint8)
        
        streaming_server_resource.send_frame(frame)
        
        # Verify frame was sent to mock server
        assert len(streaming_server_resource.server.sent_frames) == 1
        sent_frame = streaming_server_resource.server.sent_frames[0]
        assert_frame_properties(sent_frame, 854, 480, 3)

    @pytest.mark.unit
    def test_frame_receiving(self, streaming_server_resource):
        """Test frame receiving functionality."""
        # Add a frame to be received
        test_frame = MockFrame(854, 480, 3)
        test_frame.data = np.random.randint(0, 255, (480, 854, 3), dtype=np.uint8)
        streaming_server_resource.server.add_mock_received_frame(test_frame)
        
        # Receive the frame
        received_frame = streaming_server_resource.receive_frame()
        
        assert received_frame is not None
        assert_frame_properties(received_frame, 854, 480, 3)
        np.testing.assert_array_equal(received_frame.data, test_frame.data)

    @pytest.mark.unit
    def test_try_receive_frame(self, streaming_server_resource):
        """Test try_receive_frame functionality."""
        # Test with no available frames
        empty_frame = MockFrame(854, 480, 3)
        result = streaming_server_resource.try_receive_frame(empty_frame)
        assert not result
        
        # Add a frame and try again
        test_frame = MockFrame(854, 480, 3)
        test_frame.data = np.random.randint(0, 255, (480, 854, 3), dtype=np.uint8)
        streaming_server_resource.server.add_mock_received_frame(test_frame)
        
        result_frame = MockFrame(854, 480, 3)
        result = streaming_server_resource.try_receive_frame(result_frame)
        
        assert result
        assert_frame_properties(result_frame, 854, 480, 3)

    @pytest.mark.unit
    def test_event_callback_registration(self, streaming_server_resource):
        """Test event callback registration."""
        callback_called = []
        
        def test_callback(event):
            callback_called.append(event)
        
        streaming_server_resource.set_event_callback(test_callback)
        
        # Simulate an event
        streaming_server_resource.simulate_event("TEST_EVENT", {"data": "test"})
        
        assert len(callback_called) == 1
        assert callback_called[0].type == "TEST_EVENT"
        assert callback_called[0].data == {"data": "test"}

    @pytest.mark.unit
    def test_multiple_event_callbacks(self, streaming_server_resource):
        """Test multiple event callback registration."""
        callback1_calls = []
        callback2_calls = []
        
        def callback1(event):
            callback1_calls.append(event)
        
        def callback2(event):
            callback2_calls.append(event)
        
        streaming_server_resource.set_event_callback(callback1)
        streaming_server_resource.set_event_callback(callback2)
        
        # Simulate an event
        streaming_server_resource.simulate_event("MULTI_EVENT")
        
        assert len(callback1_calls) == 1
        assert len(callback2_calls) == 1

    @pytest.mark.unit
    def test_configuration_access(self, streaming_server_resource, default_config):
        """Test configuration access."""
        config = streaming_server_resource.get_config()
        assert config == default_config
        assert config["width"] == 854
        assert config["height"] == 480
        assert config["fps"] == 30

    @pytest.mark.unit
    @pytest.mark.parametrize("config_type", ["default", "high_res", "low_res", "upstream_only", "downstream_only"])
    def test_different_configurations(self, mock_server, config_type):
        """Test resource with different configurations."""
        config = get_test_configuration(config_type)
        resource = MockStreamingServerResource(mock_server, config)
        
        assert resource.get_config() == config
        assert resource.get_config()["width"] == config["width"]
        assert resource.get_config()["height"] == config["height"]

    @pytest.mark.unit
    def test_frame_sequence_processing(self, streaming_server_resource):
        """Test processing a sequence of frames."""
        frames = create_test_frame_sequence(5, 854, 480, "gradient")
        
        # Add frames for receiving
        for frame in frames:
            streaming_server_resource.server.add_mock_received_frame(frame)
        
        # Receive all frames
        received_frames = []
        for _ in range(len(frames)):
            frame = streaming_server_resource.receive_frame()
            if frame:
                received_frames.append(frame)
        
        assert len(received_frames) == len(frames)
        
        # Send all frames back
        for frame in received_frames:
            streaming_server_resource.send_frame(frame)
        
        assert len(streaming_server_resource.server.sent_frames) == len(frames)

    @pytest.mark.unit
    def test_server_state_consistency(self, streaming_server_resource):
        """Test that server state remains consistent."""
        # Test multiple start/stop cycles
        for i in range(3):
            assert not streaming_server_resource.is_running()
            streaming_server_resource.start()
            assert streaming_server_resource.is_running()
            streaming_server_resource.stop()
            assert not streaming_server_resource.is_running()

    @pytest.mark.unit
    def test_frame_data_integrity(self, streaming_server_resource):
        """Test that frame data integrity is maintained."""
        # Create frame with specific pattern
        original_frame = MockFrame(854, 480, 3)
        pattern_data = np.zeros((480, 854, 3), dtype=np.uint8)
        pattern_data[:, :, 0] = 255  # Red channel
        pattern_data[100:200, 100:200, 1] = 255  # Green square
        original_frame.data = pattern_data
        
        # Send and receive the frame
        streaming_server_resource.send_frame(original_frame)
        streaming_server_resource.server.add_mock_received_frame(original_frame)
        received_frame = streaming_server_resource.receive_frame()
        
        # Verify data integrity
        np.testing.assert_array_equal(received_frame.data, original_frame.data)
        assert received_frame.width == original_frame.width
        assert received_frame.height == original_frame.height

    @pytest.mark.unit
    def test_empty_frame_handling(self, streaming_server_resource):
        """Test handling of empty frames."""
        # Test receiving when no frames available
        frame = streaming_server_resource.receive_frame()
        assert frame is None
        
        # Test try_receive with no frames
        empty_frame = MockFrame()
        result = streaming_server_resource.try_receive_frame(empty_frame)
        assert not result

    @pytest.mark.unit 
    def test_error_handling_in_callbacks(self, streaming_server_resource):
        """Test error handling in event callbacks."""
        def failing_callback(event):
            raise Exception("Test callback error")
        
        def working_callback(event):
            working_callback.called = True
        
        working_callback.called = False
        
        streaming_server_resource.set_event_callback(failing_callback)
        streaming_server_resource.set_event_callback(working_callback)
        
        # Simulate event - should not crash despite failing callback
        streaming_server_resource.simulate_event("ERROR_TEST")
        
        # Working callback should still be called
        assert working_callback.called

    @pytest.mark.unit
    def test_resource_cleanup(self, streaming_server_resource):
        """Test resource cleanup behavior."""
        # Start server and set callbacks
        streaming_server_resource.start()
        streaming_server_resource.set_event_callback(lambda e: None)
        
        # Should be able to stop cleanly
        streaming_server_resource.stop()
        assert not streaming_server_resource.is_running()
        
        # Should be able to restart
        streaming_server_resource.start()
        assert streaming_server_resource.is_running()

    @pytest.mark.unit
    @pytest.mark.parametrize("frame_count", [1, 5, 10, 50])
    def test_high_throughput_frames(self, streaming_server_resource, frame_count):
        """Test high throughput frame processing."""
        frames = create_test_frame_sequence(frame_count, 854, 480, "checkerboard")
        
        # Add all frames for receiving
        for frame in frames:
            streaming_server_resource.server.add_mock_received_frame(frame)
        
        # Process all frames
        processed_count = 0
        for _ in range(frame_count):
            frame = streaming_server_resource.receive_frame()
            if frame:
                streaming_server_resource.send_frame(frame)
                processed_count += 1
        
        assert processed_count == frame_count
        assert len(streaming_server_resource.server.sent_frames) == frame_count

    @pytest.mark.unit
    def test_concurrent_operations(self, streaming_server_resource):
        """Test concurrent send/receive operations."""
        # Create frames for both sending and receiving
        send_frames = create_test_frame_sequence(3, 854, 480, "solid")
        receive_frames = create_test_frame_sequence(3, 854, 480, "noise")
        
        # Add frames for receiving
        for frame in receive_frames:
            streaming_server_resource.server.add_mock_received_frame(frame)
        
        # Interleave send and receive operations
        for i in range(3):
            # Send a frame
            streaming_server_resource.send_frame(send_frames[i])
            
            # Receive a frame
            received = streaming_server_resource.receive_frame()
            assert received is not None
        
        assert len(streaming_server_resource.server.sent_frames) == 3
