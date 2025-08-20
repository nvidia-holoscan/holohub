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

import pytest
from holoscan.core import Operator

try:
    from holohub.streaming_client_operator import StreamingClientOp
except ImportError:
    pytest.skip("StreamingClientOp not available - skipping tests", allow_module_level=True)

try:
    from holoscan.core import BaseOperator
except ImportError:
    from holoscan.core import _Operator as BaseOperator


class TestStreamingClientOp:
    """Test suite for StreamingClientOp operator."""

    def test_streaming_client_op_init(self, fragment):
        """Test StreamingClientOp initialization and its properties."""
        name = "streaming_client_op"
        op = StreamingClientOp(
            fragment=fragment, 
            name=name,
            width=854,
            height=480,
            fps=30,
            server_ip="127.0.0.1",
            signaling_port=48010
        )
        assert isinstance(op, BaseOperator), "StreamingClientOp should be a Holoscan operator"
        assert op.operator_type == Operator.OperatorType.NATIVE, "Operator type should be NATIVE"
        assert f"name: {name}" in repr(op), "Operator name should appear in repr()"

    def test_streaming_client_op_default_params(self, fragment):
        """Test StreamingClientOp with default parameters."""
        op = StreamingClientOp(fragment=fragment, name="streaming_client_default")
        assert isinstance(op, BaseOperator), "Should create operator with default params"

    @pytest.mark.parametrize("width,height", [
        (640, 480),
        (854, 480), 
        (1920, 1080),
        (1280, 720)
    ])
    def test_streaming_client_op_resolution_params(self, fragment, width, height):
        """Test StreamingClientOp with different video resolutions."""
        op = StreamingClientOp(
            fragment=fragment,
            width=width,
            height=height,
            fps=30,
            name=f"streaming_client_{width}x{height}"
        )
        assert isinstance(op, BaseOperator), f"Should handle {width}x{height} resolution"

    @pytest.mark.parametrize("fps", [15, 24, 30, 60])
    def test_streaming_client_op_fps_params(self, fragment, fps):
        """Test StreamingClientOp with different frame rates."""
        op = StreamingClientOp(
            fragment=fragment,
            width=854,
            height=480,
            fps=fps,
            name=f"streaming_client_{fps}fps"
        )
        assert isinstance(op, BaseOperator), f"Should handle {fps} fps"

    @pytest.mark.parametrize("server_ip", [
        "127.0.0.1",
        "192.168.1.100", 
        "10.0.0.1",
        "localhost"
    ])
    def test_streaming_client_op_server_ip_params(self, fragment, server_ip):
        """Test StreamingClientOp with different server IP addresses."""
        op = StreamingClientOp(
            fragment=fragment,
            server_ip=server_ip,
            signaling_port=48010,
            name="streaming_client_ip_test"
        )
        assert isinstance(op, BaseOperator), f"Should handle server IP: {server_ip}"

    @pytest.mark.parametrize("port", [48010, 8080, 9090, 12345])
    def test_streaming_client_op_port_params(self, fragment, port):
        """Test StreamingClientOp with different signaling ports."""
        op = StreamingClientOp(
            fragment=fragment,
            server_ip="127.0.0.1",
            signaling_port=port,
            name="streaming_client_port_test"
        )
        assert isinstance(op, BaseOperator), f"Should handle port: {port}"

    @pytest.mark.parametrize("receive_frames,send_frames", [
        (True, True),
        (True, False),
        (False, True),
        (False, False)
    ])
    def test_streaming_client_op_frame_flags(self, fragment, receive_frames, send_frames):
        """Test StreamingClientOp with different frame handling flags."""
        op = StreamingClientOp(
            fragment=fragment,
            receive_frames=receive_frames,
            send_frames=send_frames,
            name="streaming_client_flags_test"
        )
        assert isinstance(op, BaseOperator), f"Should handle rx:{receive_frames}, tx:{send_frames}"

    def test_streaming_client_op_setup(self, fragment):
        """Test StreamingClientOp setup for input/output ports."""
        from holoscan.core import OperatorSpec
        
        op = StreamingClientOp(
            fragment=fragment,
            width=854,
            height=480,
            fps=30,
            name="streaming_client_setup_test"
        )
        
        # Create operator spec and call setup
        spec = OperatorSpec(fragment)
        op.setup(spec)
        
        # Note: Port verification would require access to the spec's internal state
        # This test primarily ensures setup() doesn't crash
        assert True, "Setup should complete without errors"

    def test_streaming_client_op_invalid_width(self, fragment):
        """Test StreamingClientOp with invalid width parameter."""
        with pytest.raises((ValueError, TypeError)):
            StreamingClientOp(
                fragment=fragment,
                width=-1,  # Invalid negative width
                height=480,
                name="streaming_client_invalid_width"
            )

    def test_streaming_client_op_invalid_height(self, fragment):
        """Test StreamingClientOp with invalid height parameter."""
        with pytest.raises((ValueError, TypeError)):
            StreamingClientOp(
                fragment=fragment,
                width=854,
                height=0,  # Invalid zero height
                name="streaming_client_invalid_height"
            )

    def test_streaming_client_op_invalid_fps(self, fragment):
        """Test StreamingClientOp with invalid fps parameter."""
        with pytest.raises((ValueError, TypeError)):
            StreamingClientOp(
                fragment=fragment,
                width=854,
                height=480,
                fps=-1,  # Invalid negative fps
                name="streaming_client_invalid_fps"
            )

    def test_streaming_client_op_invalid_port(self, fragment):
        """Test StreamingClientOp with invalid port parameter."""
        with pytest.raises((ValueError, TypeError, OverflowError)):
            StreamingClientOp(
                fragment=fragment,
                signaling_port=99999,  # Invalid port number (too high)
                name="streaming_client_invalid_port"
            )

    def test_streaming_client_op_boundary_values(self, fragment):
        """Test StreamingClientOp with boundary value parameters."""
        # Test minimum reasonable values
        op_min = StreamingClientOp(
            fragment=fragment,
            width=1,
            height=1,
            fps=1,
            signaling_port=1024,  # Minimum non-privileged port
            name="streaming_client_min_values"
        )
        assert isinstance(op_min, BaseOperator), "Should handle minimum values"

        # Test maximum reasonable values
        op_max = StreamingClientOp(
            fragment=fragment,
            width=7680,  # 8K width
            height=4320,  # 8K height
            fps=120,     # High frame rate
            signaling_port=65535,  # Maximum port
            name="streaming_client_max_values"
        )
        assert isinstance(op_max, BaseOperator), "Should handle maximum values"

    def test_streaming_client_op_string_params(self, fragment):
        """Test StreamingClientOp with string parameter edge cases."""
        # Test empty server IP (should use default or handle gracefully)
        try:
            op = StreamingClientOp(
                fragment=fragment,
                server_ip="",  # Empty string
                name="streaming_client_empty_ip"
            )
            assert isinstance(op, BaseOperator), "Should handle empty IP string"
        except (ValueError, TypeError):
            # It's acceptable for empty IP to raise an error
            pass

        # Test very long server IP
        long_ip = "a" * 1000  # Unreasonably long string
        with pytest.raises((ValueError, TypeError)):
            StreamingClientOp(
                fragment=fragment,
                server_ip=long_ip,
                name="streaming_client_long_ip"
            )

    def test_streaming_client_op_initialization_lifecycle(self, fragment):
        """Test StreamingClientOp initialization and lifecycle methods."""
        op = StreamingClientOp(
            fragment=fragment,
            width=854,
            height=480,
            fps=30,
            name="streaming_client_lifecycle_test"
        )
        
        # Test initialize method doesn't crash
        # Note: Full initialization might require network setup
        try:
            op.initialize()
        except Exception as e:
            # It's acceptable for initialize to fail in test environment
            # without proper streaming infrastructure
            assert "StreamingClient" in str(e) or "connection" in str(e).lower(), \
                f"Expected streaming-related error, got: {e}"

    @pytest.mark.parametrize("name", [
        "test_op",
        "streaming_client_123", 
        "op_with_underscores",
        "op-with-dashes"
    ])
    def test_streaming_client_op_naming(self, fragment, name):
        """Test StreamingClientOp with different operator names."""
        op = StreamingClientOp(
            fragment=fragment,
            name=name
        )
        assert isinstance(op, BaseOperator), f"Should handle operator name: {name}"
        assert f"name: {name}" in repr(op), "Operator name should appear in repr()"

    def test_streaming_client_op_repr(self, fragment):
        """Test StreamingClientOp string representation."""
        op = StreamingClientOp(
            fragment=fragment,
            width=1920,
            height=1080,
            fps=60,
            name="test_streaming_client"
        )
        
        repr_str = repr(op)
        assert "StreamingClientOp" in repr_str, "Class name should be in repr"
        assert "test_streaming_client" in repr_str, "Operator name should be in repr"

    def test_streaming_client_op_multiple_instances(self, fragment):
        """Test creating multiple StreamingClientOp instances."""
        ops = []
        for i in range(3):
            op = StreamingClientOp(
                fragment=fragment,
                width=640 + i*100,
                height=480 + i*50, 
                fps=30 + i*10,
                name=f"streaming_client_{i}"
            )
            ops.append(op)
            assert isinstance(op, BaseOperator), f"Should create instance {i}"
        
        # Ensure all instances are unique
        assert len(set(id(op) for op in ops)) == 3, "All instances should be unique objects"


# Integration test with minimal application
class TestStreamingClientOpIntegration:
    """Integration tests for StreamingClientOp in application context."""

    def test_streaming_client_op_in_app(self, app):
        """Test StreamingClientOp in a minimal application context."""
        from holoscan.core import Fragment
        
        fragment = Fragment()
        op = StreamingClientOp(
            fragment=fragment,
            width=854,
            height=480,
            name="streaming_client_app_test"
        )
        
        assert isinstance(op, BaseOperator), "Should work in application context"
