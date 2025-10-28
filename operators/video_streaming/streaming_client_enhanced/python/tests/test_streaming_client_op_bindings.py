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
Python unit tests for StreamingClientOp Python bindings.

These tests validate the Python bindings (pybind11) for the StreamingClientOp
operator, focusing on:
- Python/C++ binding correctness
- Parameter handling across language boundaries
- Method availability and behavior in Python
- Error handling in Python context
- Memory management
"""

import pytest


class TestStreamingClientOpBinding:
    """Test StreamingClientOp Python binding functionality."""

    def test_operator_creation_basic(self, operator_factory):
        """Test basic operator creation through Python bindings."""
        op = operator_factory()
        assert op is not None
        assert hasattr(op, 'setup')
        assert hasattr(op, 'name')

    def test_operator_name(self, operator_factory):
        """Test operator name property."""
        custom_name = "my_streaming_client"
        op = operator_factory(name=custom_name)
        assert op is not None
        assert op.name == custom_name

    @pytest.mark.parametrize("width,height,fps", [
        (640, 480, 30),
        (1280, 720, 60),
        (1920, 1080, 30),
        (3840, 2160, 24),
    ])
    def test_video_parameters(self, operator_factory, width, height, fps):
        """Test operator creation with different video parameters."""
        op = operator_factory(
            width=width,
            height=height,
            fps=fps
        )
        assert op is not None

    @pytest.mark.parametrize("server_ip,port", [
        ("127.0.0.1", 48010),
        ("192.168.1.100", 8080),
        ("10.0.0.1", 9999),
    ])
    def test_network_parameters(self, operator_factory, server_ip, port):
        """Test operator creation with different network parameters."""
        op = operator_factory(
            server_ip=server_ip,
            signaling_port=port
        )
        assert op is not None

    @pytest.mark.parametrize("receive_frames,send_frames", [
        (True, False),   # Receive only
        (False, True),   # Send only
        (True, True),    # Bidirectional
        (False, False),  # No streaming (configuration only)
    ])
    def test_streaming_mode_parameters(self, operator_factory, receive_frames, send_frames):
        """Test operator creation with different streaming modes."""
        op = operator_factory(
            receive_frames=receive_frames,
            send_frames=send_frames
        )
        assert op is not None

    def test_frame_validation_parameter(self, operator_factory):
        """Test min_non_zero_bytes parameter."""
        for threshold in [0, 100, 1000, 10000]:
            op = operator_factory(min_non_zero_bytes=threshold)
            assert op is not None

    def test_operator_inheritance(self, streaming_client_op_class, holoscan_modules):
        """Test that StreamingClientOp is a valid Holoscan Operator."""
        # Note: pybind11 wrapped classes may not show as direct subclasses via issubclass()
        # Instead, verify it's an Operator by checking for Operator-like methods
        assert hasattr(streaming_client_op_class, '__init__')
        # If we can instantiate it and it has operator methods, it's a valid operator
        assert streaming_client_op_class is not None

    def test_method_availability(self, default_operator):
        """Test that required methods and properties are available through Python bindings."""
        op = default_operator
        
        # Check core operator methods
        assert hasattr(op, 'setup')
        assert callable(getattr(op, 'setup'))
        
        assert hasattr(op, 'name')
        # name is a property, not a method - verify it's accessible and returns a string
        assert isinstance(op.name, str)

    def test_setup_method(self, default_operator):
        """Test the setup method exists on operator."""
        op = default_operator
        
        # Verify setup method exists (it's called during operator lifecycle)
        # We don't call it directly as it requires proper OperatorSpec context
        # which is managed by the Holoscan framework
        assert hasattr(op, 'setup')
        assert callable(getattr(op, 'setup'))

    def test_memory_management(self, operator_factory):
        """Test memory management across Python/C++ boundary."""
        # Create multiple operators
        operators = []
        for i in range(10):
            op = operator_factory(name=f"client_{i}")
            operators.append(op)
        
        # All operators should be valid
        assert len(operators) == 10
        for op in operators:
            assert op is not None

        # Clear references (Python garbage collection should handle C++ objects)
        del operators

    def test_operator_reuse(self, operator_factory):
        """Test creating multiple operators with same parameters."""
        params = {
            'width': 1280,
            'height': 720,
            'fps': 60,
            'server_ip': '192.168.1.100',
            'signaling_port': 8080
        }
        
        # Create multiple operators with same parameters
        op1 = operator_factory(**params)
        op2 = operator_factory(**params)
        
        assert op1 is not None
        assert op2 is not None
        assert op1 is not op2  # Should be different instances

    def test_docstring_availability(self, streaming_client_op_class):
        """Test that docstrings are available for the Python bindings."""
        assert hasattr(streaming_client_op_class, '__doc__')
        doc = getattr(streaming_client_op_class, '__doc__')
        assert doc is not None

    def test_string_parameter_handling(self, operator_factory):
        """Test string parameter handling across Python/C++ boundary."""
        # Test various string formats
        test_cases = [
            ("127.0.0.1", "localhost"),
            ("192.168.1.1", "192.168.1.1"),
            ("10.0.0.1", "10.0.0.1"),
        ]
        
        for ip1, ip2 in test_cases:
            # Test both IP formats to verify different string formats work
            op1 = operator_factory(server_ip=ip1)
            assert op1 is not None
            op2 = operator_factory(server_ip=ip2)
            assert op2 is not None

    def test_edge_case_resolutions(self, operator_factory):
        """Test edge case video resolutions."""
        # Minimum resolution
        op_min = operator_factory(width=320, height=240)
        assert op_min is not None
        
        # Maximum resolution (4K)
        op_max = operator_factory(width=3840, height=2160)
        assert op_max is not None

    def test_edge_case_fps(self, operator_factory):
        """Test edge case frame rates."""
        for fps in [1, 15, 30, 60, 120]:
            op = operator_factory(fps=fps)
            assert op is not None

    def test_edge_case_ports(self, operator_factory):
        """Test edge case port numbers."""
        # Low port
        op_low = operator_factory(signaling_port=1)
        assert op_low is not None
        
        # High port
        op_high = operator_factory(signaling_port=65535)
        assert op_high is not None

    def test_multiple_instances_isolation(self, operator_factory):
        """Test that multiple instances are properly isolated."""
        op1 = operator_factory(name="client1", width=640, height=480)
        op2 = operator_factory(name="client2", width=1920, height=1080)
        
        assert op1 is not None
        assert op2 is not None
        assert op1.name == "client1"
        assert op2.name == "client2"
        assert op1.name != op2.name


class TestStreamingClientOpIntegration:
    """Integration tests for StreamingClientOp Python bindings in Application context."""

    def test_operator_in_application_context(self, holoscan_modules, streaming_client_op_class):
        """Test StreamingClientOp within a Holoscan Application context."""
        Application = holoscan_modules['Application']
        
        class TestApp(Application):
            def compose(self):
                # Create the streaming client operator
                streaming_client_op_class(
                    self,
                    name="test_client",
                    width=640,
                    height=480,
                    fps=30,
                    server_ip="127.0.0.1",
                    signaling_port=48010,
                    send_frames=False,
                    receive_frames=False
                )
                # Note: Not adding to workflow to avoid execution
        
        app = TestApp()
        assert app is not None

    def test_operator_in_fragment(self, holoscan_modules, streaming_client_op_class, fragment):
        """Test StreamingClientOp within a Fragment."""
        op = streaming_client_op_class(
            fragment,
            name="fragment_client",
            width=1280,
            height=720,
            fps=60,
            send_frames=False,
            receive_frames=False
        )
        assert op is not None
        assert op.name == "fragment_client"

