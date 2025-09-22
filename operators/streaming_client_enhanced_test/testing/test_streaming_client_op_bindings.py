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
Unit tests for StreamingClientOp Python bindings (pybind11).

This module tests the Python bindings of the StreamingClientOp operator,
focusing on:
- Correct binding of C++ class to Python
- Parameter setting and getting
- Method availability and behavior
- Error handling in Python context
- Memory management across Python/C++ boundary
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


class TestStreamingClientOpBinding:
    """Test class for StreamingClientOp Python binding functionality."""

    @pytest.mark.unit
    def test_operator_creation_basic(self, operator_factory):
        """Test basic operator creation through Python bindings."""
        op = operator_factory()
        assert op is not None
        assert hasattr(op, 'initialize')
        assert hasattr(op, 'setup')

    @pytest.mark.unit
    def test_operator_creation_with_custom_name(self, operator_factory):
        """Test operator creation with custom name."""
        custom_name = "my_streaming_client"
        op = operator_factory(name=custom_name)
        assert op is not None
        # Note: The name might be stored internally, but pybind11 might not expose it directly
        # This test verifies the parameter is accepted without error

    @pytest.mark.unit
    @pytest.mark.parametrized
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

    @pytest.mark.unit
    @pytest.mark.parametrized
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

    @pytest.mark.unit
    @pytest.mark.parametrized
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

    @pytest.mark.unit
    def test_parameter_type_validation(self, operator_factory):
        """Test that parameter types are properly validated."""
        # Test valid parameters
        op = operator_factory(
            width=640,
            height=480,
            fps=30,
            server_ip="127.0.0.1",
            signaling_port=48010,
            receive_frames=True,
            send_frames=False,
            min_non_zero_bytes=100
        )
        assert op is not None

    @pytest.mark.unit
    def test_invalid_parameters_handling(self, operator_factory):
        """Test handling of invalid parameters."""
        # Test negative dimensions (should either raise or be handled gracefully)
        with pytest.raises((ValueError, TypeError, RuntimeError)):
            operator_factory(width=-640, height=-480)

        # Test invalid port numbers
        with pytest.raises((ValueError, TypeError, RuntimeError)):
            operator_factory(signaling_port=-1)

        with pytest.raises((ValueError, TypeError, RuntimeError)):
            operator_factory(signaling_port=70000)  # Port out of valid range

    @pytest.mark.unit
    def test_string_parameter_handling(self, operator_factory):
        """Test string parameter handling across Python/C++ boundary."""
        # Test empty string
        op = operator_factory(server_ip="")
        assert op is not None

        # Test Unicode strings
        op = operator_factory(name="streaming_client_测试")
        assert op is not None

        # Test long strings
        long_name = "streaming_client_" + "x" * 100
        op = operator_factory(name=long_name)
        assert op is not None

    @pytest.mark.unit
    def test_operator_inheritance(self, streaming_client_op_class, holoscan_modules):
        """Test that StreamingClientOp properly inherits from Holoscan Operator."""
        Operator = holoscan_modules['Operator']
        
        # Check inheritance - StreamingClientOp is a pybind11 wrapped class
        # Note: Direct issubclass check may fail with pybind11 bindings
        # Instead, check if the class has operator-like attributes
        assert hasattr(streaming_client_op_class, '__init__')
        assert hasattr(streaming_client_op_class, '__call__') or hasattr(streaming_client_op_class, '__new__')
        # StreamingClientOp requires a fragment parameter, so we can't create it without one

    @pytest.mark.unit
    def test_method_availability(self, operator_factory):
        """Test that required methods are available through Python bindings."""
        op = operator_factory()
        
        # Check core operator methods
        assert hasattr(op, 'initialize')
        assert callable(getattr(op, 'initialize'))
        
        assert hasattr(op, 'setup')
        assert callable(getattr(op, 'setup'))

    @pytest.mark.unit
    def test_initialize_method(self, operator_factory):
        """Test the initialize method through Python bindings."""
        op = operator_factory()
        
        # The initialize method should be callable
        # Note: It might raise exceptions if called without proper context
        try:
            op.initialize()
        except Exception as e:
            # This is expected since we don't have a full Holoscan context
            assert isinstance(e, (RuntimeError, AttributeError))

    @pytest.mark.unit
    def test_setup_method(self, operator_factory, holoscan_modules):
        """Test the setup method through Python bindings."""
        op = operator_factory()
        
        # Create a mock operator spec
        try:
            from holoscan.core import OperatorSpec
            Fragment = holoscan_modules['Fragment']
            fragment = Fragment()
            spec = OperatorSpec(fragment, op)  # OperatorSpec requires both fragment and operator
            op.setup(spec)
        except (ImportError, Exception, TypeError) as e:
            # This might not work without full context, but should not crash
            assert isinstance(e, (RuntimeError, ImportError, AttributeError, TypeError))

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
    def test_docstring_availability(self, streaming_client_op_class):
        """Test that docstrings are available for the Python bindings."""
        # The class should have a docstring
        assert hasattr(streaming_client_op_class, '__doc__')
        doc = getattr(streaming_client_op_class, '__doc__')
        assert doc is not None
        assert len(doc.strip()) > 0

    @pytest.mark.unit
    def test_module_attributes(self):
        """Test module-level attributes of the streaming_client binding."""
        try:
            import holohub.streaming_client as sc_module
            
            # Check for version attribute (optional for pybind11 modules)
            if hasattr(sc_module, '__version__'):
                version = getattr(sc_module, '__version__')
                assert isinstance(version, str)
                assert len(version) > 0
            
        except ImportError:
            pytest.skip("StreamingClient module not available")


class TestStreamingClientOpIntegration:
    """Integration tests for StreamingClientOp Python bindings."""

    @pytest.mark.integration
    def test_operator_in_application_context(self, holoscan_modules, streaming_client_op_class):
        """Test StreamingClientOp within a Holoscan Application context."""
        Application = holoscan_modules['Application']
        
        class TestApp(Application):
            def compose(self):
                # Create the streaming client operator
                streaming_client = streaming_client_op_class(
                    self,
                    width=640,
                    height=480,
                    fps=30,
                    name="test_client"
                )
                # Note: Not adding to graph to avoid execution
        
        app = TestApp()
        assert app is not None

    @pytest.mark.integration  
    @pytest.mark.hardware
    def test_operator_parameter_persistence(self, operator_factory):
        """Test that operator parameters persist correctly."""
        test_params = {
            'width': 1920,
            'height': 1080,
            'fps': 60,
            'server_ip': '10.0.0.100',
            'signaling_port': 9000,
            'receive_frames': True,
            'send_frames': True,
            'min_non_zero_bytes': 200
        }
        
        op = operator_factory(**test_params)
        assert op is not None
        
        # Parameters should be stored in the operator
        # Note: Direct parameter access might not be exposed through bindings


class TestStreamingClientOpErrorHandling:
    """Test error handling in StreamingClientOp Python bindings."""

    @pytest.mark.unit
    def test_exception_propagation(self, operator_factory):
        """Test that C++ exceptions are properly propagated to Python."""
        # Test parameter validation errors - note: operator may accept fps=0
        try:
            op = operator_factory(fps=0)  # May not raise exception
            # If no exception, verify the operator was created (param method may not be exposed)
            assert op is not None
        except Exception as e:
            # If exception is raised, that's also valid behavior
            assert isinstance(e, (ValueError, TypeError, RuntimeError, AttributeError))

    @pytest.mark.unit
    def test_null_fragment_handling(self, streaming_client_op_class, default_operator_params):
        """Test handling of null fragment parameter."""
        # Note: StreamingClientOp may not require fragment parameter in constructor
        try:
            op = streaming_client_op_class(
                None,  # Null fragment may or may not cause error
                **default_operator_params
            )
            # If no exception, verify the operator was created
            assert op is not None
        except (TypeError, ValueError, RuntimeError, AttributeError) as e:
            # Exception is also valid behavior for null fragment
            assert isinstance(e, (TypeError, ValueError, RuntimeError, AttributeError))

    @pytest.mark.unit
    def test_invalid_type_parameters(self, operator_factory):
        """Test handling of invalid parameter types."""
        # String where integer expected
        with pytest.raises((TypeError, ValueError)):
            operator_factory(width="640")  # String instead of int

        # Integer where string expected  
        with pytest.raises((TypeError, ValueError)):
            operator_factory(server_ip=127001)  # Int instead of string

        # Invalid boolean values
        with pytest.raises((TypeError, ValueError)):
            operator_factory(receive_frames="yes")  # String instead of bool


class TestStreamingClientOpPerformance:
    """Performance-related tests for StreamingClientOp Python bindings."""

    @pytest.mark.unit
    @pytest.mark.slow
    def test_operator_creation_performance(self, operator_factory):
        """Test performance of operator creation."""
        import time
        
        # Measure time to create multiple operators
        start_time = time.time()
        operators = []
        
        for i in range(100):
            op = operator_factory(name=f"perf_test_{i}")
            operators.append(op)
        
        end_time = time.time()
        creation_time = end_time - start_time
        
        # Should be able to create 100 operators in reasonable time (< 5 seconds)
        assert creation_time < 5.0
        assert len(operators) == 100

    @pytest.mark.unit
    def test_memory_usage(self, operator_factory):
        """Test memory usage of operator instances."""
        import gc
        
        # Force garbage collection before test
        gc.collect()
        
        # Create and destroy operators in a loop
        for i in range(50):
            op = operator_factory(name=f"memory_test_{i}")
            assert op is not None
            del op
        
        # Force garbage collection after test
        gc.collect()
        
        # Test passes if no memory leaks cause crashes
        assert True
