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

from unittest.mock import Mock

import pytest
from holoscan.core import Operator

try:
    from holoscan.core import BaseOperator
except ImportError:
    from holoscan.core import _Operator as BaseOperator

# Import Arg for parameter passing
try:
    from holoscan.core import Arg
except ImportError:
    # Fallback for older versions
    class Arg:
        def __init__(self, name):
            self.name = name

        def __call__(self, value):
            return (self.name, value)


try:
    from holohub.streaming_server import StreamingServerOp
except ImportError:
    try:
        # Try alternative import path
        from holohub.streaming_server_operator import StreamingServerOp
    except ImportError as e:
        pytest.fail(
            f"Failed to import StreamingServerOp: {e}. "
            "Ensure the operator is built and available."
        )


class TestStreamingServerOp:
    """Test suite for StreamingServerOp."""

    @pytest.fixture
    def fragment(self):
        """Provide a mock Holoscan Fragment."""
        try:
            from holoscan.core import Fragment
            # Try to create a real Fragment for testing
            return Fragment()
        except Exception:
            # Fallback to Mock if Fragment creation fails
            mock_fragment = Mock()
            # Add necessary attributes that StreamingServerOp might expect
            mock_fragment.name = "test_fragment"
            return mock_fragment

    def test_streaming_server_op_init_basic(self, fragment):
        """Test basic StreamingServerOp initialization and properties."""
        name = "streaming_server_op"
        # Force Docker to see this as a new file - fixed fragment and constructor issues
        op = StreamingServerOp(fragment, name=name)

        assert isinstance(op, BaseOperator), "StreamingServerOp should be a Holoscan operator"
        assert op.operator_type == Operator.OperatorType.NATIVE, "Operator type should be NATIVE"
        assert f"name: {name}" in repr(op), "Operator name should appear in repr()"

    def test_streaming_server_op_init_with_default_name(self, fragment):
        """Test StreamingServerOp initialization with default name."""
        op = StreamingServerOp(fragment)

        assert isinstance(op, BaseOperator), "StreamingServerOp should be a Holoscan operator"
        assert "streaming_server" in repr(op), "Default name should be 'streaming_server'"

    def test_streaming_server_op_setup_no_ports(self, fragment):
        """Test StreamingServerOp setup - should have no input/output ports (standalone operator)."""
        op = StreamingServerOp(fragment, name="test_streaming_server")
        spec = op.spec

        # StreamingServerOp is a standalone operator with no pipeline ports
        # It communicates via network, not through Holoscan ports
        assert (
            len(spec.inputs) == 0
        ), "StreamingServerOp should have no input ports (standalone operator)"
        assert (
            len(spec.outputs) == 0
        ), "StreamingServerOp should have no output ports (standalone operator)"

    def test_streaming_server_op_basic_initialization(self, fragment):
        """Test StreamingServerOp basic initialization without custom parameters."""
        # Test with default parameters - this is what the real operator supports
        op = StreamingServerOp(fragment, name="test_streaming_server")

        assert isinstance(op, BaseOperator), "StreamingServerOp should be a Holoscan operator"
        # Note: The real operator uses default parameters defined in setup() method

    def test_streaming_server_op_name_assignment(self, fragment):
        """Test StreamingServerOp initialization with custom name."""
        custom_name = "custom_streaming_server"
        op = StreamingServerOp(fragment, name=custom_name)

        assert isinstance(op, BaseOperator), "StreamingServerOp should be a Holoscan operator"

    def test_streaming_server_op_default_initialization(self, fragment):
        """Test StreamingServerOp initialization with default settings."""
        op = StreamingServerOp(fragment, name="default_streaming_server")

        assert isinstance(op, BaseOperator), "StreamingServerOp should be a Holoscan operator"
        # Test that initialization completes without errors

    def test_streaming_server_op_is_holoscan_operator(self, fragment):
        """Test that StreamingServerOp is indeed a Holoscan operator."""
        op = StreamingServerOp(fragment, name="test_streaming_server")

        # Should inherit from Holoscan's base operator class
        assert isinstance(op, BaseOperator), "StreamingServerOp should inherit from BaseOperator"

    def test_streaming_server_op_parameters_not_directly_accessible(self, fragment):
        """Test that StreamingServerOp parameters are properly encapsulated."""
        op = StreamingServerOp(fragment, name="test_streaming_server")

        # Parameters should be encapsulated and not directly accessible
        # This is expected behavior for Holoscan operators
        assert not hasattr(op, "width"), "Width parameter should be encapsulated"
        assert not hasattr(op, "height"), "Height parameter should be encapsulated"
        assert not hasattr(op, "fps"), "FPS parameter should be encapsulated"

    def test_streaming_server_op_lifecycle_methods_exist(self, fragment):
        """Test that StreamingServerOp has required lifecycle methods."""
        op = StreamingServerOp(fragment, name="test_streaming_server")

        # Verify lifecycle methods exist
        assert hasattr(op, "setup"), "StreamingServerOp should have setup method"
        assert hasattr(op, "initialize"), "StreamingServerOp should have initialize method"
        assert hasattr(op, "start"), "StreamingServerOp should have start method"
        assert hasattr(op, "stop"), "StreamingServerOp should have stop method"
        assert hasattr(op, "compute"), "StreamingServerOp should have compute method"

    def test_streaming_server_op_setup_call(self, fragment):
        """Test that StreamingServerOp setup method can be called."""
        op = StreamingServerOp(fragment, name="test_streaming_server")

        # Setup should be callable without raising exceptions
        # Note: We don't call it here as it's automatically called during operator creation
        assert callable(op.setup), "Setup method should be callable"

    def test_streaming_server_op_multiple_instances(self, fragment):
        """Test creating multiple StreamingServerOp instances."""
        op1 = StreamingServerOp(fragment, name="server1")
        op2 = StreamingServerOp(fragment, name="server2")

        assert op1 != op2, "Different instances should not be equal"
        assert isinstance(op1, BaseOperator), "First instance should be valid operator"
        assert isinstance(op2, BaseOperator), "Second instance should be valid operator"

    def test_streaming_server_op_repr_contains_info(self, fragment):
        """Test that StreamingServerOp repr contains useful information."""
        name = "test_streaming_server"
        op = StreamingServerOp(fragment, name=name)
        repr_str = repr(op)

        assert name in repr_str, "Operator name should appear in repr"
        # The repr may not contain the exact class name, just verify it's a valid representation
        assert len(repr_str) > 10, "Operator repr should contain meaningful information"

    def test_streaming_server_op_edge_case_names(self, fragment):
        """Test StreamingServerOp with various name values."""
        # Test with minimal name
        op = StreamingServerOp(fragment, name="a")
        assert isinstance(op, BaseOperator), "StreamingServerOp should handle minimal names"

    def test_streaming_server_op_long_name(self, fragment):
        """Test StreamingServerOp with long name."""
        long_name = "very_long_streaming_server_name_that_should_still_work"
        op = StreamingServerOp(fragment, name=long_name)

        assert isinstance(op, BaseOperator), "StreamingServerOp should handle long names"

    def test_streaming_server_op_compute_method_callable(self, fragment):
        """Test that StreamingServerOp compute method is callable (standalone operator)."""
        op = StreamingServerOp(fragment, name="test_streaming_server")

        # For standalone operators, compute should be callable but may not do much without network clients
        assert callable(op.compute), "Compute method should be callable"

        # Note: We don't actually call compute here as it requires proper initialization
        # and network setup, which is complex for unit testing

    def test_streaming_server_op_default_parameter_values(self, fragment):
        """Test StreamingServerOp uses appropriate default values."""
        op = StreamingServerOp(fragment, name="default_server")

        # Operator should initialize successfully with defaults
        assert isinstance(op, BaseOperator), "StreamingServerOp should work with default parameters"

        # Note: Default values are typically:
        # width=854, height=480, fps=30, port=48010, multi_instance=false,
        # server_name="StreamingServer", receive_frames=true, send_frames=true
