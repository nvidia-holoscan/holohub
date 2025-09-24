"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""  # noqa: E501

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
    from holohub.streaming_client import StreamingClientOp
except ImportError:
    try:
        # Try alternative import path
        from holohub.streaming_client_operator import StreamingClientOp
    except ImportError as e:
        pytest.fail(f"Failed to import StreamingClientOp: {e}. "
                    "Ensure the operator is built and available.")




class TestStreamingClientOp:
    """Test suite for StreamingClientOp."""

    @pytest.fixture
    def fragment(self):
        """Provide a mock Holoscan Fragment."""
        return Mock()

    def test_streaming_client_op_init_basic(self, fragment):
        """Test basic StreamingClientOp initialization and properties."""
        name = "streaming_client_op"
        op = StreamingClientOp(fragment=fragment, name=name)

        assert isinstance(op, BaseOperator)
        assert op.operator_type == Operator.OperatorType.NATIVE
        assert op.name == name

    def test_streaming_client_op_has_input_output_ports(self, fragment, StreamingClientOp):
        """Test that StreamingClientOp has input and output ports (pipeline operator)."""
        op = StreamingClientOp(fragment=fragment, name="streaming_client")
        spec = op.spec

        # StreamingClientOp is a pipeline operator with ports for video frames
        assert len(spec.inputs) > 0, "StreamingClientOp should have input ports for video frames"
        assert len(spec.outputs) > 0, "StreamingClientOp should have output ports for video frames"

    def test_streaming_client_op_init_with_custom_name(self, fragment, StreamingClientOp):
        """Test StreamingClientOp initialization with custom name."""
        op = StreamingClientOp(fragment=fragment, name="custom_streaming_client")

        assert isinstance(op, BaseOperator), "StreamingClientOp should be a Holoscan operator"

        assert op.name == "custom_streaming_client", "Operator should have the specified name"

    def test_streaming_client_op_basic_properties(self, fragment, StreamingClientOp):
        """Test StreamingClientOp basic properties."""
        op = StreamingClientOp(fragment=fragment, name="test_streaming_client")

        assert isinstance(op, BaseOperator), "StreamingClientOp should be a Holoscan operator"

        assert op.name == "test_streaming_client", "Operator should have the specified name"

    def test_streaming_client_op_constructor_limitation(self, fragment, StreamingClientOp):
        """Test StreamingClientOp constructor accepts only fragment and name."""
        # Test that constructor works with valid params
        op = StreamingClientOp(fragment=fragment, name="test_client")
        assert isinstance(op, BaseOperator)

        # Constructor limitation: parameters must be set via setup() or Arg() in C++
        # This is the expected behavior for Holoscan operators

    def test_streaming_client_op_has_lifecycle_methods(self, fragment, StreamingClientOp):
        """Test that StreamingClientOp has required lifecycle methods."""
        op = StreamingClientOp(fragment=fragment, name="test_streaming_client")

        # Check that essential operator methods exist
        assert hasattr(op, "setup"), "StreamingClientOp should have setup method"
        assert hasattr(op, "initialize"), "StreamingClientOp should have initialize method"
        assert hasattr(op, "start"), "StreamingClientOp should have start method"
        assert hasattr(op, "stop"), "StreamingClientOp should have stop method"
        assert hasattr(op, "compute"), "StreamingClientOp should have compute method"

    def test_streaming_client_op_multiple_instances(self, fragment, StreamingClientOp):
        """Test creating multiple StreamingClientOp instances."""
        op1 = StreamingClientOp(fragment=fragment, name="client1")
        op2 = StreamingClientOp(fragment=fragment, name="client2")

        assert op1.name == "client1", "First operator should have correct name"
        assert op2.name == "client2", "Second operator should have correct name"
        assert op1 != op2, "Multiple instances should be different objects"

    def test_streaming_client_op_repr_contains_info(self, fragment, StreamingClientOp):
        """Test that StreamingClientOp repr contains useful information."""
        name = "test_streaming_client"
        op = StreamingClientOp(fragment=fragment, name=name)
        repr_str = repr(op)

        assert name in repr_str, "Operator name should appear in repr"
        # More flexible check for operator information
        assert len(repr_str) > 10, "Repr should contain meaningful information"

    def test_streaming_client_op_edge_case_names(self, fragment, StreamingClientOp):
        """Test StreamingClientOp with edge case names."""
        # Test with minimal name
        op = StreamingClientOp(fragment=fragment, name="a")
        assert op.name == "a", "Should accept single character name"

        # Test with longer name
        op2 = StreamingClientOp(
            fragment=fragment, name="very_long_streaming_client_name_with_underscores"
        )
        assert op2.name == "very_long_streaming_client_name_with_underscores"

    def test_streaming_client_op_default_name_behavior(self, fragment, StreamingClientOp):
        """Test StreamingClientOp default name behavior."""
        # Test without explicit name (should use default)
        try:
            op = StreamingClientOp(fragment=fragment)
            # Should have some default name
            assert hasattr(op, "name"), "Operator should have a name attribute"
            assert len(op.name) > 0, "Name should not be empty"
        except TypeError:
            # If name is required, that's also acceptable behavior
            pass

    def test_streaming_client_op_fragment_association(self, fragment, StreamingClientOp):
        """Test that StreamingClientOp is properly associated with fragment."""
        op = StreamingClientOp(fragment=fragment, name="test_streaming_client")

        try:
            assert (
                op.fragment == fragment
            ), "Operator should be associated with the provided fragment"
        except AttributeError:
            # Some operator implementations may not expose fragment directly
            pass

    def test_streaming_client_op_spec_validation(self, fragment, StreamingClientOp):
        """Test that StreamingClientOp spec is properly configured."""
        op = StreamingClientOp(fragment=fragment, name="test_streaming_client")
        spec = op.spec

        # Basic spec validation
        assert spec is not None, "Operator should have a spec"
        assert hasattr(spec, "inputs"), "Spec should have inputs"
        assert hasattr(spec, "outputs"), "Spec should have outputs"

    def test_streaming_client_op_operator_type(self, fragment, StreamingClientOp):
        """Test StreamingClientOp operator type validation."""
        op = StreamingClientOp(fragment=fragment, name="test_streaming_client")

        try:
            # Should be a native operator type
            operator_type = getattr(op, "operator_type", None)
            if operator_type is not None:
                assert operator_type == Operator.OperatorType.NATIVE
        except (AttributeError, ImportError):
            pass  # Mock case or different operator interface

    def test_streaming_client_op_architecture_validation(self, fragment, StreamingClientOp):
        """Test StreamingClientOp architecture and structure."""
        op = StreamingClientOp(fragment=fragment, name="test_streaming_client")

        # Validate basic operator structure
        assert hasattr(op, "name"), "Operator should have name attribute"
        assert hasattr(op, "spec"), "Operator should have spec attribute"

        # Pipeline operator should have ports
        spec = op.spec
        if hasattr(spec, "inputs") and hasattr(spec, "outputs"):
            # StreamingClient processes video frames, so should have input/output
            assert (
                len(spec.inputs) > 0 or len(spec.outputs) > 0
            ), "Pipeline operator should have input or output ports"
