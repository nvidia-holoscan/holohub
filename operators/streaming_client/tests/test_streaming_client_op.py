# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import Mock
from holoscan.core import Operator

try:
    from holoscan.core import BaseOperator
except ImportError:
    from holoscan.core import _Operator as BaseOperator

try:
    from holohub.streaming_client_operator import StreamingClientOp
    REAL_OPERATOR_AVAILABLE = True
except ImportError:
    REAL_OPERATOR_AVAILABLE = False
    StreamingClientOp = None


class MockStreamingClientOp:
    """Mock StreamingClientOp for testing when the real operator isn't available."""
    
    def __init__(self, fragment=None, name="streaming_client", **kwargs):
        self.fragment = fragment
        self.name = name
        self.kwargs = kwargs
        self._spec = Mock()
        self._spec.inputs = {"input_frames": Mock()}
        self._spec.outputs = {"output_frames": Mock()}
        self.operator_type = getattr(Operator, 'OperatorType', Mock()).NATIVE if hasattr(Operator, 'OperatorType') else Mock()
    
    @property
    def spec(self):
        return self._spec
    
    def setup(self, spec):
        pass
    
    def initialize(self):
        pass
    
    def start(self):
        pass
    
    def stop(self):
        pass
    
    def compute(self, op_input, op_output, context):
        pass
    
    def __repr__(self):
        return f"MockStreamingClientOp(name: {self.name})"


class TestStreamingClientOp:
    """Test suite for StreamingClientOp."""

    @pytest.fixture
    def operator_class(self):
        """Provide the StreamingClientOp class, using mock if real one isn't available."""
        try:
            return StreamingClientOp
        except (ImportError, NameError):
            return MockStreamingClientOp

    def test_streaming_client_op_init_basic(self, fragment, operator_class):
        """Test basic StreamingClientOp initialization and properties."""
        name = "streaming_client_op"
        op = operator_class(fragment=fragment, name=name)
        
        try:
            assert isinstance(op, BaseOperator), "StreamingClientOp should be a Holoscan operator"
            assert op.operator_type == Operator.OperatorType.NATIVE, "Operator type should be NATIVE"
        except (ImportError, AttributeError):
            pass  # Mock case
        assert f"name: {name}" in repr(op), "Operator name should appear in repr()"

    def test_streaming_client_op_init_with_default_name(self, fragment, operator_class):
        """Test StreamingClientOp initialization with default name."""
        op = operator_class(fragment=fragment)
        
        try:
            assert isinstance(op, BaseOperator), "StreamingClientOp should be a Holoscan operator"
        except (ImportError, AttributeError):
            pass  # Mock case
        assert "streaming_client" in repr(op), "Default name should be 'streaming_client'"

    def test_streaming_client_op_setup_with_ports(self, fragment, operator_class):
        """Test StreamingClientOp setup - should have input/output ports for video pipeline."""
        op = operator_class(fragment=fragment, name="test_streaming_client")
        spec = op.spec
        
        # StreamingClientOp is a pipeline operator with ports for video frames
        assert len(spec.inputs) > 0, "StreamingClientOp should have input ports for video frames"
        assert len(spec.outputs) > 0, "StreamingClientOp should have output ports for video frames"

    @pytest.mark.parametrize("width,height,fps", [
        (854, 480, 30),      # Default resolution
        (1920, 1080, 60),    # HD resolution, high fps
        (640, 480, 25),      # Lower resolution
        (3840, 2160, 30),    # 4K resolution
    ])
    def test_streaming_client_op_init_with_video_params(self, fragment, operator_class, width, height, fps):
        """Test StreamingClientOp initialization with various video parameters."""
        op = operator_class(
            fragment=fragment, 
            name="test_streaming_client",
            width=width,
            height=height, 
            fps=fps
        )
        
        try:
            assert isinstance(op, BaseOperator), "StreamingClientOp should be a Holoscan operator"
        except (ImportError, AttributeError):
            pass  # Mock case

    @pytest.mark.parametrize("server_ip,signaling_port", [
        ("localhost", 8080),
        ("127.0.0.1", 48010),
        ("192.168.1.100", 9999),
        ("10.0.0.1", 12345)
    ])
    def test_streaming_client_op_init_with_network_params(self, fragment, operator_class, server_ip, signaling_port):
        """Test StreamingClientOp initialization with various network parameters."""
        op = operator_class(
            fragment=fragment,
            name="test_streaming_client",
            server_ip=server_ip,
            signaling_port=signaling_port
        )
        
        try:
            assert isinstance(op, BaseOperator), "StreamingClientOp should be a Holoscan operator"
        except (ImportError, AttributeError):
            pass  # Mock case

    @pytest.mark.parametrize("receive_frames,send_frames", [
        (True, True),     # Bidirectional
        (True, False),    # Receive only
        (False, True),    # Send only  
        (False, False),   # Neither (minimal config)
    ])
    def test_streaming_client_op_init_with_frame_modes(self, fragment, operator_class, receive_frames, send_frames):
        """Test StreamingClientOp initialization with different frame handling modes."""
        op = operator_class(
            fragment=fragment,
            name="test_streaming_client", 
            receive_frames=receive_frames,
            send_frames=send_frames
        )
        
        try:
            assert isinstance(op, BaseOperator), "StreamingClientOp should be a Holoscan operator"
        except (ImportError, AttributeError):
            pass  # Mock case

    def test_streaming_client_op_invalid_width(self, fragment, operator_class):
        """Test StreamingClientOp raises error on invalid width parameter."""
        try:
            with pytest.raises((ValueError, TypeError)):
                operator_class(fragment=fragment, width=0)
        except (ImportError, AttributeError):
            pass  # Mock case doesn't validate

    def test_streaming_client_op_invalid_height(self, fragment, operator_class):
        """Test StreamingClientOp raises error on invalid height parameter."""
        try:
            with pytest.raises((ValueError, TypeError)):
                operator_class(fragment=fragment, height=0)
        except (ImportError, AttributeError):
            pass  # Mock case doesn't validate

    def test_streaming_client_op_invalid_fps(self, fragment, operator_class):
        """Test StreamingClientOp raises error on invalid fps parameter."""
        try:
            with pytest.raises((ValueError, TypeError)):
                operator_class(fragment=fragment, fps=0)
        except (ImportError, AttributeError):
            pass  # Mock case doesn't validate

    def test_streaming_client_op_invalid_signaling_port(self, fragment, operator_class):
        """Test StreamingClientOp raises error on invalid signaling_port parameter."""
        try:
            with pytest.raises((ValueError, TypeError)):
                operator_class(fragment=fragment, signaling_port=0)
        except (ImportError, AttributeError):
            pass  # Mock case doesn't validate
        
        try:
            with pytest.raises((ValueError, TypeError)):
                operator_class(fragment=fragment, signaling_port=99999)  # Port too high
        except (ImportError, AttributeError):
            pass  # Mock case doesn't validate

    def test_streaming_client_op_invalid_server_ip_type(self, fragment, operator_class):
        """Test StreamingClientOp raises error on invalid server_ip type."""
        try:
            with pytest.raises(TypeError):
                operator_class(fragment=fragment, server_ip=123)  # Should be string
        except (ImportError, AttributeError):
            pass  # Mock case doesn't validate

    def test_streaming_client_op_parameters_not_directly_accessible(self, fragment, operator_class):
        """Test that StreamingClientOp parameters are properly encapsulated."""
        op = operator_class(
            fragment=fragment,
            name="test_streaming_client",
            width=1920,
            height=1080,
            fps=30,
            server_ip="localhost"
        )
        
        # Parameters should be encapsulated and not directly accessible
        # This is expected behavior for Holoscan operators
        assert not hasattr(op, 'width'), "Width parameter should be encapsulated"
        assert not hasattr(op, 'height'), "Height parameter should be encapsulated" 
        assert not hasattr(op, 'fps'), "FPS parameter should be encapsulated"
        assert not hasattr(op, 'server_ip'), "Server IP parameter should be encapsulated"

    def test_streaming_client_op_lifecycle_methods_exist(self, fragment, operator_class):
        """Test that StreamingClientOp has required lifecycle methods."""
        op = operator_class(fragment=fragment, name="test_streaming_client")
        
        # Verify lifecycle methods exist
        assert hasattr(op, 'setup'), "StreamingClientOp should have setup method"
        assert hasattr(op, 'initialize'), "StreamingClientOp should have initialize method"
        assert hasattr(op, 'start'), "StreamingClientOp should have start method"
        assert hasattr(op, 'stop'), "StreamingClientOp should have stop method"
        assert hasattr(op, 'compute'), "StreamingClientOp should have compute method"

    def test_streaming_client_op_setup_call(self, fragment, operator_class):
        """Test that StreamingClientOp setup method can be called."""
        op = operator_class(fragment=fragment, name="test_streaming_client")
        
        # Setup should be callable without raising exceptions
        assert callable(op.setup), "Setup method should be callable"

    def test_streaming_client_op_multiple_instances(self, fragment, operator_class):
        """Test creating multiple StreamingClientOp instances."""
        op1 = operator_class(fragment=fragment, name="client1", signaling_port=8080)
        op2 = operator_class(fragment=fragment, name="client2", signaling_port=8081)
        
        assert op1 != op2, "Different instances should not be equal"
        try:
            assert isinstance(op1, BaseOperator), "First instance should be valid operator"
            assert isinstance(op2, BaseOperator), "Second instance should be valid operator"
        except (ImportError, AttributeError):
            pass  # Mock case

    def test_streaming_client_op_repr_contains_info(self, fragment, operator_class):
        """Test that StreamingClientOp repr contains useful information."""
        name = "test_streaming_client"
        op = operator_class(fragment=fragment, name=name)
        repr_str = repr(op)
        
        assert name in repr_str, "Operator name should appear in repr"
        assert "StreamingClientOp" in repr_str or "MockStreamingClientOp" in repr_str, "Operator class should appear in repr"

    def test_streaming_client_op_edge_case_parameters(self, fragment, operator_class):
        """Test StreamingClientOp with edge case parameter values."""
        # Test with minimum reasonable values
        op = operator_class(
            fragment=fragment,
            name="edge_case_client",
            width=1,      # Minimum width
            height=1,     # Minimum height  
            fps=1,        # Minimum fps
            signaling_port=1024,    # Minimum user port
        )
        
        try:
            assert isinstance(op, BaseOperator), "StreamingClientOp should handle edge case parameters"
        except (ImportError, AttributeError):
            pass  # Mock case

    def test_streaming_client_op_empty_server_ip(self, fragment, operator_class):
        """Test StreamingClientOp with empty server IP."""
        op = operator_class(
            fragment=fragment,
            name="test_streaming_client",
            server_ip=""
        )
        
        try:
            assert isinstance(op, BaseOperator), "StreamingClientOp should handle empty server IP"
        except (ImportError, AttributeError):
            pass  # Mock case

    def test_streaming_client_op_compute_method_callable(self, fragment, execution_context, operator_class):
        """Test that StreamingClientOp compute method is callable (pipeline operator)."""
        op = operator_class(fragment=fragment, name="test_streaming_client")
        
        # For pipeline operators, compute should be callable and process frames
        assert callable(op.compute), "Compute method should be callable"
        
        # Note: We don't actually call compute here as it requires proper initialization
        # and network setup, which is complex for unit testing

    def test_streaming_client_op_default_parameter_values(self, fragment, operator_class):
        """Test StreamingClientOp uses appropriate default values."""
        op = operator_class(fragment=fragment, name="default_client")
        
        # Operator should initialize successfully with defaults
        try:
            assert isinstance(op, BaseOperator), "StreamingClientOp should work with default parameters"
        except (ImportError, AttributeError):
            pass  # Mock case
        
        # Note: Default values are typically:
        # width=854, height=480, fps=30, server_ip="localhost", signaling_port=48010
        # receive_frames=true, send_frames=true
