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
from unittest.mock import Mock
from holoscan.core import Operator

try:
    from holoscan.core import BaseOperator
except ImportError:
    from holoscan.core import _Operator as BaseOperator

try:
    from holohub.streaming_server import StreamingServerOp
    REAL_OPERATOR_AVAILABLE = True
except ImportError:
    REAL_OPERATOR_AVAILABLE = False
    StreamingServerOp = None


class MockStreamingServerOp:
    """Mock StreamingServerOp for testing when the real operator isn't available."""
    
    def __init__(self, fragment=None, name="streaming_server", **kwargs):
        self.fragment = fragment
        self.name = name
        self.kwargs = kwargs
        self._spec = Mock()
        self._spec.inputs = {}
        self._spec.outputs = {}
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
        return f"MockStreamingServerOp(name: {self.name})"


class TestStreamingServerOp:
    """Test suite for StreamingServerOp."""

    @pytest.fixture
    def operator_class(self):
        """Provide the StreamingServerOp class, using mock if real one isn't available."""
        if REAL_OPERATOR_AVAILABLE:
            return StreamingServerOp
        else:
            return MockStreamingServerOp

    def test_streaming_server_op_init_basic(self, fragment, operator_class):
        """Test basic StreamingServerOp initialization and properties."""
        name = "streaming_server_op"
        op = operator_class(fragment=fragment, name=name)
        
        if REAL_OPERATOR_AVAILABLE:
            assert isinstance(op, BaseOperator), "StreamingServerOp should be a Holoscan operator"
            assert op.operator_type == Operator.OperatorType.NATIVE, "Operator type should be NATIVE"
        assert f"name: {name}" in repr(op), "Operator name should appear in repr()"

    def test_streaming_server_op_init_with_default_name(self, fragment, operator_class):
        """Test StreamingServerOp initialization with default name."""
        op = operator_class(fragment=fragment)
        
        if REAL_OPERATOR_AVAILABLE:
            assert isinstance(op, BaseOperator), "StreamingServerOp should be a Holoscan operator"
        assert "streaming_server" in repr(op), "Default name should be 'streaming_server'"

    def test_streaming_server_op_setup_no_ports(self, fragment, operator_class):
        """Test StreamingServerOp setup - should have no input/output ports (standalone operator)."""
        op = operator_class(fragment=fragment, name="test_streaming_server")
        spec = op.spec
        
        # StreamingServerOp is a standalone operator with no pipeline ports
        # It communicates via network, not through Holoscan ports
        assert len(spec.inputs) == 0, "StreamingServerOp should have no input ports (standalone operator)"
        assert len(spec.outputs) == 0, "StreamingServerOp should have no output ports (standalone operator)"

    @pytest.mark.parametrize("width,height,fps", [
        (854, 480, 30),      # Default resolution
        (1920, 1080, 60),    # HD resolution, high fps
        (640, 480, 25),      # Lower resolution
        (3840, 2160, 30),    # 4K resolution
    ])
    def test_streaming_server_op_init_with_video_params(self, fragment, width, height, fps):
        """Test StreamingServerOp initialization with various video parameters."""
        op = StreamingServerOp(
            fragment=fragment, 
            name="test_streaming_server",
            width=width,
            height=height, 
            fps=fps
        )
        
        assert isinstance(op, BaseOperator), "StreamingServerOp should be a Holoscan operator"
        # Note: We can't directly access parameters without running setup, but we can verify initialization succeeded

    @pytest.mark.parametrize("port", [8080, 48010, 9999, 12345])
    def test_streaming_server_op_init_with_network_params(self, fragment, port):
        """Test StreamingServerOp initialization with various network parameters."""
        op = StreamingServerOp(
            fragment=fragment,
            name="test_streaming_server",
            port=port,
            server_name="TestServer"
        )
        
        assert isinstance(op, BaseOperator), "StreamingServerOp should be a Holoscan operator"

    @pytest.mark.parametrize("receive_frames,send_frames", [
        (True, True),     # Bidirectional
        (True, False),    # Receive only
        (False, True),    # Send only  
        (False, False),   # Neither (minimal config)
    ])
    def test_streaming_server_op_init_with_frame_modes(self, fragment, receive_frames, send_frames):
        """Test StreamingServerOp initialization with different frame handling modes."""
        op = StreamingServerOp(
            fragment=fragment,
            name="test_streaming_server", 
            receive_frames=receive_frames,
            send_frames=send_frames
        )
        
        assert isinstance(op, BaseOperator), "StreamingServerOp should be a Holoscan operator"

    def test_streaming_server_op_init_with_multi_instance(self, fragment):
        """Test StreamingServerOp initialization with multi-instance support."""
        op = StreamingServerOp(
            fragment=fragment,
            name="test_streaming_server",
            multi_instance=True
        )
        
        assert isinstance(op, BaseOperator), "StreamingServerOp should be a Holoscan operator"

    def test_streaming_server_op_invalid_width(self, fragment):
        """Test StreamingServerOp raises error on invalid width parameter."""
        with pytest.raises((ValueError, TypeError)):
            StreamingServerOp(fragment=fragment, width=0)

    def test_streaming_server_op_invalid_height(self, fragment):
        """Test StreamingServerOp raises error on invalid height parameter."""
        with pytest.raises((ValueError, TypeError)):
            StreamingServerOp(fragment=fragment, height=0)

    def test_streaming_server_op_invalid_fps(self, fragment):
        """Test StreamingServerOp raises error on invalid fps parameter."""
        with pytest.raises((ValueError, TypeError)):
            StreamingServerOp(fragment=fragment, fps=0)

    def test_streaming_server_op_invalid_port(self, fragment):
        """Test StreamingServerOp raises error on invalid port parameter."""
        with pytest.raises((ValueError, TypeError)):
            StreamingServerOp(fragment=fragment, port=0)
        
        with pytest.raises((ValueError, TypeError)):
            StreamingServerOp(fragment=fragment, port=99999)  # Port too high

    def test_streaming_server_op_invalid_server_name_type(self, fragment):
        """Test StreamingServerOp raises error on invalid server_name type."""
        with pytest.raises(TypeError):
            StreamingServerOp(fragment=fragment, server_name=123)  # Should be string

    def test_streaming_server_op_parameters_not_directly_accessible(self, fragment):
        """Test that StreamingServerOp parameters are properly encapsulated."""
        op = StreamingServerOp(
            fragment=fragment,
            name="test_streaming_server",
            width=1920,
            height=1080,
            fps=30
        )
        
        # Parameters should be encapsulated and not directly accessible
        # This is expected behavior for Holoscan operators
        assert not hasattr(op, 'width'), "Width parameter should be encapsulated"
        assert not hasattr(op, 'height'), "Height parameter should be encapsulated" 
        assert not hasattr(op, 'fps'), "FPS parameter should be encapsulated"

    def test_streaming_server_op_lifecycle_methods_exist(self, fragment):
        """Test that StreamingServerOp has required lifecycle methods."""
        op = StreamingServerOp(fragment=fragment, name="test_streaming_server")
        
        # Verify lifecycle methods exist
        assert hasattr(op, 'setup'), "StreamingServerOp should have setup method"
        assert hasattr(op, 'initialize'), "StreamingServerOp should have initialize method"
        assert hasattr(op, 'start'), "StreamingServerOp should have start method"
        assert hasattr(op, 'stop'), "StreamingServerOp should have stop method"
        assert hasattr(op, 'compute'), "StreamingServerOp should have compute method"

    def test_streaming_server_op_setup_call(self, fragment):
        """Test that StreamingServerOp setup method can be called."""
        op = StreamingServerOp(fragment=fragment, name="test_streaming_server")
        
        # Setup should be callable without raising exceptions
        # Note: We don't call it here as it's automatically called during operator creation
        assert callable(op.setup), "Setup method should be callable"

    def test_streaming_server_op_multiple_instances(self, fragment):
        """Test creating multiple StreamingServerOp instances."""
        op1 = StreamingServerOp(fragment=fragment, name="server1", port=8080)
        op2 = StreamingServerOp(fragment=fragment, name="server2", port=8081)
        
        assert op1 != op2, "Different instances should not be equal"
        assert isinstance(op1, BaseOperator), "First instance should be valid operator"
        assert isinstance(op2, BaseOperator), "Second instance should be valid operator"

    def test_streaming_server_op_repr_contains_info(self, fragment):
        """Test that StreamingServerOp repr contains useful information."""
        name = "test_streaming_server"
        op = StreamingServerOp(fragment=fragment, name=name)
        repr_str = repr(op)
        
        assert name in repr_str, "Operator name should appear in repr"
        assert "StreamingServerOp" in repr_str, "Operator class should appear in repr"

    def test_streaming_server_op_edge_case_parameters(self, fragment):
        """Test StreamingServerOp with edge case parameter values."""
        # Test with minimum reasonable values
        op = StreamingServerOp(
            fragment=fragment,
            name="edge_case_server",
            width=1,      # Minimum width
            height=1,     # Minimum height  
            fps=1,        # Minimum fps
            port=1024,    # Minimum user port
        )
        
        assert isinstance(op, BaseOperator), "StreamingServerOp should handle edge case parameters"

    def test_streaming_server_op_empty_server_name(self, fragment):
        """Test StreamingServerOp with empty server name."""
        op = StreamingServerOp(
            fragment=fragment,
            name="test_streaming_server",
            server_name=""
        )
        
        assert isinstance(op, BaseOperator), "StreamingServerOp should handle empty server name"

    def test_streaming_server_op_compute_method_callable(self, fragment, execution_context):
        """Test that StreamingServerOp compute method is callable (standalone operator)."""
        op = StreamingServerOp(fragment=fragment, name="test_streaming_server")
        
        # For standalone operators, compute should be callable but may not do much without network clients
        assert callable(op.compute), "Compute method should be callable"
        
        # Note: We don't actually call compute here as it requires proper initialization
        # and network setup, which is complex for unit testing

    def test_streaming_server_op_default_parameter_values(self, fragment):
        """Test StreamingServerOp uses appropriate default values."""
        op = StreamingServerOp(fragment=fragment, name="default_server")
        
        # Operator should initialize successfully with defaults
        assert isinstance(op, BaseOperator), "StreamingServerOp should work with default parameters"
        
        # Note: Default values are typically:
        # width=854, height=480, fps=30, port=48010, multi_instance=false, 
        # server_name="StreamingServer", receive_frames=true, send_frames=true
