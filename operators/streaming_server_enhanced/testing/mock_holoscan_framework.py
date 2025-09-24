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

"""Mock Holoscan framework components for isolated testing of StreamingServer operators."""

import numpy as np
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, Optional, List, Union
import logging

logger = logging.getLogger(__name__)


class MockFragment:
    """Mock Holoscan Fragment for testing."""
    
    def __init__(self, name="mock_fragment"):
        self.name = name
        self.operators = []
        self.resources = []
        self.conditions = []
        
    def add_operator(self, operator):
        self.operators.append(operator)
        
    def add_resource(self, resource):
        self.resources.append(resource)


class MockOperatorSpec:
    """Mock OperatorSpec for testing operator setup."""
    
    def __init__(self):
        self.inputs = {}
        self.outputs = {}
        self.parameters = {}
        
    def input(self, name: str):
        """Mock input specification."""
        if name not in self.inputs:
            self.inputs[name] = MockPortSpec(name, "input")
        return self.inputs[name]
        
    def output(self, name: str):
        """Mock output specification."""
        if name not in self.outputs:
            self.outputs[name] = MockPortSpec(name, "output")
        return self.outputs[name]
        
    def param(self, name: str, **kwargs):
        """Mock parameter specification."""
        self.parameters[name] = kwargs
        return self


class MockPortSpec:
    """Mock port specification."""
    
    def __init__(self, name: str, port_type: str):
        self.name = name
        self.port_type = port_type
        self.conditions = []
        
    def condition(self, condition_type, **kwargs):
        """Mock condition specification."""
        self.conditions.append((condition_type, kwargs))
        return self


class MockComponentSpec:
    """Mock ComponentSpec for testing resource setup."""
    
    def __init__(self):
        self.parameters = {}
        
    def param(self, name: str, **kwargs):
        """Mock parameter specification."""
        self.parameters[name] = kwargs
        return self


class MockInputContext:
    """Mock InputContext for testing compute methods."""
    
    def __init__(self, tensor_map=None):
        self._tensor_map = tensor_map or MockTensorMap()
        
    def receive(self, port_name: str = None):
        """Mock receive operation."""
        if port_name:
            return self._tensor_map.get(port_name)
        return self._tensor_map


class MockOutputContext:
    """Mock OutputContext for testing compute methods."""
    
    def __init__(self):
        self.emitted_data = {}
        
    def emit(self, data, port_name: str = None):
        """Mock emit operation."""
        if port_name:
            self.emitted_data[port_name] = data
        else:
            self.emitted_data["default"] = data


class MockExecutionContext:
    """Mock ExecutionContext for testing compute methods."""
    
    def __init__(self):
        self.context_data = {}
        
    def get_input(self, name: str):
        return self.context_data.get(f"input_{name}")
        
    def get_output(self, name: str):
        return self.context_data.get(f"output_{name}")


class MockAllocator:
    """Mock Allocator for testing memory allocation."""
    
    def __init__(self, name="mock_allocator"):
        self.name = name
        self.allocated_buffers = []
        
    def allocate(self, size: int, alignment: int = 1):
        """Mock memory allocation."""
        buffer = np.zeros(size, dtype=np.uint8)
        self.allocated_buffers.append(buffer)
        return buffer
        
    def deallocate(self, buffer):
        """Mock memory deallocation."""
        if buffer in self.allocated_buffers:
            self.allocated_buffers.remove(buffer)


class MockTensor:
    """Mock Tensor for testing tensor operations."""
    
    def __init__(self, data=None, shape=None, dtype=np.uint8):
        if data is not None:
            self._data = np.array(data, dtype=dtype)
            self._shape = self._data.shape
        elif shape is not None:
            self._data = np.zeros(shape, dtype=dtype)
            self._shape = shape
        else:
            # Default tensor (480x854x3 BGR)
            self._data = np.zeros((480, 854, 3), dtype=dtype)
            self._shape = (480, 854, 3)
        self._dtype = dtype
        
    @property
    def data(self):
        """Get tensor data."""
        return self._data
        
    @property
    def shape(self):
        """Get tensor shape."""
        return self._shape
        
    @property
    def dtype(self):
        """Get tensor dtype."""
        return self._dtype
        
    @property
    def size(self):
        """Get tensor size."""
        return self._data.size
        
    @property
    def nbytes(self):
        """Get tensor size in bytes."""
        return self._data.nbytes
        
    def __array__(self):
        """Enable numpy array interface."""
        return self._data
        
    def copy(self):
        """Create a copy of the tensor."""
        return MockTensor(data=self._data.copy())


class MockTensorMap:
    """Mock TensorMap for testing tensor map operations."""
    
    def __init__(self, tensors=None):
        self._tensors = tensors or {}
        
    def get(self, key: str, default=None):
        """Get tensor by key."""
        return self._tensors.get(key, default)
        
    def __getitem__(self, key: str):
        """Get tensor by key (dict-like access)."""
        return self._tensors[key]
        
    def __setitem__(self, key: str, value):
        """Set tensor by key (dict-like access)."""
        self._tensors[key] = value
        
    def __contains__(self, key: str):
        """Check if key exists in tensor map."""
        return key in self._tensors
        
    def keys(self):
        """Get tensor map keys."""
        return self._tensors.keys()
        
    def values(self):
        """Get tensor map values."""
        return self._tensors.values()
        
    def items(self):
        """Get tensor map items."""
        return self._tensors.items()


class MockFrame:
    """Mock Frame for testing streaming operations."""
    
    def __init__(self, width=854, height=480, channels=3, dtype=np.uint8, timestamp=0):
        self.width = width
        self.height = height
        self.channels = channels
        self.dtype = dtype
        self.timestamp = timestamp
        self.format = "BGR"  # Default format
        self._data = np.zeros((height, width, channels), dtype=dtype)
        
    @property
    def data(self):
        """Get frame data."""
        return self._data
        
    @data.setter
    def data(self, value):
        """Set frame data."""
        self._data = np.array(value, dtype=self.dtype)
        
    @property
    def size(self):
        """Get frame size in bytes."""
        return self._data.nbytes
        
    def copy(self):
        """Create a copy of the frame."""
        frame_copy = MockFrame(self.width, self.height, self.channels, self.dtype, self.timestamp)
        frame_copy.data = self._data.copy()
        frame_copy.format = self.format
        return frame_copy


class MockStreamingServer:
    """Mock StreamingServer for testing streaming operations."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self._is_running = False
        self._has_clients = False
        self.received_frames = []
        self.sent_frames = []
        self.event_callbacks = []
        
    def start(self):
        """Mock server start."""
        self._is_running = True
        logger.info("Mock StreamingServer started")
        
    def stop(self):
        """Mock server stop."""
        self._is_running = False
        logger.info("Mock StreamingServer stopped")
        
    def is_running(self):
        """Check if server is running."""
        return self._is_running
        
    def has_connected_clients(self):
        """Check if clients are connected."""
        return self._has_clients
        
    def send_frame(self, frame):
        """Mock frame sending."""
        self.sent_frames.append(frame)
        logger.debug(f"Mock frame sent: {frame.width}x{frame.height}")
        
    def receive_frame(self):
        """Mock frame receiving."""
        if self.received_frames:
            frame = self.received_frames.pop(0)
            logger.debug(f"Mock frame received: {frame.width}x{frame.height}")
            return frame
        return None
        
    def try_receive_frame(self, frame):
        """Mock try receive frame."""
        received = self.receive_frame()
        if received:
            frame.data = received.data
            frame.width = received.width
            frame.height = received.height
            frame.timestamp = received.timestamp
            return True
        return False
        
    def set_event_callback(self, callback):
        """Mock event callback registration."""
        self.event_callbacks.append(callback)
        
    def simulate_client_connection(self):
        """Simulate client connection for testing."""
        self._has_clients = True
        for callback in self.event_callbacks:
            try:
                event = Mock()
                event.type = "CLIENT_CONNECTED"
                event.data = {}
                callback(event)
            except Exception as e:
                logger.warning(f"Error in event callback: {e}")
                
    def simulate_client_disconnection(self):
        """Simulate client disconnection for testing."""
        self._has_clients = False
        for callback in self.event_callbacks:
            try:
                event = Mock()
                event.type = "CLIENT_DISCONNECTED"
                event.data = {}
                callback(event)
            except Exception as e:
                logger.warning(f"Error in event callback: {e}")
                
    def add_mock_received_frame(self, frame):
        """Add a frame to be received for testing."""
        self.received_frames.append(frame)


# Mock for holoscan namespace classes
class MockHoloscanOperator:
    """Mock base Holoscan Operator class."""
    
    def __init__(self, fragment=None, name="mock_operator", **kwargs):
        self.fragment = fragment
        self.name = name
        self.kwargs = kwargs
        self._spec = MockOperatorSpec()
        
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


class MockHoloscanResource:
    """Mock base Holoscan Resource class."""
    
    def __init__(self, fragment=None, name="mock_resource", **kwargs):
        self.fragment = fragment
        self.name = name
        self.kwargs = kwargs
        self._spec = MockComponentSpec()
        
    @property
    def spec(self):
        return self._spec
        
    def setup(self, spec):
        pass
        
    def initialize(self):
        pass


def create_mock_bgr_frame(width=854, height=480, pattern="gradient", frame_number=1):
    """Create a mock BGR frame with various patterns for testing."""
    frame = MockFrame(width, height, 3, np.uint8)
    
    if pattern == "gradient":
        # Create gradient pattern
        for y in range(height):
            for x in range(width):
                offset = (frame_number * 10) % 256
                frame.data[y, x, 0] = min(255, ((x + offset) * 255) // width)      # Blue
                frame.data[y, x, 1] = min(255, ((y + offset) * 255) // height)     # Green  
                frame.data[y, x, 2] = min(255, (((x + y) + offset) * 255) // (width + height))  # Red
                
    elif pattern == "checkerboard":
        # Create checkerboard pattern
        square_size = 40
        for y in range(height):
            for x in range(width):
                square_x = x // square_size
                square_y = y // square_size
                is_white = (square_x + square_y + frame_number) % 2 == 0
                color_value = 255 if is_white else 0
                frame.data[y, x] = [color_value, color_value, color_value]
                
    elif pattern == "solid":
        # Create solid color based on frame number
        colors = [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
        ]
        color = colors[frame_number % len(colors)]
        frame.data[:, :] = color
        
    elif pattern == "noise":
        # Create noise pattern
        np.random.seed(frame_number)  # Deterministic noise
        frame.data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
    else:  # default to black
        frame.data.fill(0)
        
    frame.timestamp = frame_number * 1000  # Mock timestamp in ms
    return frame


def create_mock_tensor_from_frame(frame):
    """Convert a mock frame to a mock tensor."""
    return MockTensor(data=frame.data)


def create_mock_frame_from_tensor(tensor):
    """Convert a mock tensor to a mock frame."""
    data = np.array(tensor.data)
    if len(data.shape) == 3:
        height, width, channels = data.shape
    else:
        raise ValueError("Expected 3D tensor for frame conversion")
        
    frame = MockFrame(width, height, channels)
    frame.data = data
    return frame
