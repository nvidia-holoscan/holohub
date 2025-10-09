# Video Streaming Operators

A unified package containing both streaming client and server operators for real-time video communication in Holoscan streaming applications.

## Overview

This operator package combines `streaming_client_enhanced` and `streaming_server_enhanced` into a single, cohesive video streaming solution.

## Structure

```
video_streaming/
├── streaming_client_enhanced/    # Complete streaming client operator
│   ├── streaming_client.cpp     # Main client implementation
│   ├── frame_saver.cpp          # Frame saving utility
│   └── holoscan_client_cloud_streaming/  # Client streaming binary once NGC download is complete
├── streaming_server_enhanced/    # Complete streaming server operator
│   ├── streaming_server_*.cpp   # Server implementations
│   ├── frame_debug_utils.cpp    # Debug utilities
│   └── holoscan_server_cloud_streaming/  # Server streaming binary once NGC download is complete 
├── CMakeLists.txt               # Unified build configuration
├── metadata.json                # Combined metadata
└── README.md                    # This file
```

## Components

### Streaming Client Enhanced (`streaming_client_enhanced/`)

The client component provides bidirectional video streaming capabilities:

- **StreamingClientOp**: Main operator for video streaming client functionality
- **FrameSaverOp**: Utility operator for saving frames to disk
- **Features**: 
  - Send and receive video frames
  - V4L2 camera support
  - Frame validation and debugging

**Documentation**: See `streaming_client_enhanced/README.md` for detailed information.

### Streaming Server Enhanced (`streaming_server_enhanced/`)

The server component provides comprehensive streaming server functionality:

- **StreamingServerResource**: Shared resource managing server connections
- **StreamingServerUpstreamOp**: Handles incoming video streams from clients
- **StreamingServerDownstreamOp**: Handles outgoing video streams to clients
- **Features**:
  - Multi-client support
  - Format conversion utilities
  - Frame processing and validation
  - Debug utilities for troubleshooting

**Documentation**: See `streaming_server_enhanced/README.md` for detailed information.

## Usage


### In Applications

#### CMakeLists.txt
```cmake
add_holohub_application(my_streaming_app DEPENDS OPERATORS video_streaming)
```

#### C++ Applications
```cpp
// Client functionality
#include "streaming_client.hpp"
#include "frame_saver.hpp"

// Server functionality  
#include "streaming_server_resource.hpp"
#include "streaming_server_upstream_op.hpp"
#include "streaming_server_downstream_op.hpp"
```

#### Python Applications

Both client and server operators have Python bindings available through pybind11:

```python
# Client functionality
from holohub.streaming_client_enhanced import StreamingClientOp

# Server functionality
from holohub.streaming_server_enhanced import (
    StreamingServerResource,
    StreamingServerUpstreamOp,
    StreamingServerDownstreamOp
)
```

**Building with Python Support:**
```bash
# When building applications that use Python bindings
./holohub build your_app --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'

# When running Python applications
./holohub run your_app --language python --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'
```

**Example Python Application:**
```python
from holoscan.core import Application
from holoscan.resources import UnboundedAllocator
from holohub.streaming_server_enhanced import (
    StreamingServerResource,
    StreamingServerUpstreamOp,
    StreamingServerDownstreamOp
)

class MyStreamingApp(Application):
    def compose(self):
        # Create allocator
        allocator = UnboundedAllocator(self, name="allocator")
        
        # Create streaming server resource
        streaming_resource = StreamingServerResource(
            self,
            name="streaming_server_resource",
            port=48010,
            width=854,
            height=480,
            fps=30,
            enable_upstream=True,
            enable_downstream=True
        )
        
        # Create upstream operator (receives from clients)
        upstream_op = StreamingServerUpstreamOp(
            self,
            name="upstream_op",
            streaming_server_resource=streaming_resource
        )
        
        # Create downstream operator (sends to clients)
        downstream_op = StreamingServerDownstreamOp(
            self,
            name="downstream_op",
            streaming_server_resource=streaming_resource
        )
        
        # Connect the pipeline
        self.add_flow(upstream_op, downstream_op, {("output_frames", "input_frames")})
```

For complete Python application examples, see:
- **[Python Streaming Server Demo](../../applications/video_streaming_demo_enhanced/video_streaming_demo_server/python/streaming_server_demo.py)**
- **[Python Streaming Client Demo](../../applications/video_streaming_demo_enhanced/video_streaming_demo_client/python/streaming_client_demo.py)**


## Dependencies

### Required
- **Holoscan SDK 3.5+**: Core framework
- **CUDA**: GPU acceleration support

### Cloud Streaming Binaries
Both client and server require their respective NGC binaries:

```bash
# Client binary
ngc registry resource download-version nvidia/holoscan_client_cloud_streaming:0.2

# Server binary  
ngc registry resource download-version nvidia/holoscan_server_cloud_streaming:0.2
```

## Python Bindings

Both client and server operators provide Python bindings built with pybind11, enabling use in Python-based Holoscan applications.

### Building Python Bindings

Python bindings are automatically built when `-DHOLOHUB_BUILD_PYTHON=ON` is specified:

```bash
# Build with Python support
./holohub build video_streaming_demo_enhanced --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'
```

### Python Module Structure

The Python bindings are organized into two modules:

**Server Module (`holohub.streaming_server_enhanced`):**
- `StreamingServerResource`: Shared resource managing server connections
- `StreamingServerUpstreamOp`: Operator for receiving frames from clients
- `StreamingServerDownstreamOp`: Operator for sending frames to clients

**Client Module (`holohub.streaming_client_enhanced`):**
- `StreamingClientOp`: Operator for bidirectional video streaming client

### Using Python Bindings

**Import the operators:**
```python
from holohub.streaming_server_enhanced import (
    StreamingServerResource,
    StreamingServerUpstreamOp,
    StreamingServerDownstreamOp
)
from holohub.streaming_client_enhanced import StreamingClientOp
```

**Create operator instances:**
```python
# Server operators
streaming_resource = StreamingServerResource(
    self,
    name="streaming_server_resource",
    port=48010,
    width=854,
    height=480,
    fps=30
)

upstream_op = StreamingServerUpstreamOp(
    self,
    name="upstream_op",
    streaming_server_resource=streaming_resource
)

downstream_op = StreamingServerDownstreamOp(
    self,
    name="downstream_op",
    streaming_server_resource=streaming_resource
)

# Client operator
streaming_client = StreamingClientOp(
    self,
    allocator,  # Memory allocator for output buffers
    name="streaming_client",
    width=854,
    height=480,
    fps=30,
    server_ip="127.0.0.1",
    signaling_port=48010,
    send_frames=True,
    receive_frames=True
)
```

### Python Examples

Complete working examples are available in the demo applications:

- **[Python Server Application](../../applications/video_streaming_demo_enhanced/video_streaming_demo_server/python/streaming_server_demo.py)** - Full Python server implementation
- **[Python Client Application](../../applications/video_streaming_demo_enhanced/video_streaming_demo_client/python/streaming_client_demo.py)** - Full Python client implementation

### Running Python Applications

```bash
# Run Python server
./holohub run video_streaming_demo_server --language python \
  --run-args='--port 48010' \
  --configure-args='-DHOLOHUB_BUILD_PYTHON=ON' \
  --docker-opts='-e EnableHybridMode=1'

# Run Python client with replayer
./holohub run video_streaming_demo_client --language python \
  --run-args='--port 48010 --source replayer --width 854 --height 480' \
  --configure-args='-DHOLOHUB_BUILD_PYTHON=ON' \
  --docker-opts='-e EnableHybridMode=1'

# Run Python client with V4L2 camera
./holohub run video_streaming_demo_client --language python \
  --run-args='--port 48010 --source v4l2 --width 640 --height 480' \
  --configure-args='-DHOLOHUB_BUILD_PYTHON=ON' \
  --docker-opts='-e EnableHybridMode=1'
```

### Python Integration Testing

A comprehensive Python integration test validates the Python bindings and applications:

```bash
# Run Python integration test
cd applications/video_streaming_demo_enhanced
./integration_test_python.sh
```

The test validates:
- Python bindings build successfully
- Python server starts and remains stable
- Python client connects to server
- Bidirectional streaming works correctly
- Both processes run for 30+ seconds without errors

For more details, see the [Integration Testing documentation](../../applications/video_streaming_demo_enhanced/README.md#python-integration-test-new).

## Testing

Testing is handled at the application level through the unified `video_streaming_demo_enhanced` integration tests:

- **C++ Integration Test**: Validates C++ server and client applications
- **Python Integration Test**: Validates Python bindings, server, and client applications

Both tests provide end-to-end validation of client and server components working together in bidirectional streaming scenarios.

## Related Applications

- **[Streaming Client Demo Enhanced](../../applications/video_streaming_demo_enhanced/video_streaming_demo_client/)**: Example client application
- **[Streaming Server Demo Enhanced](../../applications/video_streaming_demo_enhanced/video_streaming_demo_server/)**: Example server application


## Performance Notes

- Both components support GPU memory allocation for optimal performance
- Configure appropriate buffer sizes for your streaming requirements
- Monitor network bandwidth for remote streaming scenarios
- Use debug utilities to troubleshoot frame processing issues

## License

Apache-2.0 - See the LICENSE file for details.
