# Video Streaming Operators

A unified package containing both streaming client and server operators for real-time video communication in Holoscan streaming applications.

## Overview

This operator package combines `video_streaming_client` and `video_streaming_server` into a single, cohesive video streaming solution.

## Structure

```
video_streaming/
├── video_streaming_client/    # Complete streaming client operator
│   ├── streaming_client.cpp     # Main client implementation
│   ├── frame_saver.cpp          # Frame saving utility
│   └── holoscan_client_cloud_streaming/  # Client streaming binary once NGC download is complete
├── video_streaming_server/    # Complete streaming server operator
│   ├── streaming_server_*.cpp   # Server implementations
│   ├── frame_debug_utils.cpp    # Debug utilities
│   └── holoscan_server_cloud_streaming/  # Server streaming binary once NGC download is complete 
├── CMakeLists.txt               # Unified build configuration
├── metadata.json                # Combined metadata
└── README.md                    # This file
```

## Components

### Streaming Client Enhanced (`video_streaming_client/`)

The client component provides bidirectional video streaming capabilities:

- **VideoStreamingClientOp**: Main operator for video streaming client functionality
- **FrameSaverOp**: Utility operator for saving frames to disk
- **Features**: 
  - Send and receive video frames
  - V4L2 camera support
  - Frame validation and debugging

**Documentation**: See `video_streaming_client/README.md` for detailed information.

### Streaming Server Enhanced (`video_streaming_server/`)

The server component provides comprehensive streaming server functionality:

- **StreamingServerResource**: Shared resource managing server connections
- **StreamingServerUpstreamOp**: Handles incoming video streams from clients
- **StreamingServerDownstreamOp**: Handles outgoing video streams to clients
- **Features**:
  - Multi-client support
  - Format conversion utilities
  - Frame processing and validation
  - Debug utilities for troubleshooting

**Documentation**: See `video_streaming_server/README.md` for detailed information.

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

Both client and server operators have Python bindings available. To use them in Python:

```python
# Client functionality
from holohub.video_streaming_client import VideoStreamingClientOp

# Server functionality
from holohub.video_streaming_server import (
    StreamingServerResource,
    StreamingServerUpstreamOp,
    StreamingServerDownstreamOp,
)
```

**Building with Python support:**
```bash
./holohub build video_streaming \
  --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'
```

**Running Python applications:**
```bash
# Python Server
./holohub run video_streaming server_python \
  --docker-file applications/video_streaming/Dockerfile \
  --docker-opts='-e EnableHybridMode=1' \
  --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'

# Python Client (Video Replayer)
./holohub run video_streaming client_python \
  --docker-file applications/video_streaming/Dockerfile \
  --docker-opts='-e EnableHybridMode=1' \
  --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'

# Python Client (V4L2 Camera)
./holohub run video_streaming client_python_v4l2 \
  --docker-file applications/video_streaming/Dockerfile \
  --docker-opts='-e EnableHybridMode=1' \
  --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'
```

For detailed Python application documentation, see:
- **[Server Application (C++ and Python)](../../applications/video_streaming/video_streaming_server/README.md)**
- **[Client Application (C++ and Python)](../../applications/video_streaming/video_streaming_client/README.md)**

## Dependencies

### Required
- **Holoscan SDK 3.5.0**: Core framework
- **CUDA 12.x**: GPU acceleration support

### Cloud Streaming Binaries

#### Client Binary

To build the client operator, first download the client binaries from NGC:

```bash
# Download using NGC CLI

cd <your_holohub_path>/operators/video_streaming/video_streaming_client
ngc registry resource download-version "nvidia/holoscan_client_cloud_streaming:0.2"
unzip -o holoscan_client_cloud_streaming_v0.2/holoscan_client_cloud_streaming.zip -d holoscan_client_cloud_streaming

# Clean up NGC download directory
rm -rf ./holoscan_client_cloud_streaming_v0.2/
```

#### Server Binary

To build the server operator, first download the server binaries from NGC:

```bash
# Download using NGC CLI

cd <your_holohub_path>/operators/video_streaming/video_streaming_server
ngc registry resource download-version "nvidia/holoscan_server_cloud_streaming:0.2"
unzip -o holoscan_server_cloud_streaming_v0.2/holoscan_server_cloud_streaming.zip -d holoscan_server_cloud_streaming

# Clean up NGC download directory
rm -rf ./holoscan_server_cloud_streaming_v0.2/
```

All dependencies need to be properly installed in the operator directory structure.

## Testing

Testing is handled at the application level through the unified `video_streaming` integration test, which provides end-to-end validation of both client and server components working together.

## Related Applications

- **[Streaming Client Demo](../../applications/video_streaming/video_streaming_client/)**: Example client application
- **[Streaming Server Demo](../../applications/video_streaming/video_streaming_server/)**: Example server application


## Performance Notes

- Both components support GPU memory allocation for optimal performance
- Configure appropriate buffer sizes for your streaming requirements
- Monitor network bandwidth for remote streaming scenarios
- Use debug utilities to troubleshoot frame processing issues

## License

Apache-2.0 - See the LICENSE file for details.
