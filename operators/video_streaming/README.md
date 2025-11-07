# Video Streaming Operators

A unified package containing both streaming client and server operators for real-time video communication in Holoscan streaming applications.

## Overview

This operator package combines `video_streaming_client` and `video_streaming_server` into a single, cohesive video streaming solution.

## Structure

```text
video_streaming/
├── video_streaming_client/               # Complete streaming client operator
│   ├── video_streaming_client.cpp        # Main client implementation
│   ├── frame_saver.cpp                   # Frame saving utility
│   └── holoscan_client_cloud_streaming/  # Client streaming binary once NGC download is complete
├── video_streaming_server/               # Complete streaming server operator
│   ├── video_streaming_server_*.cpp      # Server implementations
│   ├── frame_debug_utils.cpp             # Debug utilities
│   └── holoscan_server_cloud_streaming/  # Server streaming binary once NGC download is complete 
├── CMakeLists.txt                        # Unified build configuration
├── metadata.json                         # Combined metadata
└── README.md                             # This file
```

## Components

### Video Streaming Client (`video_streaming_client/`)

The client component provides bidirectional video streaming capabilities:

- **VideoStreamingClientOp**: Main operator for video streaming client functionality
- **FrameSaverOp**: Utility operator for saving frames to disk
- **Features**:
  - Send and receive video frames
  - V4L2 camera support
  - Frame validation and debugging

**Documentation**: See `video_streaming_client/README.md` for detailed information.

### Streaming Server (`video_streaming_server/`)

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
#include "video_streaming_client.hpp"
#include "frame_saver.hpp"

// Server functionality  
#include "video_streaming_server_resource.hpp"
#include "video_streaming_server_upstream_op.hpp"
#include "video_streaming_server_downstream_op.hpp"
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
./holohub build video_streaming --language python
```

## Running the Applications

The video streaming demo provides both client and server applications. For complete documentation and setup instructions, see the **[Applications README](../../applications/video_streaming/README.md)**.

> ⚠️ **Important:** These applications are currently only compatible with CUDA 12.x. If your system uses CUDA 13.x, ensure you add the `--cuda 12` flag to all command-line invocations shown below.

**For complete Python application documentation, see:**

- **[Server Application (C++ and Python)](../../applications/video_streaming/video_streaming_server/README.md)**
- **[Client Application (C++ and Python)](../../applications/video_streaming/video_streaming_client/README.md)**

## Dependencies

### Required

- **Holoscan SDK 3.5.0 or higher**: Core framework
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

This package includes comprehensive testing at multiple levels:

### Unit Tests

**For comprehensive test output examples, expected results, and detailed test information, see [UNIT_TESTS_SUMMARY.md](UNIT_TESTS_SUMMARY.md).**

#### Run Instructions

Run all unit tests from the holohub root directory:

```bash
# Run all unit tests with verbose output
./holohub test video_streaming --ctest-options="-R unit_tests -V"
```

### Integration Tests

Please refer to the [Integration Tests](../../applications/video_streaming/TESTING.md) for end-to-end integration tests validate the complete streaming pipeline with actual server/client communication and frame transmission.


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
