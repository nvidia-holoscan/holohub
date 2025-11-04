# Video Streaming Operators

A unified package containing both streaming client and server operators for real-time video communication in Holoscan streaming applications.

## Overview

This operator package combines `video_streaming_client` and `video_streaming_server` into a single, cohesive video streaming solution.

## Structure

```
video_streaming/
├── video_streaming_client/    # Complete streaming client operator
│   ├── video_streaming_client.cpp     # Main client implementation
│   ├── frame_saver.cpp          # Frame saving utility
│   └── holoscan_client_cloud_streaming/  # Client streaming binary once NGC download is complete
├── video_streaming_server/    # Complete streaming server operator
│   ├── video_streaming_server_*.cpp   # Server implementations
│   ├── frame_debug_utils.cpp    # Debug utilities
│   └── holoscan_server_cloud_streaming/  # Server streaming binary once NGC download is complete 
├── CMakeLists.txt               # Unified build configuration
├── metadata.json                # Combined metadata
└── README.md                    # This file
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
./holohub build video_streaming \
  --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'
```

## Running the Applications

The video streaming demo provides both client and server applications. For complete documentation and setup instructions, see the **[Applications README](../../applications/video_streaming/README.md)**.

> ⚠️ **Important:** These applications are currently only compatible with CUDA 12.x. If your system uses CUDA 13.x, ensure you add the `--cuda 12` flag to all command-line invocations shown below.

### 1. Start the Streaming Server

You can start the server in either C++ or Python mode.

**C++ Server:**

```bash
./holohub run video_streaming_server --language cpp
```

**Python Server:**

```bash
./holohub run video_streaming_server --language python
```

### 2. Start the Streaming Client (in another terminal)

You can start the client in either C++ or Python mode regardless of the server language.

#### Option A: V4L2 Camera (Webcam)

It uses `video_streaming_client_demo.yaml` and captures video from webcam with 640x480 resolution.

- **C++ V4L2 Camera Client:**

    ```bash
    ./holohub run video_streaming_client v4l2 --language cpp
    ```

- **Python V4L2 Camera Client:**

    ```bash
    ./holohub run video_streaming_client v4l2 --language python
    ```

#### Option B: Video Replayer

It uses `video_streaming_client_demo_replayer.yaml` and replays a pre-recorded video file with 854x480 resolution.

- **C++ Video Replayer Client:**

  ```bash
  ./holohub run video_streaming_client replayer --language cpp
  ```

- **Python Video Replayer Client:**

  ```bash
  ./holohub run video_streaming_client replayer --language python
  ```

> **Note:**
> - The video streaming server and client are **cross-language compatible**. You can start the server in one language (C++ or Python) and then connect with a client running in either language.  
> - You can **switch between the client modes** (replayer or V4L2 camera) at any time without restarting the server—just stop one client and start the other. The server automatically manages client connections.

### Default Configurations

**Server:**
- Port: 48010
- Resolution: 854x480
- Frame Rate: 30 fps

**Client (Video Replayer):**
- Resolution: 854x480
- Frame Rate: 30 fps
- Server: 127.0.0.1:48010

**Client (V4L2 Camera):**
- Resolution: 640x480
- Frame Rate: 30 fps
- Server: 127.0.0.1:48010

For detailed Python application documentation, see:
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

**C++ Unit Tests** for all video streaming operators (31 tests total):
- ✅ **13 tests** for `StreamingClientOp`
- ✅ **8 tests** for `StreamingServerResource`
- ✅ **6 tests** for `StreamingServerUpstreamOp`
- ✅ **4 tests** for `StreamingServerDownstreamOp`
- ✅ Fast execution (~0.13 seconds total)
- ✅ 100% pass rate
- ✅ No network dependencies
- ✅ Tests operator initialization, parameter validation, and resource management

#### Run Instructions

Run all unit tests from the holohub root directory:

```bash
# Run all unit tests with verbose output
./holohub test video_streaming --ctest-options="-R unit_tests -V"
```

**For comprehensive test output examples, expected results, and detailed test information, see [UNIT_TESTS_SUMMARY.md](UNIT_TESTS_SUMMARY.md).**

#### Acceptance Criteria

**Build Acceptance:**
- ✅ CMake configuration succeeds
- ✅ All source files compile without errors
- ✅ Both test executables link successfully
- ✅ No compiler warnings in test code

**Test Execution Acceptance:**
- ✅ All 31 unit tests pass (13 client + 18 server)
- ✅ 100% test pass rate
- ✅ Total execution time ≤ 1 second
- ✅ No test failures or skipped tests
- ✅ No memory leaks (as detected by GTest framework)

**Functional Acceptance:**
- ✅ StreamingClientOp initialization with various parameters
- ✅ StreamingClientOp parameter validation (resolution, fps, network settings)
- ✅ StreamingClientOp setup/cleanup lifecycle
- ✅ StreamingServerResource creation and configuration
- ✅ StreamingServerUpstreamOp and DownstreamOp initialization
- ✅ Shared resource patterns between multiple operators
- ✅ Edge case handling (minimum/maximum resolutions, port numbers)
- ✅ Multiple operator instances can coexist

**Documentation:**
- **[Unit Tests Summary](UNIT_TESTS_SUMMARY.md)** - Comprehensive unit test documentation
- **[Client Tests](video_streaming_client/tests/README.md)** - VideoStreamingClientOp test details
- **[Server Tests](video_streaming_server/tests/README.md)** - Server operator test details

### Integration Tests

End-to-end integration tests validate the complete streaming pipeline with actual server/client communication and frame transmission.

**Documentation:**
- **[Integration Tests](../../applications/video_streaming/TESTING.md)** - End-to-end testing documentation

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
