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
  - Multiple video formats (BGR, BGRA, RGBA)
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

## Testing

Testing is handled at the application level through the unified `video_streaming_demo_enhanced` integration test, which provides end-to-end validation of both client and server components working together.

## Related Applications

- **[Streaming Client Demo Enhanced](../../applications/video_streaming_demo_enhanced/video_streaming_demo_client/)**: Example client application
- **[Streaming Server Demo Enhanced](../../applications/video_streaming_demo_enhanced/video_streaming_demo_server/)**: Example server application

## Migration Notes

### From Separate Operators

If migrating from the previous separate operators:

1. **CMakeLists.txt**: Update dependency from `streaming_client_enhanced` and `streaming_server_enhanced` to `video_streaming`
2. **Include paths**: No changes needed - original paths are preserved

### Benefits of Unification

- **Simplified Dependencies**: Single operator dependency instead of two
- **Coordinated Releases**: Client and server components stay in sync  
- **Unified Documentation**: Comprehensive overview of the streaming ecosystem
- **Easier Maintenance**: Single build target and configuration

## Performance Notes

- Both components support GPU memory allocation for optimal performance
- Configure appropriate buffer sizes for your streaming requirements
- Monitor network bandwidth for remote streaming scenarios
- Use debug utilities to troubleshoot frame processing issues

## License

Apache-2.0 - See the LICENSE file for details.
