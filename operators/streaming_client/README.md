# StreamingClient Operator

The StreamingClientOp class implements a Holoscan operator that provides bidirectional video streaming capabilities with the following key components:

- Configuration and Initialization:
- Configurable parameters for frame dimensions (width, height), frame rate (fps), server connection (IP, port)
- Input/output ports for frame data using GXF entities
- Support for both sending and receiving frames through separate flags

Frame Processing Pipeline:
- Input handling: Receives frames as GXF entities containing H.264 encoded video tensors
- Frame conversion: Converts input tensors to VideoFrame objects with BGRA format
- Memory management: Implements safe memory handling with bounds checking and zero-padding
- Output generation: Creates GXF entities with properly configured tensors for downstream processing

Streaming Protocol Implementation:
- Bidirectional streaming support through StreamingClient class
- Frame callback system for receiving frames
- Frame source system for sending frames
- Connection management with server including timeout handling


## Dependencies

In order to build the client operator, you must first download the client binaries from NGC and add to the `lib` directory for the `streaming_client` operator folder

Download the Holoscan Client Cloud Streaming library from NGC:
https://catalog.ngc.nvidia.com/orgs/nvidia/resources/holoscan_client_cloud_streaming

```bash
# Download using NGC CLI
cd <your_holohub_path>/operators/streaming_client
ngc registry resource download-version nvidia/holoscan_client_cloud_streaming:0.1
unzip -o holoscan_client_cloud_streaming_v0.1/holoscan_client_cloud_streaming.zip

# Copy the appropriate architecture libraries to lib/ directory
# For x86_64 systems:
cp lib/x86_64/* lib/
# For aarch64 systems:
# cp lib/aarch64/* lib/

# Clean up architecture-specific directories and NGC download directory
rm -rf lib/x86_64 lib/aarch64
rm -rf holoscan_client_cloud_streaming_v0.1
```

After successful extraction and setup, your `operators/streaming_client` directory structure should look like this:

```
├── CMakeLists.txt
├── FindHoloscanStreaming.cmake
├── include
│   ├── StreamingClient.h
│   └── VideoFrame.h
├── lib
│   ├── libcrypto.so.3
│   ├── libcudart.so.12
│   ├── libcudart.so.12.0.107
│   ├── libNvStreamBase.so
│   ├── libNvStreamingSession.so
│   ├── libNvStreamServer.so
│   ├── libPoco.so
│   ├── libssl.so.3
│   ├── libStreamClientShared.so
│   └── libStreamingClient.so
├── metadata.json
├── NOTICE.txt
├── python
│   ├── CMakeLists.txt
│   └── streaming_client.cpp
├── README.md
├── streaming_client.cpp
├── streaming_client.hpp
```

All dependencies need to be properly installed in the operator directory structure.

## Troubleshooting

If you encounter build errors:
- Make sure all required files are copied to the correct locations
- Check that the libraries have appropriate permissions (644)
- Ensure the directories exist inside the container environment 

## Testing

For comprehensive testing documentation including unit tests, infrastructure tests, functional tests, and C++ tests, see [README_TESTING.md](README_TESTING.md).

**Quick test commands:**
```bash
# Run all StreamingClient tests (4 test types)
./holohub test video_streaming_client --verbose

# Run only unit tests (fastest - <1s)
cd operators/streaming_client && python3 -m pytest tests/test_streaming_client_op.py -v
```

## Supported Platforms

- Linux x86_64
- Linux aarch64
