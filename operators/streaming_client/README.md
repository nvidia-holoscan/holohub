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
# Download and extract the library
ngc registry resource download-version nvidia/holoscan_client_cloud_streaming:0.1
# Move the extracted files to the lib directory
mv holoscan_client_cloud_streaming lib
```

All dependencies need to be properly installed in the operator directory structure.

## Troubleshooting

If you encounter build errors:
- Make sure all required files are copied to the correct locations
- Check that the libraries have appropriate permissions (644)
- Ensure the directories exist inside the container environment 

## Supported Platforms

- Linux x86_64
