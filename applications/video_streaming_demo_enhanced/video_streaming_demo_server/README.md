# Video Streaming Demo Server Enhanced

This is the enhanced video streaming server demo application that provides bidirectional streaming capabilities.

## Features

- **Bidirectional Streaming**: Receive and send video frames
- **Multiple Client Support**: Handle multiple streaming clients
- **Configurable Resolution**: Support for different video resolutions
- **Real-time Processing**: Low-latency video streaming

## Running the Application

**⚠️ Important: Enhanced applications require Holoscan SDK 3.5.0. Use the base image parameter to ensure compatibility.**

```bash
# From holohub root directory - with Holoscan 3.5.0 base image
./holohub run --docker-opts='-e EnableHybridMode=1' --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu video_streaming_demo_server --language cpp
```

## Configuration

- `streaming_server_demo.yaml`: Server configuration file
- Default settings: 854x480 resolution, 30 FPS, port 48010

## Dependencies

- Holoscan SDK 3.5.0+
- video_streaming operator
- OpenCV
- CUDA Toolkit
- OpenSSL 3.0+

## Usage with Client

1. Start the server first
2. Start one or more clients in separate terminals
3. The server will handle bidirectional communication with all connected clients
