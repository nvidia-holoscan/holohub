# Video Streaming Demo Server Enhanced

This is the enhanced video streaming server demo application that provides bidirectional streaming capabilities.

## Features

- **Bidirectional Streaming**: Receive and send video frames
- **Multiple Client Support**: Handle multiple streaming clients
- **Configurable Resolution**: Support for different video resolutions
- **Real-time Processing**: Low-latency video streaming

## Running the Application

```bash
# From holohub root directory
./holohub run --docker-opts='-e EnableHybridMode=1' video_streaming_demo_server --language cpp
```

## Configuration

- `streaming_server_demo.yaml`: Server configuration file
- Default settings: 854x480 resolution, 30 FPS, port 48010

## Dependencies

- Holoscan SDK 2.7.0+
- video_streaming operator
- OpenCV
- CUDA Toolkit
- OpenSSL 3.0+

## Usage with Client

1. Start the server first
2. Start one or more clients in separate terminals
3. The server will handle bidirectional communication with all connected clients
