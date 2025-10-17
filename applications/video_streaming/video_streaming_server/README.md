# Video Streaming Demo Server Enhanced

This is the enhanced video streaming server demo application that provides bidirectional streaming capabilities.

## Features

- **Bidirectional Streaming**: Receive and send video frames
- **Multiple Client Support**: Handle multiple streaming clients
- **Configurable Resolution**: Support for different video resolutions
- **Real-time Processing**: Low-latency video streaming

## Running the Application

> [!IMPORTANT] The server applications requires Holoscan SDK 3.5.0. Either set the SDK version environment variable before running the applications, or use the `--base-img` option to specify the base image.
>
> ```bash
> # Set SDK version environment variable
> export HOLOHUB_BASE_SDK_VERSION=3.5.0
> ```

```bash
./holohub run video_streaming
```

## Configuration

- `streaming_server_demo.yaml`: Server configuration file
- Default settings: 854x480 resolution, 30 FPS, port 48010

## Dependencies

- Holoscan SDK 3.5.0
- video_streaming operator
- OpenCV
- CUDA 12.x (currently not working with CUDA 13.x)
- OpenSSL 3.0+

## Usage with Client

1. Start the server first
2. Start one or more clients in separate terminals
3. The server will handle bidirectional communication with all connected clients
