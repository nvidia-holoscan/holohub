# Video Streaming Demo

This unified application demonstrates how to use the Holoscan SDK to create both streaming client and server applications for bidirectional video communication. This demo application demonstrates bidirectional video communication between client and server with real-time visualization.

![Video Streaming Demo](screenshot_streaming_demo.png)  
*Fig. 1: Example of surgical video streaming with bidirectional communication showing the client receiving and displaying frames from the server.*

## Overview

The video streaming demo provides:

- **Streaming Client**: Captures video from V4L2 cameras or video files and streams to a server
- **Streaming Server**: Comprehensive server architecture with three main components:
  - **StreamingServerResource**: Manages server connections and client lifecycle
  - **StreamingServerUpstreamOp**: Receives video streams from clients
  - **StreamingServerDownstreamOp**: Sends video frames back to clients (passthrough/echo mode)
- **Bidirectional Communication**: Both sending and receiving video frames
- **Multiple Source Support**: V4L2 cameras, video replay files

## Requirements

- NVIDIA GPU
- CUDA 12.x (currently not working with CUDA 13.x)
- Holoscan SDK 3.5.0+
- V4L2 camera (optional, for live streaming)

### Client Dependencies

Download the client streaming binaries from NGC:

```bash
# Navigate to the client operator directory from the holohub root directory
cd operators/video_streaming/video_streaming_client

# Download using NGC CLI
ngc registry resource download-version "nvidia/holoscan_client_cloud_streaming:0.2"
unzip -o holoscan_client_cloud_streaming_v0.2/holoscan_client_cloud_streaming.zip -d holoscan_client_cloud_streaming

# Clean up
rm -rf holoscan_client_cloud_streaming_v0.2
cd - # Return to the original directory
```

### Server Dependencies

Download the server streaming binaries from NGC:

```bash
# Navigate to the server operator directory from the holohub root directory
cd operators/video_streaming/video_streaming_server

# Download using NGC CLI
ngc registry resource download-version "nvidia/holoscan_server_cloud_streaming:0.2"
unzip -o holoscan_server_cloud_streaming_v0.2/holoscan_server_cloud_streaming.zip -d holoscan_server_cloud_streaming

# Clean up
rm -rf holoscan_server_cloud_streaming_v0.2
cd - # Return to the original directory
```

## Running the Applications

The unified application provides both client and server applications.

> ⚠️ Both client and server applications require Holoscan SDK 3.5.0. Set the SDK version environment variable before running the applications in each terminal, or use the `--base-img` option to specify the base image.
>
> ```bash
> # Set SDK version environment variable
> export HOLOHUB_BASE_SDK_VERSION=3.5.0
> ```
>
> ℹ️ The client requires OpenSSL 3.4.0, which is installed inside the custom Dockerfile.

### 1. Start the Streaming Server

```bash
./holohub run video_streaming_server --language cpp
```

### 2. Start the Streaming Client (in another terminal)

- **Option A: V4L2 Camera (Webcam)**, which uses `streaming_client_demo.yaml` and captures video from webcam with 640x480 resolution.

  ```bash
  ./holohub run video_streaming_client v4l2 --language cpp
  ```

- **Option B: Video Replayer**, which uses `streaming_client_demo_replayer.yaml` and replays a pre-recorded video file with 854x480 resolution.

  ```bash
  ./holohub run video_streaming_client replayer --language cpp
  ```

**Note:** To run the applications in Python, you just need to replace the `--language cpp` with `--language python`.

### Python Bindings

The Python applications use these Holoscan operator bindings:

**Server Components:**

- `holohub.streaming_server_enhanced.StreamingServerResource` - Manages server connections
- `holohub.streaming_server_enhanced.StreamingServerUpstreamOp` - Receives frames from clients
- `holohub.streaming_server_enhanced.StreamingServerDownstreamOp` - Sends frames to clients

**Client Components:**

- `holohub.streaming_client_enhanced.VideoStreamingClientOp` - Bidirectional client streaming

**Holoscan Core Operators:**

- `holoscan.operators.VideoStreamReplayerOp` - Video file playback
- `holoscan.operators.V4L2VideoCaptureOp` - Webcam capture
- `holoscan.operators.FormatConverterOp` - Format conversion (RGBA→RGB→BGR)
- `holoscan.operators.HolovizOp` - Visualization

### Python Implementation Overview

**Server Architecture:**

- `StreamingServerResource` manages streaming connections and client lifecycle
- `StreamingServerUpstreamOp` receives frames from clients (output port: `output_frames`)
- `StreamingServerDownstreamOp` sends frames to clients (input port: `input_frames`)
- Simple pipeline: `upstream_op → downstream_op` (passthrough/echo mode)

**Client Architecture:**

- Video source: `VideoStreamReplayerOp` or `V4L2VideoCaptureOp`
- `FormatConverterOp` handles format conversion (RGBA→RGB→BGR)
- `VideoStreamingClientOp` manages bidirectional streaming (send and receive)
- `HolovizOp` visualizes received frames

**For complete Python implementation examples and code**, see:

- **[Server README](video_streaming_server/README.md)** - Full Python server implementation
- **[Client README](video_streaming_client/README.md)** - Full Python client implementation (replayer and V4L2 modes)

### Command Line Options (Python)

**Server Options:**

- `--port PORT`: Server port (default: 48010)
- `--width WIDTH`: Frame width (default: 854)
- `--height HEIGHT`: Frame height (default: 480)
- `--fps FPS`: Frames per second (default: 30)
- `--config PATH` or `-c PATH`: Path to YAML configuration file
- `--create-config PATH`: Create default configuration file at specified path
- `--help`: Show help message

**Client Options:**

- `--source {replayer,v4l2}`: Video source type (default: replayer)
- `--server-ip IP`: Server IP address (default: 127.0.0.1)
- `--port PORT`: Server port (default: 48010)
- `--width WIDTH`: Frame width (default: 854 for replayer, 640 for v4l2)
- `--height HEIGHT`: Frame height (default: 480)
- `--fps FPS`: Frames per second (default: 30)
- `--config PATH` or `-c PATH`: Path to YAML configuration file
- `--help`: Show help message

### Compatibility

- ✅ **Python server ↔ C++ client** - Fully compatible and tested
- ✅ **Python client ↔ C++ server** - Fully compatible and tested
- ✅ **Python server ↔ Python client** - Fully compatible and tested
- ✅ **C++ server ↔ C++ client** - Fully compatible and tested
- ✅ **All combinations are fully supported** - Mix and match as needed

### Cross-Language Compatibility Testing

Python clients are fully compatible with C++ servers and vice versa:

Terminal 1 - C++ Server:

```bash
./holohub run video_streaming
```

Terminal 2 - Python Client:

```bash
./holohub run video_streaming client_python
```

### Python Troubleshooting

**Import Error:**

- Ensure Holoscan SDK Python bindings are installed
- Verify build with: `./holohub build video_streaming --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'`

**Camera Not Found:**

- Check V4L2 device path: `ls -l /dev/video*`
- Test camera: `v4l2-ctl --device=/dev/video0 --info`

**Connection Failed:**

- Verify server is running and ports are correct
- Check: `netstat -tlnp | grep 48010`

**Video Files Not Found:**

- Check data directory path: `/workspace/holohub/data/endoscopy/`
- Ensure video files exist in the data directory

**Resolution Mismatch:**

- Replayer default: 854x480
- V4L2 default: 640x480
- Server default: 854x480
- Ensure client and server resolutions match

### Configuration Files

**Python Server:** `python/streaming_server_demo.yaml`
**Python Client (Replayer):** `python/streaming_client_demo_replayer.yaml`
**Python Client (V4L2):** `python/streaming_client_demo.yaml`

### Detailed Documentation

For complete implementation details, see the component-specific READMEs:

- **[Server README](video_streaming_server/README.md)** - Complete server documentation (C++ and Python)
- **[Client README](video_streaming_client/README.md)** - Complete client documentation (C++ and Python)

## Command Line Options

### Server Options

- `-h, --help`: Show help message
- `-c, --config <file>`: Configuration file path (default: streaming_server_demo.yaml)
- `-d, --data <directory>`: Data directory for video files

### Client Options  

- `-h, --help`: Show help message
- `-c, --config <file>`: Configuration file path (default: streaming_client_demo.yaml)
- `-d, --data <directory>`: Data directory for video files

**Note:** Video source type (V4L2 vs replayer) is configured in the YAML file, not via command line arguments.

## Camera Setup and Testing

> **📖 For detailed camera configuration and troubleshooting**, see the [Client Operator README](../../operators/video_streaming/streaming_client_enhanced/README.md#camera-setup-and-testing) which includes advanced v4l2-ctl commands, YAML configuration examples, and camera-specific settings.

### Testing Your V4L2 Camera

Before starting the streaming client with camera input:

```bash
# Check available video devices
ls -la /dev/video*

# Get camera information
v4l2-ctl --device=/dev/video0 --info

# Test camera with recommended resolution
v4l2-ctl --device=/dev/video0 --set-fmt-video=width=1280,height=720,pixelformat=MJPG --stream-mmap --stream-count=10

# List supported formats
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

### Recommended Resolution Settings

**For V4L2 cameras (like Logitech C920):**

- **1280x720 @ 30fps** - Best balance of quality and performance
- **1920x1080 @ 30fps** - High quality streaming (if supported)
- **854x480 @ 30fps** - Default, good for testing and lower bandwidth

**Important:** Ensure both client and server use matching resolution settings for optimal performance.

## Video Source Modes

### V4L2 Camera Mode vs Video Replayer Mode

| Feature | V4L2 Camera | Video Replayer |
|---------|-------------|----------------|
| **Config File** | `streaming_client_demo.yaml` (default) | `streaming_client_demo_replayer.yaml` (custom) |
| **Command** | `--docker-opts='-e device=/dev/video0'` | `--run-args='-c streaming_client_demo_replayer.yaml'` |
| **Source Type** | `source: "v4l2"` | `source: "replayer"` |
| **Input Format** | `rgba8888` (4 channels) | `rgb888` (3 channels) |
| **Resolution** | 640x480 | 854x480 |
| **Data Source** | Live webcam | Pre-recorded surgical video |
| **Use Case** | Real-time streaming | Testing, demos, development |

### Switching Between Modes

To switch between V4L2 camera and video replayer:

1. **Stop the current client** (Ctrl+C)
2. **Use the appropriate command:**
   - For camera: `./holohub run video_streaming client_v4l2` (or `client_python_v4l2`)
   - For video replay: `./holohub run video_streaming client_replayer` (or `client_python`)

**Important:** The server doesn't need to be restarted when switching client modes.

## Troubleshooting

### Camera Issues

- **Camera not detected:**

  ```bash
  sudo usermod -a -G video $USER
  # Log out and back in, then test again
  ```

- **Permission denied:**

  ```bash
  sudo chmod 666 /dev/video0
  ```

### Performance Issues

- **Poor streaming quality:**
  - Try lower resolution (854x480 or 640x480)
  - Reduce frame rate to 15 or 24 FPS
  - Ensure client and server resolutions match

### Connection Issues

- **Server not starting:**

  ```bash
  # Check if port is already in use
  netstat -tlnp | grep 48010
  
  # Kill existing process if needed
  sudo lsof -ti:48010 | xargs sudo kill -9
  ```

- **Client connection timeout:**
  - Verify server is running first
  - Check firewall settings for port 48010
  - Ensure server_ip and port match in both configurations

### Video Replayer Issues

- **Config file not found:**

  ```bash
  # Ensure the replayer config exists in build directory
  cp applications/streaming_client_demo/cpp/streaming_client_demo_replayer.yaml build/streaming_client_demo/
  ```

- **Format converter errors:**
  - `Invalid channel count for RGBA8888 3 != 4`: Video replayer outputs RGB888 (3 channels), not RGBA8888 (4 channels)
  - Solution: Use `streaming_client_demo_replayer.yaml` which has correct format converter settings

- **Resolution mismatch:**
  - Video file is 854x480, ensure all components use matching resolution
  - Check `video_streaming_client`, `holoviz`, and `format_converter` settings

### Expected Behavior and Logs

**Client Application:**
The streaming client may show `GXF_EXCEEDING_PREALLOCATED_SIZE` errors during BGR→BGRA conversion. This is expected behavior as the operators handle dynamic buffer allocation internally.

**Server Application:**
The server should display connection status and frame processing information. Look for messages about client connections and frame throughput.

**Successful Video Replayer Logs:**

```console
[info] Source set to: replayer
[info] Using video replayer as source
[info] Connection established successfully
[info] Tensor validation passed: 480x854x3, 1229760 bytes
[info] Frame sent successfully
```

## Integration Testing

The video streaming demo includes comprehensive integration testing for both C++ and Python implementations.

### Quick Start

```bash
./holohub test video_streaming
```

You can use `--verbose` flag to get more detailed output.

### Run specific tests

```bash
# Run C++ integration test
./applications/video_streaming/integration_test.sh

# Run Python integration test
./applications/video_streaming/integration_test_python.sh
```

Or use direct HoloHub CLI commands:

```bash
# Run C++ integration test using HoloHub CLI
./holohub test video_streaming \
  --ctest-options="-R video_streaming_integration_test"

# Run Python integration test using HoloHub CLI
./holohub test video_streaming \
  --ctest-options="-R video_streaming_integration_test_python"
```

**Test Scripts:**

- `integration_test.sh` - C++ server and client test (SDK 3.5.0)
- `integration_test_python.sh` - Python server and client test (SDK 3.6.0)

**⚠️ Important:** Both scripts run in Docker and build from **committed source code**. Commit your changes before running tests.

**For complete testing documentation**, including expected outputs, verification criteria, and troubleshooting, see **[TESTING.md](TESTING.md)**.

## Operator Documentation

For detailed information about the underlying video streaming operators used in this application, see:

📋 **[Video Streaming Operators](../../operators/video_streaming/README.md)** - Complete operator documentation

The operator documentation includes:

- **Client Components**: VideoStreamingClientOp, FrameSaverOp
- **Server Components**: StreamingServerResource, StreamingServerUpstreamOp, StreamingServerDownstreamOp
- **Parameters and Configuration**: Detailed parameter descriptions and usage examples
- **Testing Documentation**: Comprehensive test suite with 40+ tests passing
- **API Reference**: Complete API documentation for all components

## Performance Notes

- **GPU Memory**: Configure appropriate allocator block sizes for your resolution
- **Network Bandwidth**: Monitor bandwidth usage for remote streaming scenarios  
- **Frame Rate**: Higher frame rates require more GPU/CPU resources
- **Resolution**: Balance between quality and performance based on your hardware
