# Video Streaming Demo Enhanced

This unified application demonstrates how to use the Holoscan SDK to create both streaming client and server applications for bidirectional video communication. The demo includes both client and server components that can work together to showcase real-time video streaming capabilities.

## Overview

The video streaming demo provides:
- **Streaming Client**: Captures video from V4L2 cameras or video files and streams to a server
- **Streaming Server**: Receives video streams from clients and can echo them back (passthrough mode)
- **Bidirectional Communication**: Both sending and receiving video frames
- **Multiple Source Support**: V4L2 cameras, video replay files

## Requirements

- NVIDIA GPU
- CUDA 12.1 or higher  
- Holoscan SDK 3.5.0 or higher
- V4L2 camera (optional, for live streaming)

## Dependencies

### Client Dependencies

Download the client streaming binaries from NGC:

```bash
# Navigate to the client operator directory
cd <your_holohub_path>/operators/video_streaming/streaming_client_enhanced

# Download using NGC CLI
ngc registry resource download-version "nvidia/holoscan_client_cloud_streaming:0.2"
unzip -o holoscan_client_cloud_streaming_v0.2/holoscan_client_cloud_streaming.zip -d holoscan_client_cloud_streaming

# Clean up
rm -rf holoscan_client_cloud_streaming_v0.2
```

### Server Dependencies

Download the server streaming binaries from NGC:

```bash
# Navigate to the server operator directory  
cd <your_holohub_path>/operators/video_streaming/streaming_server_enhanced

# Download using NGC CLI
ngc registry resource download-version "nvidia/holoscan_server_cloud_streaming:0.2"
unzip -o holoscan_server_cloud_streaming_v0.2/holoscan_server_cloud_streaming.zip -d holoscan_server_cloud_streaming

# Clean up
rm -rf holoscan_server_cloud_streaming_v0.2
```

## Running the Applications

The unified application provides both client and server as separate components:

**⚠️ Important: Enhanced applications require Holoscan SDK 3.5.0. Use the base image parameter to ensure compatibility.**

### 1. Start the Streaming Server

```bash
# From holohub root directory - with Holoscan 3.5.0 base image
./holohub run --docker-opts='-e EnableHybridMode=1' --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu video_streaming_demo_server --language cpp
```

### 2. Start the Streaming Client (in another terminal)

**Option A: V4L2 Camera (Webcam)**
```bash
# From holohub root directory - captures live video from webcam
./holohub run --docker-opts='-e EnableHybridMode=1 -e device=/dev/video0' --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu video_streaming_demo_client --language cpp
```

**Option B: Video File Replay**
```bash
# From holohub root directory - replays pre-recorded video file
./holohub run --docker-opts='-e EnableHybridMode=1' --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu video_streaming_demo_client --language cpp --run-args='-c streaming_client_demo_replayer.yaml'
```

**Key Differences:**
- **V4L2 Mode**: Uses `streaming_client_demo.yaml` (default), captures from webcam, 640x480 resolution
- **Replayer Mode**: Uses `streaming_client_demo_replayer.yaml` (custom), plays video file, 854x480 resolution

### Backward Compatibility

The original separate applications are still available for backward compatibility:

```bash
# Original applications (still work) - with Holoscan 3.5.0 base image
./holohub run --docker-opts='-e EnableHybridMode=1' --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu streaming_server_demo_enhanced --language cpp
./holohub run --docker-opts='-e EnableHybridMode=1 -e device=/dev/video0' --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu streaming_client_demo_enhanced --language cpp
```

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

## Configuration

### Server Configuration (streaming_server_demo.yaml)

```yaml
# Streaming server settings
streaming_server:
  # Video/stream parameters
  width: 1280          # Frame width in pixels
  height: 720          # Frame height in pixels  
  fps: 30              # Frame rate
  
  # Server connection settings
  server_ip: "127.0.0.1"
  port: 48010          # Streaming port
  multi_instance: false
  server_name: "VideoStreamingServer"
  
  # Operation mode
  receive_frames: true
  send_frames: true
  visualize_frames: false
```

### Client Configuration

#### V4L2 Camera Configuration (streaming_client_demo.yaml)

```yaml
# Source configuration - V4L2 camera mode
source: "v4l2"

# Streaming client settings
streaming_client:
  width: 640           # V4L2 camera resolution
  height: 480
  fps: 30
  server_ip: "127.0.0.1"
  signaling_port: 48010
  send_frames: true
  receive_frames: true
  visualize_frames: true

# V4L2 camera configuration
v4l2_source:
  device: "/dev/video0"
  width: 640
  height: 480
  frame_rate: 30
  pixel_format: "YUYV"

# Format converter - V4L2 outputs RGBA8888 (4 channels)
format_converter:
  in_dtype: "rgba8888"         # V4L2 always outputs RGBA8888
  out_dtype: "rgb888"          # Convert to RGB888 (3 channels)
  out_tensor_name: tensor
  scale_min: 0.0
  scale_max: 255.0
  out_channel_order: [2, 1, 0] # Convert RGB to BGR
```

#### Video Replayer Configuration (streaming_client_demo_replayer.yaml)

```yaml
# Source configuration - Video file replay mode
source: "replayer"

# Streaming client settings
streaming_client:
  width: 854           # Video file resolution (matches surgical_video.gxf)
  height: 480
  fps: 30
  server_ip: "127.0.0.1"
  signaling_port: 48010
  send_frames: true
  receive_frames: true
  visualize_frames: true

# Video replayer configuration
replayer:
  directory: "/workspace/holohub/data/endoscopy"
  basename: "surgical_video"
  frame_rate: 30
  repeat: true
  realtime: true
  count: 0

# Format converter - Video replayer outputs RGB888 (3 channels)
format_converter:
  in_dtype: "rgb888"           # Video replayer outputs RGB888
  out_dtype: "rgb888"          # Keep as RGB888 (3 channels)
  out_tensor_name: tensor
  scale_min: 0.0
  scale_max: 255.0
  out_channel_order: [2, 1, 0] # Convert RGB to BGR
```

## Camera Setup and Testing

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

## Workflow Examples

### Example 1: Local Camera to Server Echo (Unified Application)

1. **Build Application:**
   ```bash
   ./holohub build video_streaming_demo_enhanced --language cpp
   ```

2. **Start Server (Terminal 1):**
   ```bash
   ./holohub run --docker-opts='-e EnableHybridMode=1' video_streaming_demo_enhanced --language cpp
   ```

3. **Start Client with Camera (Terminal 2):**
   ```bash
   cd build/video_streaming_demo_enhanced/applications/video_streaming_demo_enhanced/cpp
   docker run --net host --runtime nvidia --gpus all -v /dev:/dev -e device=/dev/video0 -v $(pwd):/workspace/app -w /workspace/app holohub:video_streaming_demo_enhanced ./streaming_client_demo_enhanced
   ```

4. **Expected Behavior:**
   - Client captures video from `/dev/video0`
   - Client sends frames to server
   - Server receives frames and echoes them back
   - Client displays received frames (if `visualize_frames: true`)

### Example 2: Video File Replay (Unified Application)

1. **Build Application:**
   ```bash
   ./holohub build video_streaming_demo_enhanced --language cpp
   ```

2. **Start Server (Terminal 1):**
   ```bash
   ./holohub run --docker-opts='-e EnableHybridMode=1' video_streaming_demo_enhanced --language cpp
   ```

3. **Start Client with Video File (Terminal 2):**
   ```bash
   cd build/video_streaming_demo_enhanced/applications/video_streaming_demo_enhanced/cpp
   docker run --net host --runtime nvidia --gpus all -v $(pwd):/workspace/app -v /home/cdinea/Downloads/enhancedapp_holohub/holohub/data:/workspace/holohub/data -w /workspace/app holohub:video_streaming_demo_enhanced ./streaming_client_demo_enhanced -c streaming_client_demo_replayer.yaml
   ```

4. **Expected Behavior:**
   - Client replays video from `/workspace/holohub/data/endoscopy/surgical_video.gxf`
   - Client sends frames to server (854x480 resolution)
   - Server receives frames and echoes them back
   - Client displays received frames (if `visualize_frames: true`)


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
   - For camera: `./holohub run --docker-opts='-e EnableHybridMode=1 -e device=/dev/video0' streaming_client_demo_enhanced --language cpp`
   - For video replay: `./holohub run --docker-opts='-e EnableHybridMode=1' streaming_client_demo_enhanced --language cpp --run-args='-c streaming_client_demo_replayer.yaml'`

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
  cp applications/streaming_client_demo_enhanced/cpp/streaming_client_demo_replayer.yaml build/streaming_client_demo_enhanced/
  ```

- **Format converter errors:**
  - `Invalid channel count for RGBA8888 3 != 4`: Video replayer outputs RGB888 (3 channels), not RGBA8888 (4 channels)
  - Solution: Use `streaming_client_demo_replayer.yaml` which has correct format converter settings

- **Resolution mismatch:**
  - Video file is 854x480, ensure all components use matching resolution
  - Check `streaming_client`, `holoviz`, and `format_converter` settings

### Expected Behavior and Logs

**Client Application:** 
The streaming client may show `GXF_EXCEEDING_PREALLOCATED_SIZE` errors during BGR→BGRA conversion. This is expected behavior as the operators handle dynamic buffer allocation internally.

**Server Application:**
The server should display connection status and frame processing information. Look for messages about client connections and frame throughput.

**Successful Video Replayer Logs:**
```
[info] Source set to: replayer
[info] Using video replayer as source
[info] Connection established successfully
[info] Tensor validation passed: 480x854x3, 1229760 bytes
[info] Frame sent successfully
```

## Directory Structure

```
video_streaming_demo_enhanced/
├── client/                          # Client application components
│   ├── cpp/                        # C++ client implementation
│   │   ├── streaming_client_demo_enhanced    # Built client executable
│   │   ├── streaming_client_demo.cpp         # Client source code
│   │   ├── streaming_client_demo.yaml        # Client V4L2 config
│   │   ├── streaming_client_demo_replayer.yaml # Client replayer config
│   │   └── CMakeLists.txt                     # Client build configuration
│   └── setup_streaming_client.sh   # Client setup script
├── server/                          # Server application components
│   ├── cpp/                        # C++ server implementation
│   │   ├── streaming_server_demo_enhanced    # Built server executable
│   │   ├── streaming_server_demo.cpp         # Server source code
│   │   ├── streaming_server_demo.yaml        # Server config
│   │   └── CMakeLists.txt                     # Server build configuration
│   └── setup_streaming_server.sh   # Server setup script
├── CMakeLists.txt                   # Main build configuration with add_holohub_application
├── metadata.json                    # Application metadata
├── Dockerfile                       # Container configuration
└── README.md                        # This file
```

**Key Features:**
- **Unified build**: Single `./holohub build` command builds both client and server
- **Clean structure**: Maintains separate client and server directories
- **HoloHub integration**: Uses `add_holohub_application` for proper CLI integration
- **Backward compatibility**: Works with existing separate application commands

## Integration Testing

An integration test is provided that automatically tests both server and client components together:

### Running the Integration Test

```bash
# Run the integration test script directly
./applications/video_streaming_demo_enhanced/integration_test.sh

# Or run via CMake/CTest (after building)
cd build
ctest -R video_streaming_integration_test -V
```

### Integration Test Design

The integration test follows this pattern:

1. **Launch Server**: Starts the streaming server in background mode with hybrid mode enabled
2. **Wait**: Allows 10 seconds for server initialization
3. **Launch Client**: Starts the streaming client with replayer configuration in background mode
4. **Stream**: Allows 30 seconds for video streaming to occur
5. **Cleanup**: Terminates both processes gracefully
6. **Verify**: Checks log files for success indicators:
   - Server: "StreamingServerResource started successfully" or similar
   - Client: "StreamingClient created successfully" or similar
7. **Report**: Outputs detailed logs and final PASS/FAIL result

### Log Files

The integration test generates:
- `streamingserver.log`: Server application logs
- `streamingclient.log`: Client application logs

These files are automatically displayed during test execution for debugging purposes.

## Operator Documentation

For detailed information about the underlying video streaming operators used in this application, see:

📋 **[Video Streaming Operators](../../operators/video_streaming/README.md)** - Complete operator documentation

The operator documentation includes:
- **Client Components**: StreamingClientOp, FrameSaverOp
- **Server Components**: StreamingServerResource, StreamingServerUpstreamOp, StreamingServerDownstreamOp
- **Parameters and Configuration**: Detailed parameter descriptions and usage examples
- **Testing Documentation**: Comprehensive test suite with 40+ tests passing
- **API Reference**: Complete API documentation for all components

## Performance Notes

- **GPU Memory**: Configure appropriate allocator block sizes for your resolution
- **Network Bandwidth**: Monitor bandwidth usage for remote streaming scenarios  
- **Frame Rate**: Higher frame rates require more GPU/CPU resources
- **Resolution**: Balance between quality and performance based on your hardware

## License

Apache-2.0 - See the LICENSE file for details.