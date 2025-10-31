# Video Streaming Client Demo

This application demonstrates how to create a bidirectional video streaming client that sends video frames to a server and receives frames back. Both C++ and Python implementations are available with support for V4L2 cameras and video file replay.

> **📚 Related Documentation:**
>
> - **[Main README](../README.md)** - Application overview, quick start, and common configuration
> - **[Server README](../video_streaming_server/README.md)** - Server setup and configuration
> - **[Testing Documentation](../TESTING.md)** - Integration testing and verification

## Features

- **Multiple Video Sources**:
  - V4L2 Camera (webcam) support with configurable resolution
  - Video file replay for testing and demos
- **Real-time Visualization**: Holoviz integration for displaying received frames
- **Configurable**: YAML configuration file support and command-line options
- **C++ and Python**: Full implementations in both languages with compatible APIs
- **Format Conversion**: Automatic format conversion for different video sources

## Requirements

- Holoscan SDK 3.5.0
- Custom Dockerfile with OpenSSL 3.4.0 (for running via holohub CLI)
- For Python: Python 3.8+ and bindings built with `-DHOLOHUB_BUILD_PYTHON=ON`
- CUDA 12.x (currently not working with CUDA 13.x)
- OpenCV
- `video_streaming_client` operator
- V4L2 camera (optional, for live streaming)

### Download Client Cloud Streaming

Download the Holoscan Client Cloud Streaming binaries from NGC:

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

## Usage

> ⚠️ This application requires and uses the Holoscan SDK 3.5.0.
> ℹ️ The client requires OpenSSL 3.4.0, which is installed inside the custom Dockerfile.

### C++ Client

**Video Replayer Mode (Default - 854x480):**

```bash
# From holohub root directory - runs with video file playback
./holohub run video_streaming_client replayer --language cpp
```

**V4L2 Camera Mode (640x480):**

```bash
# From holohub root directory - runs with V4L2 camera (webcam)
./holohub run video_streaming_client v4l2 --language cpp
```

### Python Client

**Video Replayer Mode (Default - 854x480):**

```bash
# From holohub root directory - runs with video file playback
./holohub run video_streaming_client replayer --language python
```

**V4L2 Camera Mode (640x480):**

```bash
# From holohub root directory - runs with V4L2 camera (webcam)
./holohub run video_streaming_client v4l2 --language python
```

**Default Client Configurations:**

**Video Replayer Mode:**

- Source: Video file (surgical_video)
- Resolution: 854x480
- Frame Rate: 30 fps
- Server: 127.0.0.1:48010

**V4L2 Camera Mode:**

- Source: /dev/video0 (webcam)
- Resolution: 640x480
- Frame Rate: 30 fps
- Server: 127.0.0.1:48010

**Important:** Ensure the server is configured to match the client's resolution for optimal performance.

### Command Line Options

**Python Client**:

- `--source {replayer,v4l2}`: Video source type (default: replayer)
- `--server-ip IP`: Server IP address (default: 127.0.0.1)
- `--port PORT`: Server port (default: 48010)
- `--width WIDTH`: Frame width (default: 854 for replayer, 640 for v4l2)
- `--height HEIGHT`: Frame height (default: 480)
- `--fps FPS`: Frames per second (default: 30)
- `--no-viz`: Disable visualization (Holoviz)
- `--config PATH` or `-c PATH`: Path to YAML configuration file
- `--create-config PATH`: Create default configuration file at specified path
- `--help`: Show help message

**C++ Client**:

- `-c PATH` or `--config PATH`: Path to YAML configuration file
- `-d PATH` or `--data PATH`: Data directory path (for video files)
- `-h` or `--help`: Show help message

## Configuration

### C++ Configuration

The C++ application is configured via YAML file. Configuration varies based on video source:

**Video Replayer Configuration** (`cpp/video_streaming_client_demo_replayer.yaml`):

```yaml
%YAML 1.2
---
application:
  title: Streaming Client Test App
  version: 1.0
  log_level: INFO

# Source configuration
source: "replayer"

# Video replayer configuration
replayer:
  directory: "/workspace/holohub/data/endoscopy"
  basename: "surgical_video"
  frame_rate: 30
  repeat: true
  realtime: true
  count: 0

# Format converter - replayer outputs RGB888 (3 channels)
format_converter:
  in_dtype: "rgb888"
  out_dtype: "rgb888"
  out_tensor_name: tensor
  scale_min: 0.0
  scale_max: 255.0
  out_channel_order: [2, 1, 0]  # Convert RGB to BGR

# Streaming client settings
video_streaming_client:
  width: 854
  height: 480
  fps: 30
  server_ip: "127.0.0.1"
  signaling_port: 48010
  send_frames: true
  receive_frames: true
  min_non_zero_bytes: 10

# Visualization
visualize_frames: true

holoviz:
  width: 854
  height: 480
  tensors:
    - name: "bgra_tensor"
      type: color
      image_format: "b8g8r8a8_unorm"
      opacity: 1.0
      priority: 0

# Buffer pool configuration
allocator:
  block_size: 4194304
  num_blocks: 12

scheduler: "greedy"
```

**V4L2 Camera Configuration** (`cpp/video_streaming_client_demo.yaml`):

```yaml
source: "v4l2"

# V4L2 camera configuration
v4l2_source:
  device: "/dev/video0"
  width: 640
  height: 480
  frame_rate: 30
  pixel_format: "YUYV"

# Format converter - V4L2 outputs RGBA8888 (4 channels)
format_converter:
  in_dtype: "rgba8888"
  out_dtype: "rgb888"
  out_tensor_name: tensor
  scale_min: 0.0
  scale_max: 255.0
  out_channel_order: [2, 1, 0]

# Streaming client settings
video_streaming_client:
  width: 640
  height: 480
  fps: 30
  server_ip: "127.0.0.1"
  signaling_port: 48010
  send_frames: true
  receive_frames: true
  min_non_zero_bytes: 10
```

**Key Configuration Notes**:

- **Format Converter**: V4L2 outputs `rgba8888` (4 channels), video replayer outputs `rgb888` (3 channels)
- **Channel Order**: `out_channel_order: [2, 1, 0]` converts RGB to BGR
- **Resolution**: V4L2 default is 640x480, video replayer is 854x480
- **Allocator**: Buffer pool sized for BGRA frames (4MB blocks)

### Python Configuration

The Python application is primarily configured via **command-line arguments**:

**Command-Line Parameters** (recommended):

```bash
# Video replayer mode (854x480)
python3 video_streaming_client_demo.py --source replayer --width 854 --height 480

# V4L2 camera mode (640x480)
python3 video_streaming_client_demo.py --source v4l2 --width 640 --height 480

# Custom server
python3 video_streaming_client_demo.py --server-ip 192.168.1.100 --port 48010
```

**Python YAML Structure** (optional, auto-selected based on source):

```yaml
application:
  title: "Streaming Client Python Demo"
  version: "1.0"
  log_level: "INFO"

# Source configuration
source: "replayer"  # or "v4l2"

# Streaming client settings
video_streaming_client:
  width: 854
  height: 480
  fps: 30
  server_ip: "127.0.0.1"
  signaling_port: 48010
  send_frames: true
  receive_frames: true
  min_non_zero_bytes: 10

# Video replayer settings
replayer:
  directory: "/workspace/holohub/data/endoscopy"
  basename: "surgical_video"
  frame_rate: 30
  repeat: true
  realtime: true

# V4L2 camera settings
v4l2:
  device: "/dev/video0"
  width: 640
  height: 480
  frame_rate: 30
  pixel_format: "YUYV"

# Visualization settings
visualization:
  enabled: true
  width: 854
  height: 480
```

**Configuration Files**:

- C++ V4L2: `cpp/video_streaming_client_demo.yaml`
- C++ Replayer: `cpp/video_streaming_client_demo_replayer.yaml`
- Python V4L2: `python/video_streaming_client_demo.yaml`
- Python Replayer: `python/video_streaming_client_demo_replayer.yaml`

**Note**: Python parameters set via command-line take precedence over YAML configuration. The Python app auto-selects the appropriate config file based on the `--source` argument.

## Pipeline Architecture

The client implements a bidirectional streaming pipeline with format conversion:

**Video Replayer Pipeline:**

```text
VideoStreamReplayerOp → FormatConverterOp → VideoStreamingClientOp → HoloVizOp
                                                    ↓
                                            (sends to server)
                                                    ↓
                                            (receives from server)
                                                    ↓
                                            output_frames → HoloVizOp
```

**V4L2 Camera Pipeline:**

```text
V4L2VideoCaptureOp → FormatConverterOp → VideoStreamingClientOp → HoloVizOp
                                                 ↓
                                         (sends to server)
                                                 ↓
                                         (receives from server)
                                                 ↓
                                         output_frames → HoloVizOp
```

### How It Works

1. **Video Source**: Either `VideoStreamReplayerOp` (file) or `V4L2VideoCaptureOp` (camera) provides video frames
2. **Format Converter**: Converts video format to RGB888 for streaming
3. **VideoStreamingClientOp**: Sends frames to server and receives processed frames back
4. **HoloVizOp**: Displays the received frames from the server

## C++ Implementation

The C++ implementation (`cpp/video_streaming_client_demo.cpp`) demonstrates usage of the streaming client operator:

**Video Replayer Mode:**

```cpp
#include "video_streaming_client.hpp"
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>

// Create video source (replayer)
auto replayer = make_operator<ops::VideoStreamReplayerOp>(
    "replayer",
    Arg("directory", std::string("/workspace/holohub/data/endoscopy")),
    Arg("basename", std::string("surgical_video")),
    Arg("frame_rate", 30.0f)
);

// Create format converter (RGB to BGR)
auto format_converter = make_operator<ops::FormatConverterOp>(
    "format_converter",
    Arg("in_dtype", std::string("rgb888")),
    Arg("out_dtype", std::string("rgb888")),
    Arg("out_tensor_name", std::string("tensor")),
    Arg("out_channel_order", std::vector<int>{2, 1, 0})  // RGB to BGR
);

// Create streaming client
auto video_streaming_client = make_operator<ops::VideoStreamingClientOp>(
    "video_streaming_client",
    Arg("server_ip", std::string("127.0.0.1")),
    Arg("signaling_port", uint16_t{48010}),
    Arg("width", 854U),
    Arg("height", 480U),
    Arg("fps", uint16_t{30}),
    Arg("send_frames", true),
    Arg("receive_frames", true),
    Arg("min_non_zero_bytes", static_cast<uint32_t>(10))
);

// Create visualization
auto holoviz = make_operator<ops::HolovizOp>(
    "holoviz",
    Arg("width", 854U),
    Arg("height", 480U)
);

// Connect the pipeline
add_flow(replayer, format_converter, {{"output", "source_video"}});
add_flow(format_converter, video_streaming_client);
add_flow(video_streaming_client, holoviz, {{"output_frames", "receivers"}});
```

**V4L2 Camera Mode:**

```cpp
#include "video_streaming_client.hpp"
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/v4l2_video_capture/v4l2_video_capture.hpp>

// Create video source (V4L2 camera)
auto v4l2_source = make_operator<ops::V4L2VideoCaptureOp>(
    "v4l2_camera",
    Arg("device", std::string("/dev/video0")),
    Arg("width", 640U),
    Arg("height", 480U),
    Arg("num_buffers", 4U),
    Arg("pixel_format", std::string("YUYV"))
);

// Create format converter (RGBA to RGB/BGR)
auto format_converter = make_operator<ops::FormatConverterOp>(
    "format_converter",
    Arg("in_dtype", std::string("rgba8888")),  // V4L2 outputs RGBA
    Arg("out_dtype", std::string("rgb888")),   // Convert to RGB
    Arg("out_tensor_name", std::string("tensor")),
    Arg("out_channel_order", std::vector<int>{2, 1, 0})  // RGB to BGR
);

// Create streaming client
auto video_streaming_client = make_operator<ops::VideoStreamingClientOp>(
    "video_streaming_client",
    Arg("server_ip", std::string("127.0.0.1")),
    Arg("signaling_port", uint16_t{48010}),
    Arg("width", 640U),
    Arg("height", 480U),
    Arg("fps", uint16_t{30}),
    Arg("send_frames", true),
    Arg("receive_frames", true),
    Arg("min_non_zero_bytes", static_cast<uint32_t>(10))
);

// Create visualization
auto holoviz = make_operator<ops::HolovizOp>(
    "holoviz",
    Arg("width", 640U),
    Arg("height", 480U)
);

// Connect the pipeline
add_flow(v4l2_source, format_converter, {{"signal", "source_video"}});
add_flow(format_converter, video_streaming_client);
add_flow(video_streaming_client, holoviz, {{"output_frames", "receivers"}});
```

## Python Implementation

The Python implementation (`python/video_streaming_client_demo.py`) demonstrates usage of the Python bindings:

**Video Replayer Mode:**

```python
from holohub.video_streaming_client import VideoStreamingClientOp
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import UnboundedAllocator

class StreamingClientApp(Application):
    def compose(self):
        # Create allocator
        allocator = UnboundedAllocator(self, name="allocator")

        # Create video source (replayer)
        replayer = VideoStreamReplayerOp(
            self,
            name="replayer",
            directory="/workspace/holohub/data/endoscopy",
            basename="surgical_video",
            frame_rate=30,
        )

        # Create format converter (RGB to BGR)
        format_converter = FormatConverterOp(
            self,
            name="format_converter",
            in_dtype="rgb888",
            out_dtype="rgb888",
            out_tensor_name="tensor",
            scale_min=0.0,
            scale_max=255.0,
            out_channel_order=[2, 1, 0],  # Convert RGB to BGR
        )

        # Create streaming client
        video_streaming_client = VideoStreamingClientOp(
            self,
            allocator,  # Allocator for output buffer
            name="video_streaming_client",
            server_ip="127.0.0.1",
            signaling_port=48010,
            width=854,
            height=480,
            fps=30,
            send_frames=True,
            receive_frames=True,
            min_non_zero_bytes=10,
        )

        # Create visualization (optional)
        holoviz = HolovizOp(
            self,
            name="holoviz",
            width=854,
            height=480,
        )

        # Connect the pipeline
        self.add_flow(replayer, format_converter, {("output", "source_video")})
        self.add_flow(format_converter, video_streaming_client)
        self.add_flow(video_streaming_client, holoviz, {("output_frames", "receivers")})
```

**V4L2 Camera Mode:**

```python
from holohub.video_streaming_client import VideoStreamingClientOp
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
    V4L2VideoCaptureOp,
)
from holoscan.resources import UnboundedAllocator

class StreamingClientApp(Application):
    def compose(self):
        # Create allocator
        allocator = UnboundedAllocator(self, name="allocator")

        # Create video source (V4L2 camera)
        v4l2_source = V4L2VideoCaptureOp(
            self,
            name="v4l2_camera",
            device="/dev/video0",
            width=640,
            height=480,
            frame_rate=30,
            pixel_format="YUYV",
            allocator=allocator,
        )

        # Create format converter (RGBA to RGB/BGR)
        format_converter = FormatConverterOp(
            self,
            name="format_converter",
            in_dtype="rgba8888",  # V4L2 outputs RGBA
            out_dtype="rgb888",   # Convert to RGB
            out_tensor_name="tensor",
            scale_min=0.0,
            scale_max=255.0,
            out_channel_order=[2, 1, 0],  # Convert RGB to BGR
            pool=allocator,
        )

        # Create streaming client
        video_streaming_client = VideoStreamingClientOp(
            self,
            allocator,  # Allocator for output buffer
            name="video_streaming_client",
            server_ip="127.0.0.1",
            signaling_port=48010,
            width=640,
            height=480,
            fps=30,
            send_frames=True,
            receive_frames=True,
            min_non_zero_bytes=10,
        )

        # Create visualization (optional)
        holoviz = HolovizOp(
            self,
            name="holoviz",
            width=640,
            height=480,
        )

        # Connect the pipeline
        self.add_flow(v4l2_source, format_converter, {("signal", "source_video")})
        self.add_flow(format_converter, video_streaming_client)
        self.add_flow(video_streaming_client, holoviz, {("output_frames", "receivers")})
```

**Key Points:**

- The `VideoStreamingClientOp` requires an allocator (passed as a positional argument) for output buffer allocation
- The `VideoStreamingClientOp` handles bidirectional streaming (sends and receives frames)
- **Parameters are set via constructor arguments** (from command-line or defaults), not from YAML
- The constructor parameters (`source`, `server_ip`, `port`, `width`, `height`, `fps`) configure the application
- Format conversion is necessary to convert source formats to RGB for streaming
- V4L2 always outputs RGBA8888 (4 channels) regardless of input format and uses "signal" output port
- Video replayer outputs RGB888 (3 channels) and uses "output" output port
- The `min_non_zero_bytes` parameter prevents sending empty frames during startup
- The `output_frames` port receives processed frames from the server
- Holoviz displays the received frames using the `receivers` input port
- The Python app auto-selects the appropriate config file based on `--source` argument if no config is specified

## Troubleshooting

### Common Issues

1. **Connection Failed**: Verify server is running and ports are correct

   ```bash
   # Check if server is listening
   netstat -tlnp | grep 48010
   ```

2. **Camera Not Found**: Check V4L2 device path

   ```bash
   # List available cameras
   ls -l /dev/video*
   
   # Get camera info
   v4l2-ctl --device=/dev/video0 --info
   
   # Test camera
   v4l2-ctl --device=/dev/video0 --set-fmt-video=width=640,height=480,pixelformat=YUYV --stream-mmap --stream-count=10
   ```

3. **Permission Denied (Camera)**: Add user to video group

   ```bash
   sudo usermod -a -G video $USER
   # Log out and back in
   
   # Or change permissions
   sudo chmod 666 /dev/video0
   ```

4. **Import Error (Python)**: Ensure Holoscan SDK Python bindings are installed

   ```bash
   # Build with Python bindings
   ./holohub build video_streaming --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'
   ```

5. **Video Files Not Found**: Check data directory path

   ```bash
   # Ensure video files exist
   ls -l /workspace/holohub/data/endoscopy/surgical_video*
   ```

6. **Resolution Mismatch**: Ensure client and server resolutions match
   - Replayer default: 854x480
   - V4L2 default: 640x480
   - Server default: 854x480

7. **Format Converter Errors**:
   - `Invalid channel count for RGBA8888 3 != 4`: Video replayer outputs RGB888 (3 channels), not RGBA8888
   - Solution: Use correct configuration file (`video_streaming_client_demo_replayer.yaml`)

### Debug Mode

Enable debug logging by setting log level in configuration:

```yaml
application:
  log_level: "DEBUG"
```

## Examples

See the included configuration files for complete examples:

- C++ V4L2: `cpp/video_streaming_client_demo.yaml`
- C++ Replayer: `cpp/video_streaming_client_demo_replayer.yaml`
- Python V4L2: `python/video_streaming_client_demo.yaml`
- Python Replayer: `python/video_streaming_client_demo_replayer.yaml`

## Integration with Server

### Testing with Python Server

**Option 1: Using Holohub CLI (Recommended)**

Terminal 1 - Start Python Server:

```bash
# From holohub root directory
./holohub run video_streaming server_python \
  --docker-file applications/video_streaming/Dockerfile \
  --docker-opts='-e EnableHybridMode=1' \
  --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'
```

Terminal 2 - Start Python Client with Video Replayer (854x480):

```bash
# From holohub root directory
./holohub run video_streaming client_python \
  --docker-file applications/video_streaming/Dockerfile \
  --docker-opts='-e EnableHybridMode=1' \
  --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'
```

Terminal 2 - Or Start Python Client with V4L2 Camera (640x480):

```bash
# From holohub root directory
./holohub run video_streaming client_python_v4l2 \
  --docker-file applications/video_streaming/Dockerfile \
  --docker-opts='-e EnableHybridMode=1' \
  --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'
```

### Testing with C++ Server

Terminal 1 - Start Server (C++ or Python):

```bash
# C++ Server
./holohub run video_streaming

# OR Python Server
./holohub run video_streaming server_python \
  --docker-file applications/video_streaming/Dockerfile \
  --docker-opts='-e EnableHybridMode=1' \
  --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'
```

Terminal 2 - Start C++ Client:

```bash
# Video Replayer Mode
./holohub run video_streaming client_replayer

# OR V4L2 Camera Mode
./holohub run video_streaming client_v4l2
```

**Important:**

- C++ and Python implementations are fully compatible - you can mix and match (C++ client with Python server, etc.)
- Ensure client and server resolutions match for optimal performance
- The server must be started before the client
- For V4L2 mode, ensure camera permissions are set correctly

## Video Source Modes

### V4L2 Camera vs Video Replayer

| Feature | V4L2 Camera | Video Replayer |
|---------|-------------|----------------|
| **Config File** | `video_streaming_client_demo.yaml` | `video_streaming_client_demo_replayer.yaml` |
| **Source Type** | `source: "v4l2"` | `source: "replayer"` |
| **Input Format** | `rgba8888` (4 channels) | `rgb888` (3 channels) |
| **Resolution** | 640x480 (configurable) | 854x480 |
| **Data Source** | Live webcam | Pre-recorded surgical video |
| **Use Case** | Real-time streaming | Testing, demos, development |

### Switching Between Modes

To switch between V4L2 camera and video replayer:

1. **Stop the current client** (Ctrl+C)
2. **Use the appropriate command:**
   - For camera: `./holohub run video_streaming client_v4l2` (or `client_python_v4l2`)
   - For video replay: `./holohub run video_streaming client_replayer` (or `client_python`)

**Important:** The server doesn't need to be restarted when switching client modes.

## Dependencies

- Holoscan SDK 3.5.0+
- video_streaming operator
- OpenCV
- CUDA 12.x
- OpenSSL 3.4.0 (installed inside the custom Dockerfile)
- V4L2 compatible camera (for camera mode)

## See Also

- [Main Video Streaming README](../README.md) - Complete application documentation with Python bindings and integration testing
- [Video Streaming Server](../video_streaming_server/README.md) - Server application documentation
- [Video Streaming Operators](../../../operators/video_streaming/README.md) - Complete operator documentation
