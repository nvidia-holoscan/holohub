# Video Streaming Client Demo

This application demonstrates how to create a bidirectional video streaming client that sends video frames to a server and receives frames back. Both C++ and Python implementations are available with support for V4L2 cameras and video file replay.

> **📚 Related Documentation:**
> - **[Main README](../README.md)** - Application overview, quick start, and common configuration
> - **[Server README](../video_streaming_server/README.md)** - Server setup and configuration
> - **[Testing Documentation](../TESTING.md)** - Integration testing and verification

## Features

- **Bidirectional Streaming**: Sends video frames to server and receives frames back
- **Multiple Video Sources**: 
  - V4L2 Camera (webcam) support with configurable resolution
  - Video file replay for testing and demos
- **Real-time Visualization**: Holoviz integration for displaying received frames
- **Configurable**: YAML configuration file support and command-line options
- **C++ and Python**: Full implementations in both languages with compatible APIs
- **Format Conversion**: Automatic format conversion for different video sources

## Requirements

- Holoscan SDK 3.5.0 or higher
- Custom Dockerfile with OpenSSL 3.4.0 (for running via holohub CLI)
- For Python: Python 3.8+ and bindings built with `-DHOLOHUB_BUILD_PYTHON=ON`
- CUDA 12.x (currently not working with CUDA 13.x)
- OpenCV
- video_streaming operator
- V4L2 camera (optional, for live streaming)

## Usage

> [!IMPORTANT] The client applications requires Holoscan SDK 3.5.0. Either set the SDK version environment variable before running the applications, or use the `--base-img` option to specify the base image.
>
> ```bash
> # Set SDK version environment variable
> export HOLOHUB_BASE_SDK_VERSION=3.5.0
> ```
>
> [!NOTE] The client requires OpenSSL 3.4.0, which is installed inside the custom Dockerfile.

### C++ Client

**Video Replayer Mode (Default - 854x480):**
```bash
# From holohub root directory - runs with video file playback
./holohub run video_streaming client_replayer
```

**V4L2 Camera Mode (640x480):**
```bash
# From holohub root directory - runs with V4L2 camera (webcam)
./holohub run video_streaming client_v4l2
```

### Python Client

**Video Replayer Mode (Default - 854x480):**
```bash
# From holohub root directory - runs with video file playback
./holohub run video_streaming client_python \
  --docker-file applications/video_streaming/Dockerfile \
  --docker-opts='-e EnableHybridMode=1' \
  --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'
```

**V4L2 Camera Mode (640x480):**
```bash
# From holohub root directory - runs with V4L2 camera (webcam)
./holohub run video_streaming client_python_v4l2 \
  --docker-file applications/video_streaming/Dockerfile \
  --docker-opts='-e EnableHybridMode=1' \
  --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'
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

### Command Line Options (Python)

- `--source {replayer,v4l2}`: Video source type (default: replayer)
- `--server-ip IP`: Server IP address (default: 127.0.0.1)
- `--port PORT`: Server port (default: 48010)
- `--width WIDTH`: Frame width (default: 854 for replayer, 640 for v4l2)
- `--height HEIGHT`: Frame height (default: 480)
- `--fps FPS`: Frames per second (default: 30)
- `--config PATH` or `-c PATH`: Path to YAML configuration file
- `--help`: Show help message

### Command Line Options (C++)

- `-c PATH` or `--config PATH`: Path to YAML configuration file

## Configuration

The application can be configured via YAML file or command-line arguments. Example configuration file structure:

```yaml
# Application metadata
application:
  title: "Streaming Client Demo"
  version: "1.0"
  log_level: "INFO"

# Source configuration
source: "replayer"  # or "v4l2"

# Client configuration
streaming_client:
  server_ip: "127.0.0.1"
  signaling_port: 48010
  width: 854
  height: 480
  fps: 30
  send_frames: true
  receive_frames: true

# Video replayer configuration (if source: "replayer")
replayer:
  directory: "/workspace/holohub/data/endoscopy"
  basename: "surgical_video"
  frame_rate: 30

# V4L2 camera configuration (if source: "v4l2")
v4l2:
  device: "/dev/video0"
  width: 640
  height: 480
  frame_rate: 30
  pixel_format: "YUYV"

# Visualization configuration
holoviz:
  width: 854
  height: 480

# Scheduler configuration
scheduler: "multi_thread"

multi_thread_scheduler:
  worker_thread_number: 2
  stop_on_deadlock: true
  stop_on_deadlock_timeout: 5000
```

Configuration files are located in:
- C++ V4L2: `cpp/streaming_client_demo.yaml`
- C++ Replayer: `cpp/streaming_client_demo_replayer.yaml`
- Python V4L2: `python/streaming_client_demo.yaml`
- Python Replayer: `python/streaming_client_demo_replayer.yaml`

## Pipeline Architecture

The client implements a bidirectional streaming pipeline with format conversion:

**Video Replayer Pipeline:**
```text
VideoStreamReplayerOp → FormatConverterOp → StreamingClientOp → HoloVizOp
                                                    ↓
                                            (sends to server)
                                                    ↓
                                            (receives from server)
                                                    ↓
                                            output_frames → HoloVizOp
```

**V4L2 Camera Pipeline:**
```text
V4L2VideoCaptureOp → FormatConverterOp → StreamingClientOp → HoloVizOp
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
3. **StreamingClientOp**: Sends frames to server and receives processed frames back
4. **HoloVizOp**: Displays the received frames from the server

## C++ Implementation

The C++ implementation (`cpp/streaming_client_demo.cpp`) demonstrates usage of the streaming client operator:

**Video Replayer Mode:**
```cpp
#include "streaming_client.hpp"
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
    Arg("out_tensor_name", std::string("tensor"))
);

// Create streaming client
auto streaming_client = make_operator<ops::StreamingClientOp>(
    "streaming_client",
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
add_flow(format_converter, streaming_client);
add_flow(streaming_client, holoviz, {{"output_frames", "receivers"}});
```

**V4L2 Camera Mode:**
```cpp
#include "streaming_client.hpp"
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
    Arg("out_tensor_name", std::string("tensor"))
);

// Create streaming client
auto streaming_client = make_operator<ops::StreamingClientOp>(
    "streaming_client",
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
add_flow(format_converter, streaming_client);
add_flow(streaming_client, holoviz, {{"output_frames", "receivers"}});
```

## Python Implementation

The Python implementation (`python/streaming_client_demo.py`) demonstrates usage of the Python bindings:

**Video Replayer Mode:**
```python
from holohub.streaming_client_enhanced import StreamingClientOp
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
        )

        # Create streaming client
        streaming_client = StreamingClientOp(
            self,
            allocator,  # Allocator for output buffer
            name="streaming_client",
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
        self.add_flow(format_converter, streaming_client)
        self.add_flow(streaming_client, holoviz, {("output_frames", "receivers")})
```

**V4L2 Camera Mode:**
```python
from holohub.streaming_client_enhanced import StreamingClientOp
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
            pool=allocator,
        )

        # Create streaming client
        streaming_client = StreamingClientOp(
            self,
            allocator,  # Allocator for output buffer
            name="streaming_client",
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
        self.add_flow(format_converter, streaming_client)
        self.add_flow(streaming_client, holoviz, {("output_frames", "receivers")})
```

**Key Points:**
- The `StreamingClientOp` requires an allocator (passed as a positional argument) for output buffer allocation
- The `StreamingClientOp` handles bidirectional streaming (sends and receives frames)
- Format conversion is necessary to convert source formats to RGB for streaming
- V4L2 always outputs RGBA8888 (4 channels) regardless of input format and uses "signal" output port
- Video replayer outputs RGB888 (3 channels) and uses "output" output port
- The `min_non_zero_bytes` parameter prevents sending empty frames during startup
- The `output_frames` port receives processed frames from the server
- Holoviz displays the received frames using the `receivers` input port

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
   - Solution: Use correct configuration file (`streaming_client_demo_replayer.yaml`)

### Debug Mode

Enable debug logging by setting log level in configuration:

```yaml
application:
  log_level: "DEBUG"
```

## Examples

See the included configuration files for complete examples:
- C++ V4L2: `cpp/streaming_client_demo.yaml`
- C++ Replayer: `cpp/streaming_client_demo_replayer.yaml`
- Python V4L2: `python/streaming_client_demo.yaml`
- Python Replayer: `python/streaming_client_demo_replayer.yaml`

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
| **Config File** | `streaming_client_demo.yaml` | `streaming_client_demo_replayer.yaml` |
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
