# Video Streaming Client Demo (Python)

This Python application demonstrates how to use the StreamingClientOp Python bindings to create a video streaming client that can send and receive video frames.

## Features

- **Video Sources**: Support for video file playback (VideoStreamReplayerOp) and V4L2 camera capture
- **Real-time Streaming**: Send video frames to a streaming server
- **Bidirectional**: Receive processed frames back from the server
- **Format Conversion**: Automatic RGB to BGR conversion for compatibility
- **Visualization**: Optional HoloViz integration for displaying received frames
- **Python Bindings**: Uses `holohub.streaming_client_enhanced` Python bindings

## Requirements

- Holoscan SDK 3.5.0 or higher with Python bindings
- Python 3.8 or higher
- Custom Dockerfile with OpenSSL 3.4.0 (for running via holohub CLI)
- Video files (for replayer mode) or V4L2 compatible camera (for camera mode)
- Python bindings must be built with `-DHOLOHUB_BUILD_PYTHON=ON`

## Usage

Run the Python streaming client using the Holohub CLI:

### Video Replayer Mode (Default - 854x480)

```bash
# From holohub root directory - runs with video file playback
./holohub run video_streaming client_python \
  --docker-file applications/video_streaming/Dockerfile \
  --docker-opts='-e EnableHybridMode=1' \
  --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'
```

### V4L2 Camera Mode (640x480)

```bash
# From holohub root directory - runs with V4L2 camera (webcam)
./holohub run video_streaming client_python_v4l2 \
  --docker-file applications/video_streaming/Dockerfile \
  --docker-opts='-e EnableHybridMode=1' \
  --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'
```

**Default Configurations:**

**Video Replayer Mode:**
- **Source**: Video file (surgical_video)
- **Resolution**: 854x480
- **Frame Rate**: 30 fps
- **Server**: 127.0.0.1:48010

**V4L2 Camera Mode:**
- **Source**: /dev/video0 (webcam)
- **Resolution**: 640x480
- **Frame Rate**: 30 fps
- **Server**: 127.0.0.1:48010

**Important:** Ensure the server is configured to match the client's resolution for optimal performance.

## Configuration

The application uses YAML configuration files for different modes. Key configuration sections:

### Video Replayer Configuration (`streaming_client_demo_replayer.yaml`)

```yaml
# Video source: replayer for video file playback
source: "replayer"

replayer:
  directory: "/workspace/holohub/data/endoscopy"
  basename: "surgical_video"
  frame_rate: 30

# Streaming client settings (854x480)
streaming_client:
  width: 854
  height: 480
  fps: 30
  server_ip: "127.0.0.1"
  port: 48010
  send_frames: true
  receive_frames: true
```

### V4L2 Camera Configuration (`streaming_client_demo.yaml`)

```yaml
# Video source: v4l2 for webcam capture
source: "v4l2"

v4l2:
  device: "/dev/video0"
  width: 640
  height: 480
  frame_rate: 30
  pixel_format: "YUYV"

# Streaming client settings (640x480)
streaming_client:
  width: 640
  height: 480
  fps: 30
  server_ip: "127.0.0.1"
  port: 48010
  send_frames: true
  receive_frames: true
```

## Pipeline Architecture

The Python client implements the following processing pipeline:

```
Video Source → FormatConverterOp → StreamingClientOp → HoloVizOp (optional)
```

### Pipeline Components:

1. **Video Source**: Either `VideoStreamReplayerOp` (for video files) or `V4L2VideoCaptureOp` (for webcam)
2. **FormatConverterOp**: Converts RGBA/RGB to BGR format for compatibility
3. **StreamingClientOp**: Sends frames to server and receives processed frames back
4. **HoloVizOp**: Visualizes the received frames from the server (optional)

### How It Works:

1. Video frames are captured from the source (file or camera)
2. Frames are converted to the correct format (BGR)
3. `StreamingClientOp` sends frames to the server and receives processed frames back
4. Received frames are optionally visualized using HoloViz

## Python Bindings Usage

The application demonstrates usage of the `holohub.streaming_client_enhanced` Python bindings:

```python
from holohub.streaming_client_enhanced import StreamingClientOp
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
    V4L2VideoCaptureOp,
    VideoStreamReplayerOp,
)

# Create video source (V4L2 example)
source_op = V4L2VideoCaptureOp(
    self,
    name="v4l2_camera",
    device="/dev/video0",
    width=640,
    height=480,
    frame_rate=30,
    pixel_format="YUYV",
    allocator=allocator,
)

# Create format converter
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
    name="streaming_client",
    server_ip="127.0.0.1",
    port=48010,
    width=640,
    height=480,
    fps=30,
    send_frames=True,
    receive_frames=True,
)

# Create visualization (optional)
holoviz = HolovizOp(
    self,
    name="holoviz",
    width=640,
    height=480,
)

# Connect the pipeline
self.add_flow(source_op, format_converter, {("output", "source_video")})
self.add_flow(format_converter, streaming_client)
self.add_flow(streaming_client, holoviz, {("output_frames", "receivers")})
```

**Key Points:**
- The `StreamingClientOp` handles bidirectional streaming (sends and receives frames)
- Format conversion is necessary to convert source formats to BGR for streaming
- The `output_frames` port receives processed frames from the server
- HoloViz displays the received frames using the `receivers` input port

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure Holoscan SDK Python bindings are installed
2. **Camera Not Found**: Check V4L2 device path (`/dev/video0`)
3. **Connection Failed**: Verify server is running and ports are correct
4. **Video Files Not Found**: Check data directory path

### Debug Mode

Enable debug logging by setting log level in configuration:

```yaml
application:
  log_level: "DEBUG"
```

## Examples

See the included configuration files for complete examples:
- `streaming_client_demo_replayer.yaml` - Video file playback configuration (854x480)
- `streaming_client_demo.yaml` - V4L2 camera configuration (640x480)

## Integration with Server

### Testing with Python Server

Terminal 1 - Start Python Server:
```bash
# From holohub root directory
./holohub run video_streaming server_python \
  --docker-file applications/video_streaming/Dockerfile \
  --docker-opts='-e EnableHybridMode=1' \
  --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'
```

Terminal 2 - Start Python Client (Video Replayer):
```bash
# From holohub root directory
./holohub run video_streaming client_python \
  --docker-file applications/video_streaming/Dockerfile \
  --docker-opts='-e EnableHybridMode=1' \
  --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'
```

Terminal 2 - Or start Python Client (V4L2 Camera):
```bash
# From holohub root directory
./holohub run video_streaming client_python_v4l2 \
  --docker-file applications/video_streaming/Dockerfile \
  --docker-opts='-e EnableHybridMode=1' \
  --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'
```

### Testing with C++ Server

You can also test the Python client with a C++ server (they are fully compatible):

Terminal 1 - C++ Server:
```bash
./holohub run video_streaming server \
  --docker-file applications/video_streaming/Dockerfile \
  --docker-opts='-e EnableHybridMode=1'
```

Terminal 2 - Python Client:
```bash
./holohub run video_streaming client_python \
  --docker-file applications/video_streaming/Dockerfile \
  --docker-opts='-e EnableHybridMode=1' \
  --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'
```

**Important:** Ensure client and server resolutions match for optimal performance.
