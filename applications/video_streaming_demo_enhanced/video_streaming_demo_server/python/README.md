# Video Streaming Server Demo (Python)

This Python application demonstrates how to use the StreamingServer Python bindings to create a bidirectional video streaming server that receives frames from clients and sends them back.

## Features

- **Bidirectional Streaming**: Receives frames from clients (upstream) and sends frames back (downstream)
- **Multi-client Support**: Handles multiple client connections via shared StreamingServerResource
- **Simple Pipeline**: StreamingServerUpstreamOp → StreamingServerDownstreamOp (passthrough/echo mode)
- **Configurable**: YAML configuration file support and command-line options
- **Python Bindings**: Uses `holohub.streaming_server_enhanced` Python bindings

## Requirements

- Holoscan SDK 3.5.0 or higher with Python bindings
- Python 3.8 or higher
- Custom Dockerfile with OpenSSL 3.4.0 (for running via holohub CLI)
- Python bindings must be built with `-DHOLOHUB_BUILD_PYTHON=ON`

## Usage

Run the Python streaming server using the Holohub CLI:

```bash
# From holohub root directory - runs with default settings (854x480 @ 30fps)
./holohub run video_streaming_demo_enhanced server_python \
  --docker-file applications/video_streaming_demo_enhanced/Dockerfile \
  --docker-opts='-e EnableHybridMode=1' \
  --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'
```

**Default Configuration:**
- **Port**: 48010
- **Resolution**: 854x480
- **Frame Rate**: 30 fps
- **Pipeline**: StreamingServerUpstreamOp → StreamingServerDownstreamOp (passthrough/echo mode)

**Note:** The server defaults to 854x480 resolution. For V4L2 clients that use 640x480, ensure the client is configured to match the server's resolution, or see the main README for custom resolution options.

### Command Line Options

- `--port PORT`: Server port (default: 48010)
- `--width WIDTH`: Frame width (default: 854)
- `--height HEIGHT`: Frame height (default: 480)
- `--fps FPS`: Frames per second (default: 30)
- `--config PATH` or `-c PATH`: Path to YAML configuration file
- `--create-config PATH`: Create default configuration file at specified path
- `--help`: Show help message

## Configuration

The application can be configured via YAML file or command-line arguments. Example configuration file structure:

```yaml
# Application metadata
application:
  title: "Streaming Server Python Demo"
  version: "1.0"
  log_level: "INFO"

# Server configuration
streaming_server:
  port: 48010           # Server port
  width: 854            # Frame width
  height: 480           # Frame height
  fps: 30               # Frames per second
  server_ip: "127.0.0.1"
  receive_frames: true
  send_frames: true
  visualize_frames: false

# Scheduler configuration
scheduler: "greedy"
```

## Pipeline Architecture

The Python server implements a simple bidirectional streaming pipeline:

```
Client Streams → StreamingServerUpstreamOp → StreamingServerDownstreamOp → Client Streams
```

### How It Works

1. **StreamingServerUpstreamOp**: Receives video frames from connected streaming clients
2. **StreamingServerDownstreamOp**: Sends frames back to all connected streaming clients
3. Both operators share a **StreamingServerResource** that manages the WebRTC connections

## Python Bindings Usage

The application demonstrates usage of the `holohub.streaming_server_enhanced` Python bindings:

```python
from holohub.streaming_server_enhanced import (
    StreamingServerDownstreamOp,
    StreamingServerResource,
    StreamingServerUpstreamOp,
)

# Create shared streaming server resource
streaming_resource = StreamingServerResource(
    self,
    name="streaming_server_resource",
    port=48010,
    width=854,
    height=480,
    fps=30,
    enable_upstream=True,
    enable_downstream=True,
)

# Upstream operator (receives from clients)
upstream_op = StreamingServerUpstreamOp(
    self,
    name="upstream_op",
    streaming_server_resource=streaming_resource
)

# Downstream operator (sends to clients)
downstream_op = StreamingServerDownstreamOp(
    self,
    name="downstream_op",
    streaming_server_resource=streaming_resource
)

# Connect: upstream -> downstream (passthrough/echo mode)
self.add_flow(upstream_op, downstream_op, {("output_frames", "input_frames")})
```

**Key Points:**
- Both operators share the same `StreamingServerResource` to manage WebRTC connections
- The upstream operator receives frames from clients on its `output_frames` port
- The downstream operator receives those frames on its `input_frames` port and sends them back to clients
- This creates a simple passthrough/echo streaming pipeline

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure Holoscan SDK Python bindings are installed
2. **Port Already in Use**: Check if ports are available or use different ports
3. **No Clients Connected**: Verify client configuration matches server ports
4. **Video Files Not Found**: Check data directory path (standalone mode)

### Debug Mode

Enable debug logging by setting log level in configuration:

```yaml
application:
  log_level: "DEBUG"
```

## Examples

See the included `streaming_server_demo.yaml` for a complete configuration example.

## Integration with Client

### Testing with Python Client

**Option 1: Using Holohub CLI (Recommended)**

Terminal 1 - Start Python Server:
```bash
# From holohub root directory
./holohub run video_streaming_demo_enhanced server_python \
  --docker-file applications/video_streaming_demo_enhanced/Dockerfile \
  --docker-opts='-e EnableHybridMode=1' \
  --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'
```

Terminal 2 - Start Python Client with Video Replayer (854x480):
```bash
# From holohub root directory
./holohub run video_streaming_demo_enhanced client_python \
  --docker-file applications/video_streaming_demo_enhanced/Dockerfile \
  --docker-opts='-e EnableHybridMode=1' \
  --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'
```

**Option 2: Direct Python Execution (for custom resolutions)**

After building with `./holohub build video_streaming_demo_enhanced --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'`:

Terminal 1 - Start server with 640x480 (for V4L2 clients):
```bash
cd build-video_streaming_demo_enhanced
python3 streaming_server_demo.py --width 640 --height 480
```

Terminal 2 - Start client with V4L2 camera:
```bash
cd build-video_streaming_demo_enhanced
python3 streaming_client_demo.py --source v4l2 --width 640 --height 480
```

### Testing with C++ Client

You can also test the Python server with a C++ client (they are fully compatible):

Terminal 1 - Python Server:
```bash
./holohub run video_streaming_demo_enhanced server_python \
  --docker-file applications/video_streaming_demo_enhanced/Dockerfile \
  --docker-opts='-e EnableHybridMode=1' \
  --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'
```

Terminal 2 - C++ Client:
```bash
./holohub run video_streaming_demo_client --language cpp \
  --docker-file applications/video_streaming_demo_enhanced/Dockerfile \
  --docker-opts='-e EnableHybridMode=1' \
  --run-args='-c streaming_client_demo_replayer.yaml'
```

**Important:** Ensure client and server resolutions match for optimal performance.
