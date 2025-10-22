# Video Streaming Server Demo

This application demonstrates how to create a bidirectional video streaming server that receives frames from clients and sends them back. Both C++ and Python implementations are available.

> **📚 Related Documentation:**
> - **[Main README](../README.md)** - Application overview, quick start, and common configuration
> - **[Client README](../video_streaming_client/README.md)** - Client setup and configuration
> - **[Testing Documentation](../TESTING.md)** - Integration testing and verification

## Features

- **Bidirectional Streaming**: Receives frames from clients (upstream) and sends frames back (downstream)
- **Multi-client Support**: Handles multiple client connections via shared StreamingServerResource
- **Simple Pipeline**: StreamingServerUpstreamOp → StreamingServerDownstreamOp (passthrough/echo mode)
- **Configurable**: YAML configuration file support and command-line options
- **C++ and Python**: Full implementations in both languages with compatible APIs
- **Real-time Processing**: Low-latency video streaming

## Requirements

- Holoscan SDK 3.5.0 or higher
- Custom Dockerfile with OpenSSL 3.4.0 (for running via holohub CLI)
- For Python: Python 3.8+ and bindings built with `-DHOLOHUB_BUILD_PYTHON=ON`
- CUDA 12.x (currently not working with CUDA 13.x)
- OpenCV
- video_streaming operator

## Usage

> [!IMPORTANT] The server applications requires Holoscan SDK 3.5.0. Either set the SDK version environment variable before running the applications, or use the `--base-img` option to specify the base image.
>
> ```bash
> # Set SDK version environment variable
> export HOLOHUB_BASE_SDK_VERSION=3.5.0
> ```

### C++ Server

```bash
# From holohub root directory - runs with default settings (854x480 @ 30fps)
./holohub run video_streaming
```

### Python Server

```bash
# From holohub root directory - runs with default settings (854x480 @ 30fps)
./holohub run video_streaming server_python \
  --docker-file applications/video_streaming/Dockerfile \
  --docker-opts='-e EnableHybridMode=1' \
  --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'
```

**Default Configuration:**
- **Port**: 48010
- **Resolution**: 854x480
- **Frame Rate**: 30 fps
- **Pipeline**: StreamingServerUpstreamOp → StreamingServerDownstreamOp (passthrough/echo mode)

**Note:** The server defaults to 854x480 resolution. For V4L2 clients that use 640x480, ensure the client is configured to match the server's resolution.

### Command Line Options (Python)

- `--port PORT`: Server port (default: 48010)
- `--width WIDTH`: Frame width (default: 854)
- `--height HEIGHT`: Frame height (default: 480)
- `--fps FPS`: Frames per second (default: 30)
- `--config PATH` or `-c PATH`: Path to YAML configuration file
- `--create-config PATH`: Create default configuration file at specified path
- `--help`: Show help message

### Command Line Options (C++)

- `-c PATH` or `--config PATH`: Path to YAML configuration file

## Configuration

The application can be configured via YAML file or command-line arguments. Example configuration file structure:

```yaml
# Application metadata
application:
  title: "Streaming Server Demo"
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
  multi_instance: false
  server_name: "StreamingServerTest"

# Scheduler configuration
scheduler: "multi_thread"

multi_thread_scheduler:
  worker_thread_number: 2
  stop_on_deadlock: true
  stop_on_deadlock_timeout: 5000
```

Configuration files are located in:
- C++: `cpp/streaming_server_demo.yaml`
- Python: `python/streaming_server_demo.yaml`

## Pipeline Architecture

The server implements a simple bidirectional streaming pipeline:

```
Client Streams → StreamingServerUpstreamOp → StreamingServerDownstreamOp → Client Streams
```

### How It Works

1. **StreamingServerUpstreamOp**: Receives video frames from connected streaming clients
2. **StreamingServerDownstreamOp**: Sends frames back to all connected streaming clients
3. Both operators share a **StreamingServerResource** that manages the streaming connections

## C++ Implementation

The C++ implementation (`cpp/streaming_server_demo.cpp`) demonstrates usage of the streaming server operators:

```cpp
#include "streaming_server_downstream_op.hpp"
#include "streaming_server_resource.hpp"
#include "streaming_server_upstream_op.hpp"

// Create shared streaming server resource
auto streaming_resource = make_resource<StreamingServerResource>(
    "streaming_server_resource",
    Arg("port", 48010),
    Arg("width", 854U),
    Arg("height", 480U),
    Arg("fps", uint16_t{30}),
    Arg("enable_upstream", true),
    Arg("enable_downstream", true)
);

// Upstream operator (receives from clients)
auto upstream_op = make_operator<StreamingServerUpstreamOp>(
    "upstream_op",
    Arg("streaming_server_resource", streaming_resource)
);

// Downstream operator (sends to clients)
auto downstream_op = make_operator<StreamingServerDownstreamOp>(
    "downstream_op",
    Arg("streaming_server_resource", streaming_resource)
);

// Connect: upstream -> downstream (passthrough/echo mode)
add_flow(upstream_op, downstream_op, {{"output_frames", "input_frames"}});
```

## Python Implementation

The Python implementation (`python/streaming_server_demo.py`) demonstrates usage of the Python bindings:

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
- Both operators share the same `StreamingServerResource` to manage streaming connections
- The upstream operator receives frames from clients on its `output_frames` port
- The downstream operator receives those frames on its `input_frames` port and sends them back to clients
- This creates a simple passthrough/echo streaming pipeline

## Troubleshooting

### Common Issues

1. **Port Already in Use**: Check if ports are available or use different ports
   ```bash
   # Check if port is in use
   netstat -tlnp | grep 48010
   sudo lsof -ti:48010 | xargs sudo kill -9
   ```

2. **No Clients Connected**: Verify client configuration matches server ports and resolution

3. **Import Error (Python)**: Ensure Holoscan SDK Python bindings are installed

4. **Video Files Not Found**: Check data directory path (standalone mode)

5. **Build Failures**: Clean build and retry
   ```bash
   rm -rf build/
   ./holohub build video_streaming --language cpp
   # or for Python
   ./holohub build video_streaming --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'
   ```

### Debug Mode

Enable debug logging by setting log level in configuration:

```yaml
application:
  log_level: "DEBUG"
```

## Examples

See the included configuration files for complete examples:
- `cpp/streaming_server_demo.yaml` - C++ server configuration
- `python/streaming_server_demo.yaml` - Python server configuration

## Integration with Client

### Testing with Python Client

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

### Testing with C++ Client

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
- C++ and Python implementations are fully compatible - you can mix and match (C++ server with Python client, etc.)
- Ensure client and server resolutions match for optimal performance
- The server must be started before the client

## Dependencies

- Holoscan SDK 3.5.0
- video_streaming operator
- OpenCV
- CUDA 12.x
- OpenSSL 3.4.0

## See Also

- [Main Video Streaming README](../README.md) - Complete application documentation with integration testing
- [Video Streaming Client](../video_streaming_client/README.md) - Client application documentation
