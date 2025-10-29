# Video Streaming Server Demo

This application demonstrates how to create a bidirectional video streaming server that receives frames from clients and sends them back. Both C++ and Python implementations are available.

> **📚 Related Documentation:**
>
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
- video_streaming operator

### Download Server Cloud Streaming

Download the Holoscan Server Cloud Streaming binaries from NGC:

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


## Usage

> ⚠️ The server applications require Holoscan SDK 3.5.0. Set the SDK version environment variable before running the applications in each terminal, or use the `--base-img` option to specify the base image.
>
> ```bash
> # Set SDK version environment variable
> export HOLOHUB_BASE_SDK_VERSION=3.5.0
> ```

### C++ Server

```bash
# From holohub root directory - runs with default settings (854x480 @ 30fps)
./holohub run video_streaming_server --language cpp
```

### Python Server

```bash
# From holohub root directory - runs with default settings (854x480 @ 30fps)
./holohub run video_streaming_server --language python

# With custom parameters via command-line arguments
./holohub run video_streaming_server --language python \
  --extra-args '--port 48010 --width 854 --height 480 --fps 30'
```

**Default Configuration:**

- **Port**: 48010
- **Resolution**: 854x480
- **Frame Rate**: 30 fps
- **Pipeline**: StreamingServerUpstreamOp → StreamingServerDownstreamOp (passthrough/echo mode)

**Note:** The server defaults to 854x480 resolution. For V4L2 clients that use 640x480, ensure the client is configured to match the server's resolution.

### Command Line Options

**Python Server**:

- `--port PORT`: Server port (default: 48010)
- `--width WIDTH`: Frame width (default: 854)
- `--height HEIGHT`: Frame height (default: 480)
- `--fps FPS`: Frames per second (default: 30)
- `--config PATH` or `-c PATH`: Path to YAML configuration file
- `--create-config PATH`: Create default configuration file at specified path
- `--help`: Show help message

**C++ Server**:

- `-c PATH` or `--config PATH`: Path to YAML configuration file
- `-?` or `--help`: Show help message

## Configuration

### C++ Configuration

The C++ application is configured via YAML file. Example configuration file structure:

```yaml
%YAML 1.2
---
# Application metadata
application:
  title: Streaming Server Test App
  version: 1.0
  log_level: INFO

# Streaming server settings
streaming_server:
  # Video/stream parameters
  width: 854
  height: 480
  fps: 30
  
  # Server connection settings
  port: 48010
  multi_instance: false
  server_name: "StreamingServerTest"
  
  # Operation mode - Bidirectional streaming
  receive_frames: true
  send_frames: true
  allocator: !ref "allocator"

# Upstream operator configuration (receives frames from clients)
upstream_op: {}

# Downstream operator configuration (sends frames to clients)
downstream_op: {}

# Memory allocator configuration
allocator:
  type: "holoscan::UnboundedAllocator"

# Scheduler configuration
scheduler: "multi_thread"

multi_thread_scheduler:
  worker_thread_number: 2
  stop_on_deadlock: true
  stop_on_deadlock_timeout: 5000

# Enable data flow tracking for debugging/profiling
tracking: false
```

**C++ Configuration File**: `cpp/streaming_server_demo.yaml`

### Python Configuration

The Python application is primarily configured via **command-line arguments**, with optional YAML support for advanced settings:

**Command-Line Parameters** (recommended):

- `--port PORT`: Server port (default: 48010)
- `--width WIDTH`: Frame width (default: 854)
- `--height HEIGHT`: Frame height (default: 480)
- `--fps FPS`: Frames per second (default: 30)
- `--config PATH` or `-c PATH`: Path to YAML configuration file
- `--create-config PATH`: Create default configuration file

**Python YAML Structure** (optional, different from C++):

```yaml
application:
  title: "Streaming Server Python Demo"
  version: "1.0"
  log_level: "INFO"

# Server configuration (if using YAML)
server:
  signaling_port: 48010
  streaming_port: 48020
  standalone_mode: false

# Stream settings
stream:
  width: 854
  height: 480
  fps: 30

# Scheduler configuration
scheduler: "multi_thread"

multi_thread_scheduler:
  worker_thread_number: 2
  stop_on_deadlock: true
  stop_on_deadlock_timeout: 5000
```

**Python Configuration File**: `python/streaming_server_demo.yaml`

**Note**: Python parameters set via command-line take precedence over YAML configuration. For most use cases, command-line arguments are sufficient.

## Pipeline Architecture

The server implements a simple bidirectional streaming pipeline:

```text
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

// Create shared streaming server resource from config
// Configuration loaded from YAML 'streaming_server' section
holoscan::ArgList streaming_server_args;
try {
    streaming_server_args = from_config("streaming_server");
} catch (const std::exception& e) {
    HOLOSCAN_LOG_WARN("Missing streaming_server config section, using defaults");
}
auto streaming_server_resource =
    make_resource<ops::StreamingServerResource>("streaming_server_resource",
                                                 streaming_server_args);

// Upstream operator (receives from clients)
// Configuration loaded from YAML 'upstream_op' section
holoscan::ArgList upstream_args;
try {
    upstream_args = from_config("upstream_op");
} catch (const std::exception& e) {
    HOLOSCAN_LOG_WARN("Missing upstream_op config section, using defaults");
}
auto upstream_op = make_operator<ops::StreamingServerUpstreamOp>("upstream_op", upstream_args);
upstream_op->add_arg(Arg("streaming_server_resource", streaming_server_resource));

// Downstream operator (sends to clients)
// Configuration loaded from YAML 'downstream_op' section
holoscan::ArgList downstream_args;
try {
    downstream_args = from_config("downstream_op");
} catch (const std::exception& e) {
    HOLOSCAN_LOG_WARN("Missing downstream_op config section, using defaults");
}
auto downstream_op =
    make_operator<ops::StreamingServerDownstreamOp>("downstream_op", downstream_args);
downstream_op->add_arg(Arg("streaming_server_resource", streaming_server_resource));

// Connect: upstream -> downstream (passthrough/echo mode)
add_flow(upstream_op, downstream_op, {{"output_frames", "input_frames"}});
```

**Key Points:**

- All operators use the `ops::` namespace prefix
- Configuration is loaded from YAML using `from_config()` for flexibility
- The `StreamingServerResource` is created first and passed to operators using `add_arg()`
- The resource is configured from the `streaming_server` YAML section
- Upstream and downstream operators are configured from their respective YAML sections (`upstream_op`, `downstream_op`)
- Both operators must reference the same shared `streaming_server_resource`
- This pattern allows for dynamic configuration without recompiling

## Python Implementation

The Python implementation (`python/streaming_server_demo.py`) demonstrates usage of the Python bindings:

```python
from holohub.video_streaming_server import (
    StreamingServerDownstreamOp,
    StreamingServerResource,
    StreamingServerUpstreamOp,
)

class StreamingServerApp(Application):
    def __init__(self, port=48010, width=854, height=480, fps=30):
        """Initialize the streaming server application.
        
        Args:
            port: Server port (set via --port command-line argument)
            width: Frame width (set via --width command-line argument)
            height: Frame height (set via --height command-line argument)
            fps: Frames per second (set via --fps command-line argument)
        """
        super().__init__()
        self.port = port
        self.width = width
        self.height = height
        self.fps = fps

    def compose(self):
        """Compose the application pipeline.
        
        Simple bidirectional streaming:
        upstream_op (receives from clients) -> downstream_op (sends back to clients)
        """
        
        # Create shared streaming server resource with parameters from constructor
        streaming_resource = StreamingServerResource(
            self,
            name="streaming_server_resource",
            port=self.port,
            width=self.width,
            height=self.height,
            fps=self.fps,
            enable_upstream=True,
            enable_downstream=True,
        )

        # Upstream operator (receives from clients)
        upstream_op = StreamingServerUpstreamOp(
            self, name="upstream_op", streaming_server_resource=streaming_resource
        )

        # Downstream operator (sends to clients)
        downstream_op = StreamingServerDownstreamOp(
            self, name="downstream_op", streaming_server_resource=streaming_resource
        )

        # Connect: upstream -> downstream (passthrough/echo mode)
        self.add_flow(upstream_op, downstream_op, {("output_frames", "input_frames")})
```

**Key Points:**

- Both operators share the same `StreamingServerResource` to manage streaming connections
- The resource is configured with `port`, `width`, `height`, `fps`, and enables both upstream and downstream
- The upstream operator receives frames from clients on its `output_frames` port
- The downstream operator receives those frames on its `input_frames` port and sends them back to clients
- This creates a simple passthrough/echo streaming pipeline
- **Parameters are set via constructor arguments** (from command-line or defaults), not from YAML
- The constructor parameters (`port`, `width`, `height`, `fps`) are passed directly to `StreamingServerResource`

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

## Configuration Examples

See the included configuration files for complete examples:

- `cpp/streaming_server_demo.yaml` - C++ server configuration (used by C++ application)
- `python/streaming_server_demo.yaml` - Python server configuration (optional, Python primarily uses command-line args)

**Note**: C++ and Python use different YAML structures. The C++ version uses `streaming_server` section with direct parameters, while Python uses `server` and `stream` sections. For Python, command-line arguments are the recommended configuration method.

## Integration with Client

### Testing with Python Client

#### Option 1: Using Holohub CLI (Recommended)

Terminal 1 - Start Python Server:

```bash
# From holohub root directory
./holohub run video_streaming server_python \
  --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'
```

Terminal 2 - Start Python Client with Video Replayer (854x480):

```bash
# From holohub root directory
./holohub run video_streaming client_python \
  --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'
```

### Testing with C++ Client

Terminal 1 - Start Server (C++ or Python):

```bash
# C++ Server
./holohub run video_streaming

# OR Python Server
./holohub run video_streaming server_python \
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
