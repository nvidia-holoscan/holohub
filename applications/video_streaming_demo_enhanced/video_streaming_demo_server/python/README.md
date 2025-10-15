# Video Streaming Server Demo (Python)

This Python application demonstrates how to use the StreamingServer Python bindings to create a video streaming server that can receive and send video frames.

## Features

- **Bidirectional Streaming**: Receive frames from clients and send processed frames back
- **Standalone Mode**: Generate test video content and stream to clients
- **Frame Processing**: Optional frame processing (mirroring, filtering, etc.)
- **Multi-client Support**: Handle multiple client connections via shared resource
- **Configurable**: YAML configuration file support and command-line options

## Requirements

- Holoscan SDK 3.5.0 or higher with Python bindings
- Python 3.8 or higher
- Video files (for standalone mode, optional)

## Usage

### Basic Usage

```bash
# Run in bidirectional mode (default)
python3 streaming_server_demo.py

# Run in standalone mode (generate test video)
python3 streaming_server_demo.py --standalone

# Enable frame processing
python3 streaming_server_demo.py --enable-processing --processing-type mirror

# Use custom ports
python3 streaming_server_demo.py --signaling-port 8554 --streaming-port 8555
```

### Configuration File

```bash
# Create a default configuration file
python3 streaming_server_demo.py --create-config server_config.yaml

# Run with configuration file
python3 streaming_server_demo.py --config server_config.yaml
```

### Command Line Options

- `--signaling-port PORT`: Signaling port (default: 48010)
- `--streaming-port PORT`: Streaming port (default: 48020)
- `--config PATH`: Path to YAML configuration file
- `--enable-processing`: Enable frame processing
- `--processing-type {none,mirror,filter}`: Type of frame processing
- `--standalone`: Run in standalone mode
- `--create-config PATH`: Create default configuration file
- `--help`: Show help message

## Configuration

The application can be configured via YAML file or command-line arguments. Key configuration sections:

### Server Settings
```yaml
server:
  signaling_port: 48010
  streaming_port: 48020
  standalone_mode: false
```

### Stream Settings
```yaml
stream:
  width: 854
  height: 480
  fps: 30
```

### Processing Settings
```yaml
processing:
  enabled: false
  type: "none"  # Options: "none", "mirror", "filter"
```

## Operating Modes

### Bidirectional Mode (Default)

In bidirectional mode, the server:
1. Receives video frames from connected clients (StreamingServerUpstreamOp)
2. Optionally processes the frames (FormatConverterOp)
3. Sends processed frames back to clients (StreamingServerDownstreamOp)

### Standalone Mode

In standalone mode, the server:
1. Generates test video content (VideoStreamReplayerOp)
2. Converts format as needed (FormatConverterOp)
3. Streams content to connected clients (StreamingServerDownstreamOp)

## Pipeline Architecture

### Bidirectional Pipeline
```
Client Streams → StreamingServerUpstreamOp → [Processing] → StreamingServerDownstreamOp → Client Streams
```

### Standalone Pipeline
```
VideoStreamReplayer → FormatConverter → StreamingServerDownstreamOp → Client Streams
```

## Python Bindings Usage

The application demonstrates usage of the following Python bindings:

```python
from holoscan.operators import (
    StreamingServerUpstreamOp,
    StreamingServerDownstreamOp,
    StreamingServerResource
)

# Create shared resource
resource = StreamingServerResource(
    self,
    width=854, height=480, fps=30,
    signaling_port=48010, streaming_port=48020
)

# Create upstream operator (receives from clients)
upstream = StreamingServerUpstreamOp(
    self,
    streaming_server_resource=resource
)

# Create downstream operator (sends to clients)
downstream = StreamingServerDownstreamOp(
    self,
    enable_processing=True,
    processing_type="mirror",
    streaming_server_resource=resource
)
```

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

To test the server with the Python client:

1. Start the server:
   ```bash
   python3 streaming_server_demo.py
   ```

2. In another terminal, start the client:
   ```bash
   python3 ../video_streaming_demo_client/python/streaming_client_demo.py
   ```
