# Video Streaming Client Demo (Python)

This Python application demonstrates how to use the StreamingClientOp Python bindings to create a video streaming client that can send and receive video frames.

## Features

- **Video Sources**: Support for video file playback and V4L2 camera capture
- **Real-time Streaming**: Send video frames to a streaming server
- **Bidirectional**: Receive processed frames back from the server
- **Format Conversion**: Automatic RGB to BGR conversion for compatibility
- **Visualization**: Optional HoloViz integration for displaying received frames
- **Configurable**: YAML configuration file support and command-line options

## Requirements

- Holoscan SDK 3.5.0 or higher with Python bindings
- Python 3.8 or higher
- Video files (for replayer mode) or V4L2 compatible camera (for camera mode)

## Usage

### Basic Usage

```bash
# Run with video file playback (default)
python3 streaming_client_demo.py

# Run with camera capture
python3 streaming_client_demo.py --source v4l2

# Connect to specific server
python3 streaming_client_demo.py --server-ip 192.168.1.100 --signaling-port 8554

# Disable visualization
python3 streaming_client_demo.py --no-viz
```

### Configuration File

```bash
# Create a default configuration file
python3 streaming_client_demo.py --create-config client_config.yaml

# Run with configuration file
python3 streaming_client_demo.py --config client_config.yaml
```

### Command Line Options

- `--source {replayer,v4l2}`: Video source type (default: replayer)
- `--config PATH`: Path to YAML configuration file
- `--server-ip IP`: Server IP address (default: 127.0.0.1)
- `--signaling-port PORT`: Server signaling port (default: 48010)
- `--no-viz`: Disable visualization
- `--create-config PATH`: Create default configuration file
- `--help`: Show help message

## Configuration

The application can be configured via YAML file or command-line arguments. Key configuration sections:

### Video Source
```yaml
source: "replayer"  # or "v4l2"

replayer:
  directory: "/workspace/holohub/data/endoscopy"
  basename: "surgical_video"
  frame_rate: 30

v4l2:
  device: "/dev/video0"
  width: 640
  height: 480
  frame_rate: 30
```

### Streaming Client
```yaml
streaming_client:
  width: 854
  height: 480
  fps: 30
  server_ip: "127.0.0.1"
  signaling_port: 48010
  send_frames: true
  receive_frames: true
```

## Pipeline Architecture

The application creates the following processing pipeline:

1. **Video Source** (VideoStreamReplayer or V4L2VideoCaptureOp)
2. **Format Converter** (RGB to BGR conversion)
3. **Streaming Client** (StreamingClientOp)
4. **Visualization** (HoloVizOp, optional)

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

See the included `streaming_client_demo.yaml` for a complete configuration example.
