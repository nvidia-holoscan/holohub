# Streaming Server Demo

This application demonstrates how to use the Holoscan SDK to create a streaming server application that can send and receive video streams to/from streaming clients. 

**Note**: This demo provides a basic echo server that receives frames from clients and sends them back without server-side processing. The server acts as a passthrough, allowing you to test bidirectional streaming functionality.

## Requirements

- NVIDIA GPU
- CUDA 12.1 or higher
- Holoscan SDK 3.4.0 or higher

## Dependencies

In order to build the server operator, you must first download the server binaries from NGC:

```bash
# Download using NGC CLI
cd <your_holohub_path>/operators/streaming_server_enhanced
ngc registry resource download-version "nvidia/holoscan/holoscan_server_cloud_streaming:1.2"
unzip -o holoscan_server_cloud_streaming_v1.0/holoscan_server_cloud_streaming.zip

# Move the extracted contents to the expected location
mv streaming_server_enhanced/holoscan_server_cloud_streaming ./

# Clean up extraction directory and NGC download directory
rm -rf streaming_server_enhanced holoscan_server_cloud_streaming_v1.0
```



## Running the Application

To run the application use this command from the holohub root directory:

```bash
./holohub run --docker-opts='-e EnableHybridMode=1' streaming_server_demo_enhanced --language cpp
```

### Command Line Options

- `-h, --help`: Show help message
- `-c, --config <file>`: Configuration file path (default: streaming_server_demo_04_80.yaml)
- `-d, --data <directory>`: Data directory (default: environment variable HOLOSCAN_INPUT_PATH or current directory)

## Configuration

The application can be configured using a YAML file. By default, it looks for `streaming_server_demo_04_80.yaml` in the current directory.

### Server Resolution Configuration

Edit `streaming_server_demo_04_80.yaml` to configure the streaming server resolution:

```yaml
# Streaming server settings
streaming_server:
  # Video/stream parameters
  width: 1280          # Frame width in pixels
  height: 720          # Frame height in pixels  
  fps: 30              # Frame rate
  
  # Server connection settings
  port: 48010          # Streaming port
  multi_instance: false
  server_name: "StreamingServerTest"
  
  # Operation mode
  receive_frames: true
  send_frames: true
```

### Python Application Options

The Python version supports command-line resolution configuration:

```bash
# Run with custom resolution
python streaming_server_demo.py --width 1280 --height 720 --port 48010

# Run with 1080p resolution
python streaming_server_demo.py --width 1920 --height 1080 --fps 30

# Test with different configurations
python streaming_server_demo.py --width 1280 --height 720 --fps 60
```

### Recommended Settings

**For streaming with V4L2 cameras (like Logitech C920):**
- **1280x720 @ 30fps** - Best balance of quality and performance
- **1920x1080 @ 30fps** - High quality streaming
- **854x480 @ 30fps** - Default, good for testing

**Important:** Make sure the server resolution matches your camera's resolution configured in the streaming client for optimal performance.

## Camera Setup and Testing

### Testing Your V4L2 Camera

Before starting the streaming server, verify your camera is working properly:

```bash
# Check available video devices
ls -la /dev/video*

# Get camera information
v4l2-ctl --device=/dev/video0 --info

# Test camera with recommended resolution
v4l2-ctl --device=/dev/video0 --set-fmt-video=width=1280,height=720,pixelformat=MJPG --stream-mmap --stream-count=10
```

### Troubleshooting

**Camera not detected:**
```bash
# Check camera permissions
sudo usermod -a -G video $USER
# Log out and back in, then test again
```

**Poor streaming performance:**
- Try lower resolution (e.g., 854x480 or 640x480)
- Reduce frame rate to 15 or 24 FPS
- Ensure client and server resolutions match

**Connection issues:**
- Check that the port (default 48010) is not blocked by firewall
- Verify server and client are using the same resolution settings
- Test with `netstat -tlnp | grep 48010` to see if port is listening

## Related Applications

- [Streaming Client Demo Application](../streaming_client_demo_enhanced/README.md) 