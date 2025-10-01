# StreamingClient Operator

The StreamingClientOp class implements a Holoscan operator that provides bidirectional video streaming capabilities with the following key components:

- Configuration and Initialization:
- Configurable parameters for frame dimensions (width, height), frame rate (fps), server connection (IP, port)
- Input/output ports for frame data using GXF entities
- Support for both sending and receiving frames through separate flags

Frame Processing Pipeline:
- Input handling: Receives frames as GXF entities containing H.264 encoded video tensors
- Frame conversion: Converts input tensors to VideoFrame objects with BGRA format
- Memory management: Implements safe memory handling with bounds checking and zero-padding
- Output generation: Creates GXF entities with properly configured tensors for downstream processing

Streaming Protocol Implementation:
- Bidirectional streaming support through StreamingClient class
- Frame callback system for receiving frames
- Frame source system for sending frames
- Connection management with server including timeout handling


## Dependencies

In order to build the client operator, you must first download the client binaries from NGC:

```bash
# Download using NGC CLI

cd <your_holohub_path>/operators/streaming_client_enhanced
ngc registry resource download-version "nvidia/holoscan_client_cloud_streaming:0.2"
unzip -o holoscan_client_cloud_streaming_v0.2/holoscan_client_cloud_streaming.zip -d holoscan_client_cloud_streaming

# Clean up extraction directory and NGC download directory
rm -rf streaming_client_enhanced holoscan_client_cloud_streaming_v0.2
```

All dependencies need to be properly installed in the operator directory structure.

## Troubleshooting

If you encounter build errors:
- Make sure all required files are copied to the correct locations
- Check that the libraries have appropriate permissions (644)
- Ensure the directories exist inside the container environment 

## Camera Setup and Testing

### Testing Your V4L2 Camera

Before using the streaming client with your camera, verify it's working properly:

```bash
# Check available video devices
ls -la /dev/video*

# Get camera information
v4l2-ctl --device=/dev/video0 --info

# List supported formats and resolutions
v4l2-ctl --device=/dev/video0 --list-formats-ext

# Test camera capture (replace resolution as needed)
v4l2-ctl --device=/dev/video0 --set-fmt-video=width=1280,height=720,pixelformat=MJPG --stream-mmap --stream-count=10
```

### Configuring Camera Resolution

The streaming client applications use YAML configuration files to set camera parameters. Edit the appropriate config file:

#### For video_streaming_demo_client:
Edit `applications/video_streaming_demo_enhanced/video_streaming_demo_client/cpp/streaming_client_demo.yaml`:

```yaml
# V4L2 camera configuration
v4l2_source:
  device: "/dev/video0"        # Camera device path
  width: 1280                  # Camera resolution width
  height: 720                  # Camera resolution height  
  frame_rate: 30               # Camera frame rate
  pixel_format: "MJPG"         # Pixel format (MJPG recommended for higher resolutions)
  # Optional camera settings:
  # exposure_time: 100         # Exposure time in multiples of 100μs
  # gain: 10                   # Camera gain value
```

### Recommended Settings by Camera Type

**For Logitech HD Pro Webcam C920:**
- **1280x720 @ 30fps MJPG** - Best balance of quality and performance
- **1920x1080 @ 30fps MJPG** - High quality (higher bandwidth)
- **640x480 @ 30fps YUYV** - Low bandwidth testing

**General Guidelines:**
- Use **MJPG** format for resolutions above 640x480 for better performance
- Use **YUYV** format for lower resolutions or when uncompressed data is needed
- Start with 30 FPS and adjust based on your system performance
- Match the resolution between client and server applications

### Troubleshooting Camera Issues

**Camera not detected:**
```bash
# Check camera permissions
sudo usermod -a -G video $USER
# Log out and back in, then test again
```

**Poor performance:**
- Try lower resolution (e.g., 640x480)
- Switch from YUYV to MJPG format
- Reduce frame rate to 15 or 24 FPS

**Format not supported:**
```bash
# Check what formats your camera actually supports
v4l2-ctl --device=/dev/video0 --list-formats-ext | grep -E "Size:|Interval:"
```

## Testing

Testing is handled at the application level through the unified `video_streaming_demo_enhanced` integration test, which provides comprehensive end-to-end validation of the streaming client working with the server.

## Related Applications

- **[Streaming Client Demo Enhanced](../../applications/video_streaming_demo_enhanced/video_streaming_demo_client/README.md)** - Complete application demonstrating the streaming client operator
- **[Streaming Server Demo Enhanced](../../applications/video_streaming_demo_enhanced/video_streaming_demo_server/README.md)** - Companion server application for bidirectional streaming

## Supported Platforms

- Linux x86_64
- Linux aarch64
