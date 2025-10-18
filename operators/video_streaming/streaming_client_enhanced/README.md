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

> **📚 Related Documentation:**
> - **[Main Operators README](../README.md)** - Setup, dependencies, NGC downloads, and Python examples
> - **[Client Application README](../../../applications/video_streaming/video_streaming_client/README.md)** - Complete client application with usage examples
> - **[Server Operator README](../streaming_server_enhanced/README.md)** - Companion server operator documentation
> - **[Testing Documentation](../../../applications/video_streaming/TESTING.md)** - Integration testing and verification

## Architecture Overview

The StreamingClient operator integrates with the Holoscan Client Cloud Streaming library to provide seamless video streaming capabilities:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Holoscan Application                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────────────────────────────────────────┐ │
│  │   Input Source  │    │              StreamingClientOp                      │ │
│  │                 │    │                                                     │ │
│  │  • V4L2 Camera  │───▶│  ┌─────────────────┐    ┌─────────────────────────┐ │ │
│  │  • Video File   │    │  │  Frame Convert  │    │    VideoFrame Object   │ │ │
│  │  • Tensor Data  │    │  │  BGR → BGRA     │───▶│    • Width/Height      │ │ │
│  │                 │    │  │  Validation     │    │    • Pixel Data        │ │ │
│  └─────────────────┘    │  └─────────────────┘    │    • Format (BGRA)     │ │ │
│                         │                         │    • Timestamp         │ │ │
│  ┌─────────────────┐    │                         └─────────────────────────┘ │ │
│  │  Output Sink    │◀───┤                                      │              │ │
│  │                 │    │                                      ▼              │ │
│  │  • HoloViz      │    │  ┌─────────────────────────────────────────────────┐ │ │
│  │  • File Writer  │    │  │         Holoscan Client Cloud Streaming        │ │ │
│  │  • Next Op      │    │  │                                                 │ │ │
│  └─────────────────┘    │  │  ┌─────────────────┐    ┌─────────────────────┐ │ │ │
│                         │  │  │ StreamingClient │    │   Network Protocol  │ │ │ │
│                         │  │  │                 │    │                     │ │ │ │
│                         │  │  │ • sendFrame()   │───▶│  • WebRTC/NVST     │ │ │ │
│                         │  │  │ • Callbacks     │    │  • Signaling       │ │ │ │
│                         │  │  │ • Connection    │    │  • Media Transport  │ │ │ │
│                         │  │  │   Management    │    │  • Encryption       │ │ │ │
│                         │  │  └─────────────────┘    └─────────────────────┘ │ │ │
│                         │  │                                      │          │ │ │
│                         │  └──────────────────────────────────────┼──────────┘ │ │
│                         │                                         │            │ │
│                         └─────────────────────────────────────────┼────────────┘ │
└───────────────────────────────────────────────────────────────────┼──────────────┘
                                                                    │
                          ┌─────────────────────────────────────────┼──────────────┐
                          │                    Network                             │
                          │                                                        │
                          │  ┌─────────────────────────────────────────────────┐   │
                          │  │              Streaming Server                   │   │
                          │  │                                                 │   │
                          │  │  • Holoscan Server Cloud Streaming             │   │
                          │  │  • Multi-client support                        │   │
                          │  │  • Bidirectional communication                 │   │
                          │  │  • Frame processing and relay                  │   │
                          │  └─────────────────────────────────────────────────┘   │
                          └────────────────────────────────────────────────────────┘
```

### Component Interactions

1. **Input Processing**: The operator receives video frames from upstream Holoscan operators (V4L2, video replayer, etc.)

2. **Frame Conversion**: Input tensors are converted to VideoFrame objects with proper format validation and memory management

3. **Cloud Streaming Integration**: The VideoFrame is passed to the Holoscan Client Cloud Streaming library via `StreamingClient::sendFrame()`

4. **Network Transport**: The cloud streaming library handles:
   - WebRTC/NVST protocol implementation
   - Signaling and connection establishment
   - Media encoding and transport
   - Security and encryption

5. **Bidirectional Communication**: Frames received from the server are processed through callbacks and converted back to Holoscan tensors

6. **Output Generation**: Processed frames are emitted as GXF entities for downstream operators (HoloViz, file writers, etc.)

## Requirements & Setup

For complete setup instructions including:
- Holoscan SDK 3.5.0 and CUDA 12.x requirements
- NGC binary downloads (client streaming binaries)
- Build troubleshooting

**See the [Main Operators README](../README.md) for detailed setup instructions.** 

## Camera Setup and Testing

This section provides detailed technical camera configuration for the StreamingClient operator. For application-level camera setup and quick start instructions, see the [Application README](../../../applications/video_streaming/README.md#camera-setup-and-testing).

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

#### For video_streaming_client:
Edit `../../../applications/video_streaming/video_streaming_client/cpp/streaming_client_demo.yaml`:

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

## FrameSaver Utility Class

The `FrameSaverOp` is a utility operator that can save video frames to disk for debugging and analysis purposes. This operator is not integrated into the main streaming client but can be used as a standalone debugging tool.

### Features

- **Frame Capture**: Saves individual video frames to disk
- **Multiple Formats**: Supports both raw binary (.raw) and BGR format (.bgr) output
- **GPU/CPU Support**: Automatically handles frames from both GPU and CPU memory
- **Configurable Output**: Customizable output directory and filename patterns
- **Data Analysis**: Includes frame content analysis and logging

### Usage

#### Basic Configuration

```yaml
# Add FrameSaverOp to your Holoscan application
frame_saver:
  output_dir: "debug_frames"           # Directory to save frames
  base_filename: "frame_"              # Base filename for saved frames
  save_as_raw: false                   # false = .bgr format, true = .raw format
```

#### Integration Example

```cpp
#include "frame_saver.hpp"

// In your application setup
auto frame_saver = make_operator<holoscan::ops::FrameSaverOp>(
    "frame_saver",
    holoscan::Arg("output_dir", std::string("debug_frames")),
    holoscan::Arg("base_filename", std::string("frame_")),
    holoscan::Arg("save_as_raw", false)
);

// Connect to your frame source
add_flow(frame_saver, source, {{"input_frames", "output_frames"}});
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | string | "output_frames" | Directory where frames will be saved |
| `base_filename` | string | "frame_" | Base name for saved frame files |
| `save_as_raw` | bool | false | Whether to save as raw binary (.raw) or BGR format (.bgr) |

#### Output Files

- **BGR Format (.bgr)**: Standard BGR pixel format, can be opened with image viewers
- **Raw Format (.raw)**: Binary data, useful for debugging memory layouts
- **Naming**: `frame_000001.bgr`, `frame_000002.bgr`, etc.

#### Debugging Features

The FrameSaver includes built-in debugging features:

- **Content Analysis**: Logs frame size, zero-pixel count, and data validity
- **Memory Handling**: Automatically copies GPU frames to CPU before saving
- **Error Handling**: Comprehensive error reporting for file operations

#### Example Output

```
Frame 0 data analysis: size=1843200, all_zeros=false, non_zero_count=95
Saved frame 0 to debug_frames/frame_000001.bgr
```

### Building the FrameSaver

To use the FrameSaver in your application, you'll need to:

1. **Include the source files** in your CMakeLists.txt:
```cmake
add_library(frame_saver
  frame_saver.cpp
  frame_saver.hpp
)
```

2. **Link against required libraries**:
```cmake
target_link_libraries(frame_saver
  holoscan::core
  CUDA::cudart
)
```

### Use Cases

- **Debugging**: Save frames at specific points in your pipeline
- **Analysis**: Examine frame content and format
- **Testing**: Verify frame data integrity
- **Development**: Visual inspection of processed frames

## Python Bindings & Applications

For Python usage, application examples, and testing:
- **[Main Operators README](../README.md)** - Python bindings overview and setup
- **[Client Application README](../../../applications/video_streaming/video_streaming_client/README.md)** - Complete Python client implementation
- **[Testing Documentation](../../../applications/video_streaming/TESTING.md)** - Integration testing guide
