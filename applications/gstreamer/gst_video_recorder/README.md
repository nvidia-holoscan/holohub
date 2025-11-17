# GStreamer Video Recorder

A Holoscan application that demonstrates video recording using the GStreamer encoding pipeline.

## Description

This application showcases how to:
- Generate video frames in Holoscan using the pattern generator
- Feed video frames to GStreamer for encoding
- Record encoded video to files in various formats (MP4, MKV)
- Use different video codecs (H.264, H.265, and other GStreamer-supported codecs)
- Support both host and CUDA device memory for zero-copy operation

## Requirements

- NVIDIA Holoscan SDK
- GStreamer 1.0 with the following plugins:
  - gstreamer1.0-plugins-base (videoconvert for host memory support)
  - gstreamer1.0-plugins-bad (cudaconvert, nvh264enc, nvh265enc for NVIDIA hardware encoding)
  - gstreamer1.0-plugins-good (mp4mux, matroskamux for container formats)
  - gstreamer1.0-plugins-ugly (x264enc for CPU-based H.264 encoding)
  - Additional codecs available through gstreamer1.0-libav if needed

## Building

### Option 1: Containerized Build (Recommended)
No setup required - all dependencies are included in the container:

```bash
./holohub build gst_video_recorder
```

### Option 2: Local Build
For faster builds and easier debugging. First install dependencies:

```bash
# From the gst_video_recorder directory
./install_deps.sh

# Then build locally
./holohub build --local gst_video_recorder
```

The `install_deps.sh` script installs:
- pkg-config (required for CMake)
- GStreamer development libraries
- All necessary GStreamer plugins for encoding

## Usage

```bash
gst-video-recorder [OPTIONS]
```

### Options

- `-o, --output <filename>` - Output video filename (default: output.mp4)
  - Supported formats: .mp4, .mkv
  - If no extension, defaults to .mp4
- `-e, --encoder <name>` - Encoder base name (default: nvh264)
  - Examples: nvh264, nvh265, x264, x265
  - Note: 'enc' suffix is automatically appended
- `-c, --count <number>` - Number of frames to generate (default: unlimited)
- `-w, --width <pixels>` - Frame width (default: 1920)
- `-h, --height <pixels>` - Frame height (default: 1080)
- `-f, --framerate <rate>` - Frame rate as fraction or decimal (default: 30/1)
  - Examples: '30/1', '30000/1001', '29.97', '60'
  - Use '0/1' for live mode (no throttling, real-time timestamps)
- `--pattern <type>` - Pattern type (default: 0)
  - 0 = animated gradient
  - 1 = animated checkerboard
  - 2 = color bars (SMPTE style)
- `--storage <type>` - Memory storage type (default: 1)
  - 0 = host memory
  - 1 = device/CUDA memory
- `--property <key=value>` - Set encoder property (can be used multiple times)
  - Examples: --property bitrate=8000 --property preset=1
  - Property types are automatically detected and converted
- `--help` - Show help message

### Examples

#### Record 10 seconds of video at 30fps (300 frames)

```bash
gst-video-recorder --count 300 -o video.mp4
```

#### Record high quality H.265 video

```bash
gst-video-recorder --count 300 --encoder nvh265 --property bitrate=10000 -o video.mp4
```

#### Record 720p video

```bash
gst-video-recorder --count 300 --width 1280 --height 720 -o video_720p.mp4
```

#### Record using host memory (CPU)

```bash
gst-video-recorder --count 300 --storage 0 --encoder x264 -o video.mp4
```

#### Record with H.265 to MKV container

```bash
gst-video-recorder --count 300 --encoder nvh265 -o video.mkv
```

#### Record animated checkerboard pattern

```bash
gst-video-recorder --count 300 --pattern 1 -o checkerboard.mp4
```

#### Record with custom encoder properties

```bash
gst-video-recorder --count 300 --property bitrate=8000 --property preset=1 --property gop-size=30 -o custom.mp4
```

#### Record with NTSC framerate (29.97 fps)

```bash
gst-video-recorder --count 300 --framerate 30000/1001 -o ntsc.mp4
```

## Architecture

The application consists of two main components:

1. **PatternGenOperator**: Generates animated test patterns as Holoscan entities with tensors
2. **GstVideoRecorderOperator**: Receives video frames, manages the GStreamer pipeline, and handles encoding

### Pipeline Flow

```
PatternGenOperator → GstVideoRecorderOperator → GStreamer Encoding Pipeline → File
```

The GStreamer encoding pipeline is automatically constructed based on the encoder and file format:

- **Pipeline structure**: `[converter] ! [encoder]enc ! [parser] ! [muxer] ! filesink`
- **Converter**: Automatically selected based on memory type (videoconvert for host, cudaconvert for device)
- **Encoder**: Specified via `--encoder` option (nvh264, nvh265, x264, x265, etc.)
- **Parser**: Automatically determined from encoder (h264parse, h265parse, etc.)
- **Muxer**: Automatically determined from file extension (mp4mux for .mp4, matroskamux for .mkv)

Example pipelines:

- **NVIDIA H.264 to MP4**: `cudaconvert ! nvh264enc ! h264parse ! mp4mux ! filesink`
- **NVIDIA H.265 to MKV**: `cudaconvert ! nvh265enc ! h265parse ! matroskamux ! filesink`
- **CPU x264 to MP4**: `videoconvert ! x264enc ! h264parse ! mp4mux ! filesink`

## Performance

The application supports both host and device (CUDA) memory:

- **Device memory** (`--storage 1`, default): Zero-copy operation for better performance when using NVIDIA hardware encoders (nvh264enc, nvh265enc)
- **Host memory** (`--storage 0`): Required for CPU encoders (x264, x265) but involves memory copies

## Notes

- The pattern generator supports three test patterns:
  - Animated gradient (default): Colorful sine wave patterns
  - Animated checkerboard: Moving checkerboard with variable square size
  - Color bars: SMPTE-style color bars (7 colors)
- To use your own video source, replace `PatternGenOperator` with your video capture operator
- The application waits for encoding to complete before exiting to ensure proper file finalization
- EOS (End-Of-Stream) signal is sent automatically when recording completes
- Video parameters (width, height, format, storage) are automatically detected from incoming frames

## See Also

- `holopattern_to_gst` - Similar application that can output to any GStreamer pipeline
- GStreamer documentation: https://gstreamer.freedesktop.org/documentation/


