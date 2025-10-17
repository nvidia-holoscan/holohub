# GStreamer Video Recorder

A Holoscan application that demonstrates video recording using the GStreamer encoding pipeline.

## Description

This application showcases how to:
- Generate video frames in Holoscan using the pattern generator
- Feed video frames to GStreamer for encoding
- Record encoded video to files in various formats (MP4, WebM)
- Use different video codecs (H.264, H.265, VP8, VP9)
- Support both host and CUDA device memory for zero-copy operation

## Requirements

- NVIDIA Holoscan SDK
- GStreamer 1.0 with the following plugins:
  - gstreamer1.0-plugins-base (videoconvert, videoscale)
  - gstreamer1.0-plugins-good (various codecs)
  - gstreamer1.0-plugins-ugly (x264enc for H.264)
  - gstreamer1.0-libav (various codecs)

## Building

```bash
./holohub build --local gst_video_recorder
```

## Usage

```bash
gst-video-recorder [OPTIONS]
```

### Options

- `-o, --output <file>` - Output file path (default: output.mp4)
- `-c, --codec <codec>` - Video codec: h264, h265, vp8, vp9 (default: h264)
- `-b, --bitrate <kbps>` - Bitrate in kbps (default: 5000)
- `-n, --frames <number>` - Number of frames to record (default: 300)
- `-w, --width <pixels>` - Frame width (default: 1920)
- `-h, --height <pixels>` - Frame height (default: 1080)
- `-f, --framerate <fps>` - Frame rate (default: 30)
- `--storage <type>` - Memory storage: 0=host, 1=device/CUDA (default: 0)
- `--help` - Show help message

### Examples

#### Record 10 seconds of video at 30fps

```bash
gst-video-recorder -n 300 -o video.mp4
```

#### Record high quality H.265 video

```bash
gst-video-recorder -c h265 -b 10000 -o video.mp4
```

#### Record 720p video

```bash
gst-video-recorder -w 1280 -h 720 -o video_720p.mp4
```

#### Record using CUDA device memory (zero-copy)

```bash
gst-video-recorder --storage 1 -o video.mp4
```

#### Record with VP9 codec to WebM

```bash
gst-video-recorder -c vp9 -o video.webm
```

## Architecture

The application consists of three main components:

1. **PatternGenOperator**: Generates animated test patterns as Holoscan entities with tensors
2. **GstVideoRecorderOperator**: Receives video frames and pushes them to GStreamer
3. **GStreamerApp**: Manages the encoding pipeline (encoder → muxer → file sink)

### Pipeline Flow

```
PatternGenOperator → GstVideoRecorderOperator → GStreamer Encoding Pipeline → File
```

The GStreamer encoding pipeline varies by codec:

- **H.264**: `videoconvert ! x264enc ! h264parse ! mp4mux ! filesink`
- **H.265**: `videoconvert ! x265enc ! h265parse ! mp4mux ! filesink`
- **VP8**: `videoconvert ! vp8enc ! webmmux ! filesink`
- **VP9**: `videoconvert ! vp9enc ! webmmux ! filesink`

## Performance

The application supports both host and device (CUDA) memory:

- **Host memory** (`--storage 0`): Simpler but requires memory copies
- **Device memory** (`--storage 1`): Zero-copy operation for better performance when using CUDA-enabled GStreamer elements

## Notes

- The pattern generator creates an animated gradient for testing purposes
- To use your own video source, replace `PatternGenOperator` with your video capture operator
- The application waits for encoding to complete before exiting to ensure proper file finalization
- EOS (End-Of-Stream) signal is sent automatically when recording completes

## See Also

- `holopattern_to_gst` - Similar application that can output to any GStreamer pipeline
- GStreamer documentation: https://gstreamer.freedesktop.org/documentation/


