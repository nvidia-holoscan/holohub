# Holoscan Pattern to GStreamer (holopattern-to-gst)

This application demonstrates how to push pattern data from a Holoscan operator into a GStreamer pipeline using `GstSrcResource`.

## Overview

The application creates a Holoscan pipeline that:
1. Generates animated test patterns (gradient, checkerboard, or color bars)
2. Pushes frames into a GStreamer pipeline via `GstSrcResource`
3. Processes/outputs the video using standard GStreamer elements

This is the complementary example to `gst-to-holoviz` - instead of pulling data from GStreamer into Holoscan, this pushes data from Holoscan into GStreamer.

## Building

```bash
cd /workspace/holohub/applications/gstreamer/holopattern_to_gst
mkdir build && cd build
cmake ..
make
```

## Usage

### Basic Examples

**Display animated gradient (default):**
```bash
./holopattern-to-gst
```

**Display animated checkerboard:**
```bash
./holopattern-to-gst --pattern 1
```

**Display color bars:**
```bash
./holopattern-to-gst --pattern 2
```

**Custom resolution:**
```bash
./holopattern-to-gst --width 1280 --height 720
```

**Save video to file (300 frames):**
```bash
./holopattern-to-gst --count 300 --pipeline "videoconvert name=first ! x264enc ! mp4mux ! filesink location=output.mp4"
```

**Stream over network (RTP):**
```bash
./holopattern-to-gst --pipeline "videoconvert name=first ! x264enc ! rtph264pay ! udpsink host=127.0.0.1 port=5000"
```

**Apply GStreamer effects:**
```bash
./holopattern-to-gst --pipeline "videoconvert name=first ! videoflip method=horizontal-flip ! autovideosink"
```

### Command Line Options

- `-c, --count <number>` - Number of frames to generate (default: unlimited)
- `-w, --width <pixels>` - Frame width (default: 1920)
- `-h, --height <pixels>` - Frame height (default: 1080)
- `--pattern <type>` - Pattern type: 0=gradient, 1=checkerboard, 2=color bars (default: 0)
- `-p, --pipeline <desc>` - GStreamer pipeline description
- `--caps <caps_string>` - GStreamer capabilities for the source (default: `video/x-raw,format=RGBA`)
- `--help` - Show help message

### Important Requirements

**Your GStreamer pipeline MUST name the first element as 'first'**

This is required for the application to properly link the Holoscan source to your pipeline.

**Correct:**
```bash
--pipeline "videoconvert name=first ! autovideosink"
```

**Incorrect:**
```bash
--pipeline "videoconvert ! autovideosink"  # Missing name=first
```

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Holoscan Application                      │
│                                                              │
│                     ┌─────────────────────────────┐         │
│                     │   GstSrcOperator            │         │
│                     │  (Source Operator)          │         │
│                     │  • Generates test pattern   │         │
│                     │  • Creates GstBuffer        │         │
│                     │  • Pushes to GstSrcResource │         │
│                     └──────────┬──────────────────┘         │
│                                │                             │
│                     ┌──────────▼──────────────────┐         │
│                     │   GstSrcResource            │         │
│                     │  (holoscansrc element)      │         │
│                     └──────────┬──────────────────┘         │
└────────────────────────────────┼──────────────────────────── ┘
                                 │
                                 ▼
                  ┌────────────────────────────────────┐
                  │      GStreamer Pipeline            │
                  │                                    │
                  │  videoconvert → x264enc → filesink │
                  │       OR                           │
                  │  effects → filters → output        │
                  └────────────────────────────────────┘
```

### Data Flow

1. **GstSrcOperator** generates animated test patterns (RGBA format)
2. **GstSrcOperator** creates GStreamer buffers with timestamp information
3. **GstSrcResource** (the `holoscansrc` element) provides buffers to the GStreamer pipeline
4. **GStreamer Pipeline** processes and outputs the video data

### Pattern Types

- **Gradient (0)**: Animated colorful sine wave patterns
- **Checkerboard (1)**: Animated black and white checkerboard with varying square sizes
- **Color Bars (2)**: SMPTE-style color bars (white, yellow, cyan, green, magenta, red, blue)

## Use Cases

### Video Processing
Apply GStreamer's rich ecosystem of video filters and effects:
```bash
./holopattern-to-gst --pipeline "videoconvert name=first ! videobalance saturation=0 ! autovideosink"
```

### Video Encoding
Encode Holoscan output to various formats:
```bash
# H.264
./holopattern-to-gst --pipeline "videoconvert name=first ! x264enc ! mp4mux ! filesink location=output.mp4"

# VP9 (WebM)
./holopattern-to-gst --pipeline "videoconvert name=first ! vp9enc ! webmmux ! filesink location=output.webm"
```

### Streaming
Stream Holoscan output over the network:
```bash
# RTSP server (requires gst-rtsp-server)
# RTP/UDP
./holopattern-to-gst --pipeline "videoconvert name=first ! x264enc tune=zerolatency ! rtph264pay ! udpsink host=127.0.0.1 port=5000"
```

### Multi-output
Use GStreamer's `tee` element to send output to multiple destinations:
```bash
./holopattern-to-gst --pipeline "videoconvert name=first ! tee name=t ! queue ! autovideosink t. ! queue ! x264enc ! mp4mux ! filesink location=output.mp4"
```

## Troubleshooting

### "Could not find element named 'first'"
Make sure your pipeline includes `name=first` on the first element after the source.

### Pipeline won't start
Check that all GStreamer elements in your pipeline are installed:
```bash
gst-inspect-1.0 <element-name>
```

### Buffer underruns / overruns
Adjust the queue_limit parameter in GstSrcResource if needed.

## See Also

- `gst-to-holoviz` - The complementary example that pulls data from GStreamer into Holoscan
- GStreamer documentation: https://gstreamer.freedesktop.org/documentation/
- Holoscan SDK documentation: https://docs.nvidia.com/holoscan/

