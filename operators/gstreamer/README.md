# GStreamer Bridge Components

This directory contains components that provide a bridge between Holoscan and GStreamer, enabling integration with the vast ecosystem of GStreamer plugins for video encoding, streaming, and multimedia processing.

## Overview

The GStreamer bridge provides:
- **Operators** for integrating GStreamer functionality into Holoscan pipelines
- **Resources** for managing GStreamer elements and memory allocation within the Holoscan resource framework
- **Low-level bridge objects** that can be used in both Holoscan and non-Holoscan applications
- **C++ RAII wrappers** for safe and convenient GStreamer API usage

These components allow you to:
- Integrate GStreamer's extensive plugin ecosystem into Holoscan applications
- Record and stream video using GStreamer encoders and protocols
- Support both host (CPU) and device (GPU) memory for efficient processing
- Build complex media processing pipelines combining Holoscan and GStreamer
- Use GStreamer functionality in custom applications outside the Holoscan framework

## Components

### Operators

#### GstVideoRecorderOp

Records incoming Holoscan tensors to video files using GStreamer encoding pipelines.

**Key Features:**
- Multiple codec support: H.264 (nvh264, x264), H.265 (nvh265, x265)
- Configurable encoder properties (bitrate, preset, quality, etc.)
- Support for both host and device (CUDA) memory
- MP4 and MKV container output

**Input:**
- Port `"input"`: Video frames as Holoscan tensors (RGB888/RGBA8888 format)

**Parameters:**
- `filename` (string): Output video file path (default: "output.mp4")
- `encoder` (string): Encoder to use - "nvh264", "nvh265", "x264", "x265" (default: "nvh264")
- `format` (string): Pixel format - "RGBA", "RGB", "BGRA", "BGR", "GRAY8" (default: "RGBA")
- `framerate` (string): Target framerate as fraction "num/den" or decimal (default: "30/1")
- `properties` (map<string,string>): Encoder-specific properties (e.g., bitrate, preset, gop-size)
- `max-buffers` (size_t): Maximum number of buffers to queue, 0 = unlimited (default: 10)
- `block` (bool): Block when queue is full vs. drop/timeout (default: true)

### Resources

Resources provide Holoscan-managed wrappers for GStreamer elements and allocators, integrating them into the Holoscan resource lifecycle and memory management system.

**Note:** Resource implementations will be added in future updates.

### Low-Level Bridge Objects

Low-level bridge objects implement core GStreamer integration functionality in a framework-agnostic way. These can be used:
- Within Holoscan applications (via operators and resources)
- In standalone C++ applications without Holoscan dependencies
- In custom integration scenarios

These objects handle the detailed work of data transfer, format conversion, and GStreamer API interaction.

#### GstSrcBridge

`GstSrcBridge` is a framework-agnostic class that bridges tensor/video data into GStreamer pipelines via the `appsrc` element. It can be used independently of Holoscan in any C++ application that needs to feed data into GStreamer.

**Key capabilities:**
- Push video frames from host or device (CUDA) memory into GStreamer pipelines
- Configurable caps (capabilities) for proper format specification
- Buffer queuing with configurable size limits and blocking behavior
- Support for various pixel formats (RGBA, RGB, NV12, I420, etc.)
- Automatic timestamp generation based on framerate
- End-of-stream (EOS) signaling
- Zero-copy operation when wrapping tensor memory

### C++ Wrapper Classes

The bridge includes a set of C++ RAII wrapper classes for GStreamer objects (located in the `gst/` subdirectory).

**Key wrapper classes include (subset):**
- **`gst::Element`**: Wrapper for GstElement (pipeline elements)
- **`gst::Pipeline`**: Wrapper for GstPipeline (top-level pipelines)
- **`gst::Bus`**: Wrapper for GstBus (message handling)
- **`gst::Caps`**: Wrapper for GstCaps (media format capabilities)
- **`gst::Buffer`**: Wrapper for GstBuffer (data buffers)
- **`gst::Message`**: Wrapper for GstMessage (bus messages)
- **`gst::Allocator`**: Wrapper for GstAllocator (memory allocation)

These wrappers provide:
- Automatic reference counting and cleanup
- Type-safe property setting with compile-time string conversion
- Convenient API for common GStreamer operations
- Exception-based error handling

**Note:** These wrapper classes can be used independently in non-Holoscan applications for safer GStreamer programming.

## Usage Example

For a complete working example demonstrating how to use `GstVideoRecorderOp` in a Holoscan application, see the [`gst_video_recorder`](../../applications/gstreamer/gst_video_recorder/) application. It shows:
- Integration with `V4L2VideoCaptureOp` for camera input
- Integration with pattern generators for synthetic video
- Proper use of `FormatConverterOp` for format handling
- Configuration of encoder properties
- YAML-based configuration

## Requirements

**System Packages:**

A Dockerfile with all dependencies pre-installed is provided at [`applications/gstreamer/gst_video_recorder/Dockerfile`](../../applications/gstreamer/gst_video_recorder/Dockerfile) for containerized builds.

For local development, install all required dependencies using the provided script:
```bash
./applications/gstreamer/gst_video_recorder/install_deps.sh
```

**Optional (for CUDA support):**
- `gstreamer1.0-cuda` (requires GStreamer 1.24+)

**Holoscan SDK:**
- Minimum version: 3.8.0

## Advanced Configuration

### Encoder Properties

Encoder properties are passed directly to the underlying GStreamer encoder element. Common properties include:

**NVIDIA H.264/H.265 (nvh264enc/nvh265enc):**
- `bitrate`: Target bitrate in kbps (e.g., "5000")
- `preset`: Encoding preset 0-3 (0=slowest/best quality, 3=fastest/lower quality)
- `gop-size`: GOP (Group of Pictures) size in frames

**x264/x265 (software encoders):**
- `bitrate`: Target bitrate in kbps
- `speed-preset`: Encoding speed preset (e.g., "ultrafast", "medium", "slow")
- `tune`: Tuning preset (e.g., "zerolatency", "film")

For a complete list of available properties, consult the GStreamer plugin documentation or use `gst-inspect-1.0 <encoder>` (e.g., `gst-inspect-1.0 nvh264enc`).

### CUDA Memory Support

When `HOLOSCAN_GSTREAMER_CUDA_SUPPORT` is enabled (requires GStreamer 1.24+), the operator automatically detects and uses CUDA memory for zero-copy data transfer from GPU to encoder.

## References

- [GStreamer Documentation](https://gstreamer.freedesktop.org/documentation/)
- [Holoscan SDK Documentation](https://docs.nvidia.com/holoscan/sdk-user-guide/)
- [Application Example: gst_video_recorder](../../applications/gstreamer/gst_video_recorder/)