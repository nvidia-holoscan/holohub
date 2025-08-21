# YUAN QCAP Source Operator

The `QCAPSourceOp` operator provides video stream capture from YUAN High-Tech capture cards, supporting various video formats and configurations for professional video acquisition.

## Overview

This operator interfaces with YUAN High-Tech capture cards to acquire video streams with configurable resolution, frame rate, and pixel formats. It supports both C++ and Python implementations and provides flexible parameter configuration for different capture scenarios.

## Features

- **YUAN Capture Card Support**: Direct integration with YUAN High-Tech capture hardware
- **Configurable Resolution**: Support for various video resolutions up to 4K
- **Flexible Frame Rates**: Adjustable frame rates for different applications
- **Multiple Pixel Formats**: Support for RGB, RGBA, and other color formats
- **RDMA Support**: Optional RDMA (Remote Direct Memory Access) for improved performance
- **Multi-Platform**: Available for both x86_64 and aarch64 architectures

## Usage

### Python Usage

```python
from holoscan.operators import QCAPSourceOp

# Create operator with default settings
qcap_op = QCAPSourceOp(
    fragment=fragment,
    device="SC0710 PCI",
    channel=0,
    width=3840,
    height=2160,
    framerate=60,
    rdma=True,
    pixel_format="bgr24",
    input_type="auto"
)
```

### C++ Usage

```cpp
#include "qcap_source.hpp"

auto qcap_op = std::make_shared<holoscan::ops::QCAPSourceOp>(
    Arg{"device", "SC0710 PCI"},
    Arg{"channel", 0},
    Arg{"width", 3840},
    Arg{"height", 2160},
    Arg{"framerate", 60},
    Arg{"rdma", true},
    Arg{"pixel_format", "bgr24"},
    Arg{"input_type", "auto"}
);
```

Please refer to the following Holoscan reference applications for usage of this operator:

- [Endoscopy Tool Tracking](../../applications/endoscopy_tool_tracking/python/endoscopy_tool_tracking.py)
- [H.264 Endoscopy Tool Tracking](../../applications/h264/h264_endoscopy_tool_tracking/python/h264_endoscopy_tool_tracking.py)

## Parameters

### Device Configuration

- **`device`** (string, default: "SC0710 PCI"): Device specifier for the capture card
- **`channel`** (uint32_t, default: 0): Channel number to use for capture

### Video Settings

- **`width`** (uint32_t, default: 3840): Width of the video stream in pixels
- **`height`** (uint32_t, default: 2160): Height of the video stream in pixels
- **`framerate`** (uint32_t, default: 60): Frame rate of the video stream in fps

### Performance Options

- **`rdma`** (bool, default: false): Enable RDMA for improved memory transfer performance
- **`pixel_format`** (string, default: "bgr24"): Pixel format of the video stream
- **`input_type`** (string, default: "auto"): Input type configuration
- **`mst_mode`** (uint32_t, default: 0): MST (Multi-Stream Transport) mode setting
- **`sdi12g_mode`** (uint32_t, default: 0): SDI 12G mode configuration

## Input/Output

### Output

- **`video_buffer_output`**: Video buffer containing the captured frame data
  - Format: Video buffer with specified resolution and pixel format
  - Rate: Matches the configured frame rate
  - Type: GXF Entity with video buffer data

## Supported Configurations

### Video Resolutions

- 4K: 3840x2160 (default)
- 2K: 2048x1080
- 1080p: 1920x1080
- 720p: 1280x720
- Custom resolutions supported

### Pixel Formats

- `bgr24`: 24-bit BGR format (default)
- `rgb24`: 24-bit RGB format
- `rgba32`: 32-bit RGBA format
- Additional formats based on hardware support

### Frame Rates

- 60 fps (default)
- 30 fps
- 25 fps
- 24 fps
- Custom frame rates supported

## Hardware Requirements

- YUAN High-Tech capture card (e.g., SC0710 PCI)
- Compatible PCIe slot
- Sufficient bandwidth for video stream
- GPU memory for RDMA operations (if enabled)

## Integration

The operator is designed to work within Holoscan pipelines and can be connected to:

- Video processing operators
- Encoding/compression operators
- Display/visualization operators
- Recording/storage operators
- AI inference pipelines
