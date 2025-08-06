# Emergent Source Operator

The `emergent_source` operator provides support for Emergent Vision Technologies cameras as video sources. This operator enables high-performance video streaming through Mellanox ConnectX SmartNIC using the Rivermax SDK.

## Overview

The EmergentSourceOp is designed to capture video streams from Emergent Vision Technologies cameras with high frame rates and resolution support. It leverages RDMA (Remote Direct Memory Access) capabilities for efficient data transfer and supports various camera parameters for optimal performance.

## Features

- **High Resolution Support**: Default resolution of 4200x2160 pixels
- **High Frame Rate**: Default frame rate of 240 FPS
- **RDMA Support**: Optional RDMA for enhanced performance
- **Configurable Parameters**: Adjustable width, height, framerate, exposure, and gain
- **Cross-Platform**: Supports x86_64 and aarch64 architectures
- **Python & C++ APIs**: Available in both programming languages

## Requirements

- **Holoscan SDK**: Minimum version 0.5.0 (tested with 0.5.0)
- **GXF Extensions**: Requires `emergent_source` extension version 1.0
- **Hardware**: Mellanox ConnectX SmartNIC for optimal performance
- **Camera**: Emergent Vision Technologies camera

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signal` | `gxf::Handle<gxf::Transmitter>` | - | Output signal channel |
| `width` | `uint32_t` | 4200 | Width of the video stream in pixels |
| `height` | `uint32_t` | 2160 | Height of the video stream in pixels |
| `framerate` | `uint32_t` | 240 | Frame rate of the video stream in FPS |
| `rdma` | `bool` | false | Enable RDMA for enhanced performance |
| `exposure` | `uint32_t` | 3072 | Exposure time setting |
| `gain` | `uint32_t` | 4095 | Analog gain setting |

## Usage Examples

### C++ Example

```cpp
#include "holoscan/operators/emergent_source/emergent_source.hpp"

// Create the operator
auto emergent_source = fragment.make_operator<holoscan::ops::EmergentSourceOp>(
    "emergent_source",
    holoscan::Arg{"width", 1920},
    holoscan::Arg{"height", 1080},
    holoscan::Arg{"framerate", 60},
    holoscan::Arg{"rdma", true},
    holoscan::Arg{"exposure", 2048},
    holoscan::Arg{"gain", 2048}
);
```

### Python Example

```python
from holoscan.operators import EmergentSourceOp

# Create the operator
emergent_source = EmergentSourceOp(
    fragment,
    width=1920,
    height=1080,
    framerate=60,
    rdma=True,
    exposure=2048,
    gain=2048,
    name="emergent_source"
)
```

## Building

The operator can be built as part of the Holohub build system:

```bash
# Build C++ version
cmake -B build -S .
cmake --build build

# Build Python bindings (if HOLOHUB_BUILD_PYTHON is enabled)
cmake -B build -S . -DHOLOHUB_BUILD_PYTHON=ON
cmake --build build
```

## Dependencies

- Holoscan Core SDK
- GXF Extensions (emergent_source extension)
- Rivermax SDK (for camera communication)
- Mellanox ConnectX SmartNIC drivers
