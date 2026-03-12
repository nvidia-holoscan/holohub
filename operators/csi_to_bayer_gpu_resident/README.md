# Hololink CSI to Bayer GPU-Resident Operator

This operator converts CSI (Camera Serial Interface) packed data to Bayer format using GPU-resident processing. It supports RAW8, RAW10, and RAW12 pixel formats.

## Overview

The `CsiToBayerGpuResidentOp` is a GPU-resident Holoscan operator that:
- Receives packed CSI data from camera sensors
- Unpacks the data to 16-bit Bayer format
- Operates entirely on GPU memory for low-latency processing

## Requirements

- Holoscan SDK 3.5.0 or later
- Hololink library installed and available
- CUDA-capable GPU

## Usage

### C++

```cpp
#include <csi_to_bayer_gpu_resident/csi_to_bayer_gpu_resident.hpp>

// In your application setup:
auto csi_to_bayer = make_operator<hololink::operators::CsiToBayerGpuResidentOp>("csi_to_bayer");

// Configure the operator before initialization
csi_to_bayer->configure(
    start_byte,       // Offset to first pixel data
    bytes_per_line,   // Bytes per line in CSI data
    pixel_width,      // Image width in pixels
    pixel_height,     // Image height in pixels
    pixel_format,     // hololink::csi::PixelFormat::RAW_8/RAW_10/RAW_12
    trailing_bytes    // Trailing bytes after image data
);

// Add to your pipeline
add_flow(receiver, csi_to_bayer, {{"out", "in"}});
add_flow(csi_to_bayer, next_operator, {{"out", "in"}});
```

## Supported Pixel Formats

- `RAW_8`: 8-bit raw data, unpacked to 16-bit (upper 8 bits)
- `RAW_10`: 10-bit raw data (4 pixels packed in 5 bytes), unpacked to 16-bit
- `RAW_12`: 12-bit raw data (2 pixels packed in 3 bytes), unpacked to 16-bit

## Build

This operator is built as part of the HoloHub build system:

```bash
./run build csi_to_bayer_gpu_resident
```

## Dependencies

- `hololink::core` - Core hololink library
- `hololink::common` - Common hololink utilities (CUDA helpers)
- `holoscan::core` - Holoscan SDK
- `fmt` - Formatting library
- `CUDA` - CUDA driver API
