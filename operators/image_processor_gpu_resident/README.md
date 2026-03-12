# Hololink Image Processor GPU-Resident Operator

This operator performs image processing operations on Bayer image data using GPU-resident processing. It includes optical black correction, histogram calculation, and automatic white balance.

## Overview

The `ImageProcessorGpuResidentOp` is a GPU-resident Holoscan operator that:

- Applies optical black level correction
- Calculates per-channel histograms
- Computes white balance gains automatically
- Applies white balance correction

All operations are performed entirely on GPU memory for low-latency processing.

## Requirements

- Holoscan SDK 3.5.0 or later
- Hololink library installed and available
- CUDA-capable GPU

## Usage

### C++

```cpp
#include <image_processor_gpu_resident/image_processor_gpu_resident.hpp>

// In your application setup:
auto image_processor = make_operator<hololink::operators::ImageProcessorGpuResidentOp>(
    "image_processor",
    holoscan::Arg("width", width),
    holoscan::Arg("height", height),
    holoscan::Arg("pixel_format", static_cast<int>(pixel_format)),
    holoscan::Arg("bayer_format", static_cast<int>(bayer_format)),
    holoscan::Arg("optical_black", optical_black_value)
);

// Add to your pipeline after CSI-to-Bayer conversion
add_flow(csi_to_bayer, image_processor, {{"out", "in"}});
add_flow(image_processor, next_operator, {{"out", "in"}});
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `width` | int32_t | Image width in pixels |
| `height` | int32_t | Image height in pixels |
| `pixel_format` | int | Pixel format (RAW_8=0, RAW_10=1, RAW_12=2) |
| `bayer_format` | int | Bayer pattern (RGGB=0, GBRG=1, BGGR=2, GRBG=3) |
| `optical_black` | int32_t | Optical black value to subtract (default: 0) |

## Processing Pipeline

1. **Optical Black Correction**: Subtracts the optical black value and rescales
2. **Histogram Calculation**: Computes per-channel (R, G, B) histograms
3. **White Balance Gain Calculation**: Computes gains based on gray-world assumption
4. **Apply White Balance**: Applies the computed gains to each pixel

## Build

This operator is built as part of the HoloHub build system:

```bash
./run build image_processor_gpu_resident
```

## Dependencies

- `hololink::core` - Core hololink library
- `hololink::common` - Common hololink utilities (CUDA helpers)
- `holoscan::core` - Holoscan SDK
- `fmt` - Formatting library
- `CUDA` - CUDA driver API
