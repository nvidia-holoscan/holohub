# Gamma Correction Operator

The `GammaCorrectionOp` is a Holoscan operator that applies gamma correction to images using GPU-accelerated compute shaders. It provides efficient gamma correction processing for both single-channel and multi-channel images, with support for various data types and automatic normalization.

## Overview

The GammaCorrectionOp extends the SlangShaderOp to provide specialized gamma correction functionality. It automatically handles data type conversion, normalization for integer types, and multi-component processing while maintaining high performance through GPU acceleration.

## Features

- **Automatic Data Type Handling**: Supports various data types (uint8, uint16, float32, etc.) with automatic normalization
- **Multi-Component Support**: Processes images with 1-4 components (grayscale, RGB, RGBA)
- **Configurable Gamma Value**: Adjustable gamma correction factor (default: 2.2)
- **GPU Acceleration**: Leverages CUDA compute shaders for high-performance processing
- **Python and C++ APIs**: Available in both Python and C++ interfaces

## Requirements

- Holoscan SDK 3.4.0 or later
- CUDA-compatible GPU
- Supported platforms: x86_64, aarch64

## Installation

The GammaCorrectionOp is included as part of the HoloHub operators. It will be automatically built when you build the HoloHub project.

## Usage

### Basic Usage

The operator can be configured with data type and component count parameters:

```python
from holoscan.operators import GammaCorrectionOp

# Basic gamma correction for uint8 grayscale image
op = GammaCorrectionOp(
    fragment=app,
    data_type="uint8_t",
    component_count=1,
    gamma=2.2,
    name="gamma_correction"
)
```

### C++ Usage

```cpp
#include <gamma_correction/gamma_correction.hpp>

// Create the operator with uint8_t data type
auto gamma_op = make_operator<holoscan::ops::GammaCorrectionOp>("gamma_correction",
    Arg("data_type", "uint8_t"),
    Arg("component_count", 3),  // RGB
    Arg("gamma", 2.2f));
```

### Parameters

- **`data_type`** (required): The data type of the input buffer
  - Supported types: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `float`, `double`
- **`component_count`** (optional): Number of components in the input buffer
  - Default: 1 (grayscale)
  - Supported: 1-4 components
- **`gamma`** (optional): Gamma correction factor
  - Default: 2.2
  - Range: Any positive float value

### Supported Data Types

#### Integer Types
- `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`: Automatically normalized to [0,1] range before processing
- Gamma correction applied in normalized space
- Results scaled back to original range

#### Floating Point Types
- `float`, `double`: Processed directly without normalization
- Assumes input values are already in [0,1] range

### Example: RGB Image Processing

```python
# Process RGB image with custom gamma
op = GammaCorrectionOp(
    fragment=app,
    data_type="uint8_t",
    component_count=3,
    gamma=1.8,
    name="rgb_gamma_correction"
)
```

### Example: Float Image Processing

```python
# Process float image (assumes values in [0,1] range)
op = GammaCorrectionOp(
    fragment=app,
    data_type="float",
    component_count=1,
    gamma=2.2,
    name="float_gamma_correction"
)
```

## Testing

The GammaCorrectionOp includes comprehensive testing to ensure reliability and correctness:

### Running Tests

```bash
./holohub test gamma_correction
```

## Contributing

The GammaCorrectionOp is part of the HoloHub project. Contributions are welcome through the standard HoloHub contribution process.

## License

This operator is licensed under the Apache License 2.0, same as the HoloHub project.

