# SlangShaderOp

The `SlangShaderOp` is a Holoscan operator that enables execution of [Slang](https://github.com/shader-slang/slang) shaders within Holoscan applications. It provides a bridge between the Slang shading language and Holoscan's data processing pipeline, allowing developers to write GPU-accelerated compute shaders that can process data flowing through Holoscan applications.

## Overview

The SlangShaderOp compiles Slang shader source code into CUDA kernels and executes them on GPU devices. It supports dynamic parameter binding, automatic input/output port generation, and seamless integration with Holoscan's data flow model.

## Features

- **Slang Shader Compilation**: Compiles Slang shader source code to CUDA PTX
- **Dynamic Port Generation**: Automatically creates input/output ports based on shader attributes
- **Parameter Binding**: Supports scalar parameter types (bool, int, float, etc.)
- **Structured Buffer Support**: Handles input/output structured buffers
- **Grid Size Configuration**: Configurable compute grid dimensions
- **CUDA Stream Integration**: Integrates with Holoscan's CUDA stream management
- **Python and C++ APIs**: Available in both Python and C++ interfaces

## Requirements

- Holoscan SDK 3.3.0 or later
- CUDA-compatible GPU
- Slang compiler (automatically fetched during build)
- Supported platforms: x86_64, aarch64

## Installation

The SlangShaderOp is included as part of the HoloHub operators. It will be automatically built when you build the HoloHub project.

## Usage

### Basic Usage

The operator can be configured with either a shader source string or a shader source file:

```python
from holoscan.operators import SlangShaderOp

# Using shader source string
shader_source = """
import holoscan

[holoscan_input("input_data")]
StructuredBuffer<float> input_buffer;

[holoscan_output("output_data")]
RWStructuredBuffer<float> output_buffer;

[holoscan_parameter("scale_factor")]
float scale;

[numthreads(256, 1, 1)]
[holoscan_grid_size_of("input_data")]
void main(uint3 tid : SV_DispatchThreadID) {
    output_buffer[tid.x] = input_buffer[tid.x] * scale;
}
"""

op = SlangShaderOp(
    fragment=app,
    shader_source=shader_source,
    name="my_shader"
)
```

Note that the data sent to the input of `SlangShaderOp` must be a data buffer (currently `holoscan::Tensor` and `nvidia::gxf::VideoBuffer` types are supported). For Python, any array-like objects implementing the `__dlpack__`, `__array_interface__` or `__cuda_array_interface__` are also supported.

### Shader Attributes

The SlangShaderOp uses special attributes to define how shader parameters interact with Holoscan:

#### Input/Output Attributes

- `[holoscan_input("port_name")]`: Marks a structured buffer as an input port
- `[holoscan_output("port_name")]`: Marks a structured buffer as an output port
- `[holoscan_alloc_size_of("port_name")]`: Specifies allocation size based on input port

#### Parameter Attributes

- `[holoscan_parameter("param_name")]`: Marks a scalar as a configurable parameter
- `[holoscan_size_of("port_name")]`: Provides size information from input port

#### Compute Grid Attributes

- `[holoscan_grid_size_of("port_name")]`: Sets grid size based on input tensor dimensions
- `[holoscan_grid_size(x, y, z)]`: Sets fixed grid dimensions

### Supported Data Types

#### Scalar Parameters
- `bool`, `int8`, `uint8`, `int16`, `uint16`
- `int32`, `uint32`, `int64`, `uint64`
- `float32`, `float64`

#### Buffer Types
- `StructuredBuffer<T>`: Input buffers
- `RWStructuredBuffer<T>`: Output buffers

### Example: Image Processing Shader

```slang
import holoscan

// Simple image processing shader
[holoscan_input("input_image")]
StructuredBuffer<float4> input_image;

[holoscan_output("output_image")]
RWStructuredBuffer<float4> output_image;

[holoscan_parameter("brightness")]
float brightness;

[holoscan_size_of("input_image")]
int3 image_size;

[numthreads(16, 16, 1)]
[holoscan_grid_size_of("input_image")]
void main(uint3 tid : SV_DispatchThreadID) {
    uint index = tid.y * image_size.x + tid.x;
    float4 pixel = input_image[index];

    // Apply brightness adjustment
    output_image[index] = pixel * brightness;
}
```

### C++ Usage

```cpp
#include <slang_shader_op.hpp>

// Create the operator with shader source from a string
std::string shader_source_string = R"
include holoscan;
...
";
auto shader_op_str  = make_operator<holoscan::ops::SlangShaderOp>("Slang",
    Arg("shader_source", shader_source_string));

// Or create the operator with a Slang shader source file
auto shader_op_file = make_operator<holoscan::ops::SlangShaderOp>("Slang",
    Arg("shader_source_file", "my_shader.slang"));
```

## Architecture

The SlangShaderOp consists of several key components:

### Core Classes

- **`SlangShaderOp`**: Main operator class that orchestrates shader execution
- **`SlangShader`**: Manages shader compilation and CUDA kernel retrieval
- **`Command`**: Command pattern implementation for various operations
- **`CommandWorkspace`**: Centralized workspace for command execution

### Execution Flow

1. **Setup Phase**:
   - Compiles Slang shader source to PTX
   - Analyzes shader reflection to generate ports and parameters
   - Creates command sequences for pre-launch, launch, and post-launch operations

2. **Compute Phase**:
   - Executes pre-launch commands (input handling, parameter setup)
   - Launches CUDA kernels with configured grid/block dimensions
   - Executes post-launch commands (output handling)

### Command Types

- **`CommandInput`**: Handles input port data reception
- **`CommandOutput`**: Handles output port data emission
- **`CommandParameter`**: Manages scalar parameter binding
- **`CommandSizeOf`**: Provides size information to shaders
- **`CommandAllocSizeOf`**: Handles dynamic allocation sizing
- **`CommandLaunch`**: Executes CUDA kernels

## Error Handling

The operator provides comprehensive error handling:

- **Compilation Errors**: Detailed Slang compilation diagnostics
- **Runtime Errors**: CUDA execution error reporting
- **Parameter Validation**: Type checking and attribute validation
- **Resource Management**: Automatic cleanup of CUDA resources

## Performance Considerations

- **Kernel Compilation**: Shaders are compiled once during setup
- **Memory Management**: Uses Holoscan's allocator system for buffer management
- **Stream Management**: Integrates with Holoscan's CUDA stream pool
- **Parameter Updates**: Efficient parameter updates without recompilation

## Limitations

- Only compute shaders are supported (no vertex/fragment shaders)
- Structured buffers are the only supported buffer type
- Grid size must be specified via attributes
- Shader compilation happens at operator setup time

## Troubleshooting

### Common Issues

1. **Compilation Errors**: Check shader syntax and ensure all attributes are properly defined
2. **Parameter Type Mismatches**: Verify parameter types match between shader and operator
3. **Grid Size Issues**: Ensure grid size attributes are correctly specified
4. **Memory Errors**: Verify buffer sizes and allocation parameters

### Debugging

Enable debug logging to see detailed execution information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

The SlangShaderOp is part of the HoloHub project. Contributions are welcome through the standard HoloHub contribution process.

## License

This operator is licensed under the Apache License 2.0, same as the HoloHub project.
