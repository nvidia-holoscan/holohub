# Tensor to File Operator

A Holoscan operator that writes tensor data to a file. This operator is designed to stream input tensor data to a file, with the file opened at initialization and tensor binary data appended in the same order in which messages are received.

## Overview

The Tensor to File operator is a data persistence component that takes tensor data as input and writes it directly to a file. It's particularly useful for:

- Saving encoded video frames to elementary stream files (H.264/H.265)
- Persisting tensor data for later analysis or processing
- Creating data dumps for debugging purposes
- Building data pipelines that require file output

## Features

- **Binary Data Writing**: Writes tensor data as binary to maintain data integrity
- **Performance Optimized**: Configurable buffer size for optimal I/O performance
- **File Validation**: Validates output file paths and creates directories as needed
- **Progress Tracking**: Optional verbose mode with performance statistics
- **Cross-Platform**: Supports both x86_64 and aarch64 architectures
- **Multi-Language**: Available in both C++ and Python interfaces

## Usage

### C++ Interface

```cpp
#include "holoscan/operators/tensor_to_file/tensor_to_file.hpp"

// Create operator instance
auto tensor_to_file = std::make_unique<holoscan::ops::TensorToFileOp>();

// Configure parameters
tensor_to_file->add_arg<std::string>("tensor_name", "input_tensor");
tensor_to_file->add_arg<std::string>("output_file", "/path/to/output.h264");
tensor_to_file->add_arg<bool>("verbose", true);
tensor_to_file->add_arg<size_t>("buffer_size", 1024 * 1024); // 1MB buffer
```

### Python Interface

```python
import holoscan as hs
from holoscan.operators import TensorToFileOp

# Create operator
tensor_to_file = TensorToFileOp(
    tensor_name="input_tensor",
    output_file="/path/to/output.h264",
    verbose=True,
    buffer_size=1024 * 1024,  # 1MB buffer
    name="tensor_writer"
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensor_name` | string | "" | Name of the tensor to write to the file |
| `output_file` | string | "" | Output file path for the data |
| `allocator` | Allocator | - | Allocator for output buffers |
| `verbose` | bool | false | Print detailed writer information including frame count and bytes written |
| `buffer_size` | size_t | 1MB | Buffer size for file I/O operations in bytes |

## Input/Output

### Input
- **Entity**: Contains tensor data to be written to file
- **Tensor**: Binary data (typically uint8) representing the content to be saved

### Output
- **File**: Binary file containing the tensor data in the order received

## File Format Support

The operator supports various file extensions for video data:
- `.h264`, `.264` - H.264 elementary stream
- `.h265`, `.265`, `.hevc` - H.265/HEVC elementary stream  
- `.mp4` - MP4 container format

## Configuration Examples

### Basic Usage
```python
# Simple tensor to file writing
tensor_to_file = TensorToFileOp(
    tensor_name="encoded_frame",
    output_file="output.h264"
)
```

### Verbose Mode with Custom Buffer
```python
# With detailed logging and custom buffer size
tensor_to_file = TensorToFileOp(
    tensor_name="encoded_frame",
    output_file="output.h265",
    verbose=True,
    buffer_size=2 * 1024 * 1024  # 2MB buffer
)
```

### Integration in Pipeline
```python
# As part of a video processing pipeline
pipeline = holoscan.Pipeline()

# Add operators to pipeline
encoder = VideoEncoderOp(...)
tensor_to_file = TensorToFileOp(
    tensor_name="encoded_frame",
    output_file="processed_video.h264",
    verbose=True
)

# Connect operators
pipeline.add_operator(encoder)
pipeline.add_operator(tensor_to_file)
pipeline.add_flow(encoder, tensor_to_file)
```

## Error Handling

The operator includes comprehensive error handling for:
- Invalid file paths
- File system permission issues
- Insufficient disk space
- Corrupted tensor data
- I/O operation failures

## Dependencies

- **Holoscan Core**: Core framework functionality
- **GXF**: Graph execution framework
- **Standard C++ Library**: File I/O and system operations
- **Filesystem Library**: Path and directory operations

## Platform Support

- **Architectures**: x86_64, aarch64
- **Operating Systems**: Linux
- **Holoscan SDK**: 3.3.0 - 3.4.0 (tested versions)

## Related Operators

- **VideoEncoderOp**: Produces encoded video frames that can be written by this operator
- **TensorToFileOp**: This operator for writing tensor data to files
- **FileToTensorOp**: Complementary operator for reading tensor data from files 