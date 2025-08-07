# Prohawk Video Processing Operator

## Description

The Prohawk Video Processing Operator is a Holoscan SDK operator that integrates Prohawk Technology Group's video restoration and enhancement capabilities. This operator provides real-time video processing with multiple filter options for various applications including medical imaging, surveillance, and broadcast video enhancement.

## Features

- **Real-time Video Processing**: Process video streams in real-time with minimal latency
- **Multiple Filter Presets**:
  - AFS (Adaptive Frame Stabilization) - Default filter with adaptive frame stabilization
  - Low Light Enhancement - Optimized for low-light conditions
  - Vascular Detail Enhancement - Specialized for medical imaging applications
  - Vaper Filter - Advanced video enhancement algorithm
- **Interactive Controls**: Real-time filter switching and display options
- **Side-by-Side Comparison**: View original and processed video side-by-side
- **OpenCV Integration**: Seamless integration with OpenCV for image processing
- **CUDA Support**: GPU-accelerated processing for improved performance

## Prerequisites

### System Requirements

- **Platform**: ARM64 (aarch64) architecture
- **Operating System**: Linux
- **GPU**: NVIDIA GPU with CUDA support (recommended)

### Software Dependencies

- **Holoscan SDK**
- **Prohawk Runtime**
- **OpenCV**

## Usage

### Basic Usage

The operator can be used in Holoscan applications to process video streams:

```python
import holoscan as holo
from holoscan.operators import ProhawkOp

# Create a fragment
fragment = holo.Fragment()

# Add the Prohawk operator
prohawk_op = ProhawkOp(fragment, name="prohawk_processor")

# Connect to your video source and output
# ... configure your pipeline
```

### Interactive Controls

When running the operator, you can use the following keyboard controls:

- **0**: Enable AFS (Adaptive Frame Stabilization) filter
- **1**: Enable Low Light Enhancement filter
- **2**: Enable Vascular Detail Enhancement filter
- **3**: Enable Vaper filter
- **d**: Disable restoration (pass-through mode)
- **v**: Enable side-by-side view
- **m**: Display menu items
- **q**: Quit the application

### Filter Descriptions

#### AFS (Adaptive Frame Stabilization) - Filter 0

- **Purpose**: Adaptive frame stabilization with noise reduction
- **Best For**: General video enhancement and stabilization
- **Parameters**:
  - Radius: 60x60 pixels
  - Threshold: 8
  - Accumulation: 128

#### Low Light Enhancement - Filter 1

- **Purpose**: Optimized for low-light video conditions
- **Best For**: Surveillance, night vision, low-light recording
- **Parameters**:
  - Radius: 60x60 pixels
  - Threshold: 8
  - Accumulation: 128

#### Vascular Detail Enhancement - Filter 2

- **Purpose**: Specialized for medical imaging applications
- **Best For**: Medical video processing, vascular imaging
- **Parameters**:
  - Radius: 24x24 pixels
  - Threshold: 16
  - Accumulation: 12

#### Vaper Filter - Filter 3

- **Purpose**: Advanced video enhancement algorithm
- **Best For**: High-quality video restoration
- **Parameters**:
  - Radius: 161x161 pixels
  - Threshold: 8
  - Accumulation: 0

## API Reference

### C++ API

#### ProhawkOp Class

```cpp
class ProhawkOp : public Operator {
public:
    // Constructor
    ProhawkOp();
    
    // Setup method
    void setup(OperatorSpec& spec) override;
    
    // Compute method
    void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override;
};
```

#### Input/Output Specifications

- **Input**: `gxf::Entity` containing video frame data
- **Output**: `gxf::Entity` containing processed video frame data

### Python API

```python
class ProhawkOp(Operator):
    """
    Operator class to use ProHawk filters.
    """
    
    def __init__(self, fragment, name="prohawk_video_processing"):
        """
        Initialize the Prohawk operator.
        
        Parameters
        ----------
        fragment : holoscan.Fragment
            The fragment to add this operator to
        name : str, optional
            The name of the operator
        """
```

## Configuration

### CMake Configuration Options

- `HOLOSCAN_SDK_PATH`: Path to Holoscan SDK installation
- `HOLOHUB_BUILD_PYTHON`: Enable Python bindings (default: ON)
- `OpenCV_DIR`: Path to OpenCV installation (if not in system path)

### Environment Variables

- `CUDA_VISIBLE_DEVICES`: Specify which GPU to use for processing
- `PROHAWK_SDK_PATH`: Override default Prohawk SDK path

## Examples

### Basic Video Processing Pipeline

```python
import holoscan as holo
from holoscan.operators import VideoStreamReplayerOp, ProhawkOp, HolovizOp

# Create fragment
fragment = holo.Fragment()

# Add operators
replayer = VideoStreamReplayerOp(fragment, name="replayer")
prohawk = ProhawkOp(fragment, name="prohawk")
holoviz = HolovizOp(fragment, name="holoviz")

# Connect operators
fragment.add_flow(replayer, prohawk)
fragment.add_flow(prohawk, holoviz)

# Run the application
app = holo.Application(fragment=fragment)
app.run()
```

### Custom Filter Configuration

The operator supports real-time filter switching through interactive controls. The filter parameters are automatically configured based on the selected preset, but you can modify the source code to customize these parameters for your specific use case.

## Integration

### With Holoscan Applications

The Prohawk operator integrates seamlessly with other Holoscan operators:

- **Input Sources**: VideoStreamReplayerOp, VideoCaptureOp, etc.
- **Output Sinks**: HolovizOp, VideoWriterOp, etc.
- **Processing**: Can be chained with other video processing operators

### With Existing Video Pipelines

The operator can be easily integrated into existing video processing pipelines by:

1. Adding the operator to your Holoscan fragment
2. Connecting it to your video source
3. Connecting the output to your desired sink
4. Configuring the appropriate filter for your use case

## Development

### Building for Development

```bash
# Debug build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# Release build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Testing

The operator includes interactive testing capabilities through the built-in display window. You can test different filters and configurations in real-time.

### Extending the Operator

To add new filters or modify existing ones:

1. Edit `prohawkop.cpp` to add new filter cases
2. Update the interactive controls in the compute method
3. Rebuild the operator
4. Test with your video stream

## Dependencies

### Required Libraries

- **Holoscan Core**: Core Holoscan SDK functionality
- **Prohawk PTGDE**: Prohawk video processing library
- **OpenCV**: Computer vision library for image processing
- **CUDA**: GPU acceleration (optional but recommended)

### Build Dependencies

- **CMake**: Build system
- **pybind11**: Python bindings (if building Python support)
- **C++ Compiler**: GCC or Clang with C++17 support
