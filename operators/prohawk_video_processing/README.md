# Prohawk Video Processing Operator

## Overview

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

## Usage

Please refer to [Prohawk Video Processing](../../applications/prohawk_video_replayer/README.md) for requirements and usage examples.

### Basic Usage

The operator can be used in Holoscan applications to process video streams:

```python
from holoscan.operators import ProhawkOp

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
