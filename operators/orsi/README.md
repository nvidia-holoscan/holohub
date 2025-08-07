# Orsi Academy Operators

A collection of specialized operators for medical imaging and surgical visualization applications, providing comprehensive tools for format conversion, segmentation processing, and 3D visualization.

## Overview

The Orsi Academy operators provide a complete pipeline for medical imaging applications, from data preprocessing to final visualization. These operators are designed for surgical guidance systems and medical AI workflows, supporting both CPU and GPU processing with CUDA acceleration.

## Operators

### 1. Format Converter (`orsi_format_converter`)

Converts between different image and tensor formats with support for resizing, normalization, and channel reordering.

**Key Features:**
- Multiple format conversions (RGB888, RGBA8888, Float32, YUV420, NV12)
- Image resizing with different interpolation modes
- Channel reordering and normalization
- CUDA-accelerated processing using NPP libraries
- Support for ROI (Region of Interest) cropping

**Usage:**
```python
from holoscan.operators import OrsiFormatConverterOp

converter = OrsiFormatConverterOp(
    fragment=fragment,
    allocator=allocator,
    out_dtype="float32",
    in_dtype="rgb888",
    scale_min=0.0,
    scale_max=1.0,
    resize_width=512,
    resize_height=512
)
```

### 2. Segmentation Preprocessor (`orsi_segmentation_preprocessor`)

Prepares input data for segmentation neural networks with normalization and preprocessing.

**Key Features:**
- Data format conversion (HWC/CHW)
- Mean/std normalization
- CUDA-accelerated preprocessing
- Support for different network input formats
- Flexible tensor handling

**Usage:**
```python
from holoscan.operators import OrsiSegmentationPreprocessorOp

preprocessor = OrsiSegmentationPreprocessorOp(
    fragment=fragment,
    allocator=allocator,
    data_format="hwc",
    normalize_means=[0.485, 0.456, 0.406],
    normalize_stds=[0.229, 0.224, 0.225]
)
```

### 3. Segmentation Postprocessor (`orsi_segmentation_postprocessor`)

Processes neural network outputs to generate segmentation masks and visualizations.

**Key Features:**
- Network output processing (softmax, sigmoid)
- Segmentation mask generation
- Image resizing and ROI handling
- CUDA-accelerated postprocessing
- Support for different output formats

**Usage:**
```python
from holoscan.operators import OrsiSegmentationPostprocessorOp

postprocessor = OrsiSegmentationPostprocessorOp(
    fragment=fragment,
    allocator=allocator,
    network_output_type="softmax",
    data_format="hwc",
    output_img_size=[512, 512]
)
```

### 4. Visualizer (`orsi_visualizer`)

Advanced 3D visualization system for surgical guidance with OpenGL rendering and VTK integration.

**Key Features:**
- Real-time video frame rendering
- 3D STL model visualization with VTK
- Surgical tool overlay effects
- Anonymization effects for privacy
- Interactive camera controls
- CUDA-OpenGL interop for performance
- Multi-window support

**Usage:**
```python
from holoscan.operators import OrsiVisualizationOp

visualizer = OrsiVisualizationOp(
    fragment=fragment,
    stl_file_path="/path/to/anatomy.stl",
    stl_names=["liver", "kidney"],
    stl_colors=[[255, 0, 0], [0, 255, 0]],
    registration_params_path="/path/to/registration.json"
)
```

## Common Parameters

### Allocator
All operators require a shared allocator for memory management:
```python
allocator = holoscan.resources.UnboundedAllocator(fragment)
```

### CUDA Stream Pool
For GPU-accelerated operations:
```python
cuda_stream_pool = holoscan.resources.CudaStreamPool(fragment)
```

## Data Flow

Typical pipeline configuration:

```
Video Source → Format Converter → Segmentation Preprocessor → AI Model → Segmentation Postprocessor → Visualizer
```

## Supported Formats

### Input Formats
- RGB888, RGBA8888
- Float32 tensors
- YUV420, NV12 video formats
- Various tensor layouts (HWC, CHW)

### Output Formats
- Processed video frames
- Segmentation masks
- 3D visualizations
- Normalized tensors

## Integration

These operators are designed to work together in medical imaging pipelines:

- **Preprocessing**: Format conversion and normalization
- **AI Processing**: Segmentation model inference
- **Postprocessing**: Mask generation and refinement
- **Visualization**: Real-time surgical guidance display

## Hardware Requirements

- NVIDIA GPU with CUDA support
- Sufficient GPU memory for video processing
- OpenGL-compatible graphics driver
- VTK library for 3D visualization 