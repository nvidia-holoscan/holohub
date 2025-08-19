# Intel RealSense Camera Operator

## Overview

The RealSense Camera Operator captures synchronized color and depth frames from Intel RealSense cameras using the RealSense SDK. This operator provides real-time streaming capabilities with configurable resolution and frame rates, making it ideal for computer vision applications, robotics, and 3D reconstruction tasks.

## Features

- **Dual Stream Capture**: Simultaneously captures color (RGBA) and depth (Z16) video streams
- **Synchronized Frames**: Ensures temporal alignment between color and depth data
- **GPU Memory Management**: Efficient CUDA memory allocation and transfer
- **Camera Intrinsics**: Provides camera calibration parameters for both streams
- **Configurable Resolution**: Supports various resolution and frame rate combinations
- **Real-time Processing**: Optimized for low-latency streaming applications

## Usage

### Basic Example

```cpp
#include "holoscan/holoscan.hpp"
#include "realsense_camera.hpp"

class MyApp : public holoscan::Application {
 public:
  void compose() override {
    auto realsense = make_operator<holoscan::ops::RealsenseCameraOp>(
        "realsense",
        Arg("allocator", make_resource<holoscan::UnboundedAllocator>("pool")));
    
    add_operator(realsense);
  }
};
```

### Advanced Configuration

```cpp
auto realsense = make_operator<holoscan::ops::RealsenseCameraOp>(
    "realsense",
    Arg("allocator", make_resource<holoscan::UnboundedAllocator>("pool")));
```

Please refer to the [Intel RealSense Camera Visualizer](../../applications/realsense_visualizer/cpp/realsense_visualizer.cpp) reference application for an example usage.

## Outputs

The operator provides the following outputs:

### Video Buffers

- **`color_buffer`**: RGBA8 color video stream (1280x720 @ 30fps by default)
- **`depth_buffer`**: Z16 depth video stream (1280x720 @ 30fps by default)

### Camera Models

- **`color_camera_model`**: Intrinsic parameters for the color camera
- **`depth_camera_model`**: Intrinsic parameters for the depth camera

## Camera Model Structure

Each camera model contains:

- **Dimensions**: Width and height in pixels
- **Focal Length**: fx, fy in pixels
- **Principal Point**: ppx, ppy in pixels
- **Distortion Type**: Currently set to Perspective

## Configuration

### Default Settings

- **Color Stream**: 1280x720, RGBA8 format, 30fps
- **Depth Stream**: 1280x720, Z16 format, 30fps
- **Alignment**: Depth frames aligned to color stream

### Supported Formats

- **Color**: RGBA8, RGB8, BGR8, YUYV
- **Depth**: Z16, Y16, Y8
