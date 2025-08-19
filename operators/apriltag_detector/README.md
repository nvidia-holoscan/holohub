# AprilTag Detection Operator

The `apriltag_detection` extension provides real-time detection of [April tags](https://github.com/AprilRobotics/apriltag) from different tag families. The detection and processing is performed efficiently using CUDA acceleration for optimal performance in real-time applications.

## Overview

April tags are 2D barcode-like patterns that can be used for camera calibration, pose estimation, and object tracking. This operator integrates the AprilTag detection library with NVIDIA Holoscan, providing GPU-accelerated tag detection capabilities.

## Features

- **CUDA-accelerated processing**: Leverages GPU computing for high-performance tag detection
- **Multiple tag family support**: Compatible with various AprilTag families
- **Real-time performance**: Optimized for low-latency applications
- **Configurable detection parameters**: Adjustable detection sensitivity and output limits

## Usage

```python
from holoscan.operators import ApriltagDetectorOp

# Create the operator
apriltag_op = ApriltagDetectorOp(
    width=1920,
    height=1080,
    number_of_tags=10
)
```

## API Reference

 `ApriltagDetectorOp`: The AprilTag detection operator class that processes input video streams and outputs detected tag information.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **`width`** | `int` | `None` | Width of the input video stream in pixels. Must match the actual input resolution. |
| **`height`** | `int` | `None` | Height of the input video stream in pixels. Must match the actual input resolution. |
| **`number_of_tags`** | `int` | `None` | Maximum number of April tags to detect and output. Higher values may impact performance. |

### Input/Output Ports

- **`input`**: Video stream input
- **`output`**: Detected AprilTag ID and corner coordinates
