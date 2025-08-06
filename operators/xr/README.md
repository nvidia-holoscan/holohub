# OpenXR Operators

A collection of Holoscan operators for OpenXR (Extended Reality) integration, enabling real-time XR applications with Vulkan graphics and CUDA compute capabilities.

## Overview

The XR operators provide a complete framework for building Holoscan applications that can render to XR headsets. The implementation leverages the [OpenXR standard](https://www.khronos.org/openxr/) for cross-platform XR support, Vulkan for graphics, and CUDA for compute operations, enabling high-performance real-time XR applications.

## Core Components

### Frame Management Operators

#### XrBeginFrameOp

- Synchronizes application with XR device timing
- Calls `xrWaitFrame` and `xrBeginFrame`
- Outputs frame state for downstream operators

#### XrEndFrameOp

- Submits composition layers to XR device
- Calls `xrEndFrame` with rendered content
- Handles multiple composition layers

### XrSession

The central resource that manages the OpenXR session lifecycle:

- **Initialization**: Creates OpenXR instance, system, session, and Vulkan graphics context
- **Plugin System**: Supports extensible plugins for additional XR functionality
- **Space Management**: Provides reference and view spaces for pose tracking
- **Graphics Integration**: Manages Vulkan instance, device, and queue creation

### Rendering Infrastructure

#### XrSwapchainCuda

- Manages OpenXR swapchains with CUDA interop
- Provides Vulkan/CUDA memory sharing via external memory
- Supports multiple formats: RGBA8, depth buffers
- Handles GPU synchronization between CUDA and Vulkan

#### XrCompositionLayerManager

- Manages color and depth swapchains
- Creates composition layer storage for frame rendering
- Provides simplified interface for XR rendering

#### XrCompositionLayerProjectionStorage

- Stores projection layer views and depth information
- Supports stereo rendering with side-by-side layout
- Handles depth range and near/far plane configuration

### Input Tracking

#### XrHandTracker

- Implements OpenXR hand tracking extension
- Provides joint locations for both hands
- Supports 26 hand joints per hand
- Real-time pose tracking with validation

## Features

### Graphics Pipeline

- **Vulkan Integration**: Native Vulkan graphics with OpenXR
- **CUDA Interop**: Direct CUDA memory access for compute operations
- **Multi-format Support**: Color (RGBA8) and depth (D16/D32) formats
- **Stereo Rendering**: Side-by-side layout for VR headsets

### Input Systems

- **Hand Tracking**: Full hand joint tracking with OpenXR EXT_hand_tracking
- **Pose Tracking**: 6DOF head and controller tracking
- **Space Management**: Reference and view space coordinate systems

### Performance Optimizations

- **GPU Synchronization**: Efficient CUDA-Vulkan synchronization
- **Memory Sharing**: Zero-copy memory between CUDA and Vulkan
- **Command Buffering**: Pre-recorded Vulkan command buffers
- **External Semaphores**: Cross-API synchronization primitives

## Dependencies

### External Libraries

- **OpenXR SDK**: 1.0.26 - Cross-platform XR API
- **OpenXR-Hpp**: 1.0.26 - C++ headers for OpenXR
- **GLM**: 1.0.1 - Mathematics library for graphics
- **Vulkan**: Graphics API for XR rendering
- **CUDA**: Compute API for GPU operations

### Holoscan Requirements

- **Minimum SDK Version**: 3.1.0
- **Tested Versions**: 3.1.0
- **Platforms**: x86_64

## Usage Example

```python
import holoscan as hs
from holoscan.operators.xr import (
    XrSession, XrBeginFrameOp, XrEndFrameOp,
    XrCompositionLayerManager, XrSwapchainCuda
)

class XRApp(hs.Application):
    def compose(self):
        # Create XR session
        xr_session = XrSession(
            self,
            application_name="My XR App",
            application_version=1
        )
        
        # Create composition layer manager
        composition_manager = XrCompositionLayerManager(
            self, xr_session=xr_session
        )
        
        # Frame management operators
        begin_frame = XrBeginFrameOp(
            self, xr_session=xr_session
        )
        
        end_frame = XrEndFrameOp(
            self, xr_session=xr_session
        )
        
        # Define workflow
        self.add_flow(begin_frame, end_frame)
```

## Python API

### Core Classes

**XrSession**

```python
# Create XR session
session = XrSession(
    fragment,
    application_name="My App",
    application_version=1
)

# Access session properties
view_configs = session.view_configurations()
depth_range = session.view_configuration_depth_range()
```

**XrSwapchainCuda**

```python
# Create swapchain
swapchain = XrSwapchainCuda(
    session,
    format=XrSwapchainCudaFormat.R8G8B8A8_SRGB,
    width=1920, height=1080
)

# Acquire image for rendering
tensor = swapchain.acquire()
# ... render to tensor ...
swapchain.release(cuda_stream)
```

**XrHandTracker**

```python
# Create hand tracker
hand_tracker = XrHandTracker(
    fragment,
    xr_session=session,
    hand=XrHandEXT.XR_HAND_LEFT_EXT
)

# Get hand joint locations
joints = hand_tracker.locate_hand_joints()
```

### Data Types

**Pose and Transform**

```python
# 3D position
position = XrVector3f(x=0.0, y=0.0, z=0.0)

# Quaternion rotation
orientation = XrQuaternionf(x=0.0, y=0.0, z=0.0, w=1.0)

# Complete pose
pose = XrPosef(position=position, orientation=orientation)
```

**Frame State**

```python
# Frame timing information
frame_state = XrFrameState(
    predictedDisplayTime=1234567890,
    predictedDisplayPeriod=16666667,
    shouldRender=True
)
```

## Building

### CMake Configuration

```bash
./holohub build xr
```

## Platform Support

- **XR Runtimes**: OpenXR-compatible headsets (Meta Quest, Valve Index, etc.)
- **Graphics**: Vulkan 1.3+
- **Compute**: CUDA 11.0+
- **OS**: Linux (x86_64)

## Performance Considerations

### Memory Management

- External memory handles enable zero-copy between CUDA and Vulkan
- Pre-allocated command buffers reduce CPU overhead
- Efficient swapchain image management

### Synchronization

- External semaphores coordinate CUDA and Vulkan operations
- Proper fence management ensures GPU synchronization
- Minimal CPU-GPU synchronization overhead

### Rendering Pipeline

- Side-by-side stereo layout optimized for VR
- Depth buffer support for proper occlusion
- Configurable depth ranges for different applications

## Troubleshooting

### Common Issues

#### XR Session Creation Fails

- Verify XR runtime is installed and running
- Check Vulkan driver compatibility
- Ensure proper permissions for XR devices

#### CUDA-Vulkan Interop Issues

- Verify CUDA and Vulkan driver versions are compatible
- Check external memory handle support
- Ensure proper memory alignment

#### Performance Issues

- Monitor GPU utilization and memory usage
- Profile frame timing and synchronization overhead
- Consider reducing swapchain resolution or complexity
