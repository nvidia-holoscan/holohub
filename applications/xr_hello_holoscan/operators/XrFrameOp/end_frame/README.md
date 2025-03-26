### XrEndFrame Operator

The `XrEndFrameOp` operator completes the rendering of a single OpenXR frame by passing populated color and depth buffer for the left and right eye to the OpenXR device. Note that a single connection `xr_frame` from `XrBeginFrameOp` to `XrEndFrameOp` is required to synchronize the OpenXR calls issued by the two operators.

#### `holoscan::openxr::XrEndFrameOp`

##### Parameters 

- **`XrSession`**: A class that encapsulates a single OpenXR session
  - type: `holoscan::openxr::XrSession`

##### Inputs
 
Render buffers populated by application
- **`color_buffer`**: color buffer
  - type: `holoscan::gxf::VideoBuffer`
- **`depth_buffer`**: depth buffer
  - type: `holoscan::gxf::VideoBuffer`

OpenXR synchronization
- **`XrFrame`**: connection to synchronize `XrBeginFrameOp` and `XrEndFrameOp`
  - type: `XrFrame`



Note:

- **`XrCudaInteropSwapchain`**: A class that encapsulates the Vulkan buffers of the OpenXR runtime and compatible CUDA buffer to provide interoperability between ClaraViz and OpenXR