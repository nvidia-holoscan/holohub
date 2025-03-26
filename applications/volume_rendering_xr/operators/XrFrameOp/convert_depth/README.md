### Convert Depth To Screen Space Operator

The `ConvertDepthToScreenSpaceOp` operator remaps the depth buffer from Clara Viz to an OpenXR specific range. The depth buffer is converted in place.

#### `holoscan::openxr::ConvertDepthToScreenSpaceOp`

Converts a depth buffer from linear world units to screen space ([0,1])

##### Inputs

- **`depth_buffer_in`**: input depth buffer to be remapped
  - type: `holoscan::gxf::VideoBuffer`
- **`depth_range`**: Allocator used to allocate the volume data
  - type: `nvidia::gxf::Vector2f`

##### Outputs
- **`depth_buffer_out`**: output depth buffer 
  - type: `holoscan::gxf::Entity`
