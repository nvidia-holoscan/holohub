### Volume Renderer

The `volume_renderer` operator renders a volume using ClaraViz (https://github.com/NVIDIA/clara-viz).

#### `holoscan::ops::VolumeRenderer`

Operator class to render a volume.

##### Parameters

- **`config_file`**: Config file path. The content of the file is passed to `clara::viz::JsonInterface::SetSettings()` at initialization time.
  - type: `std::string`
- **`allocator`**: Allocator used to allocate render buffer outputs when no pre-allocated color or depth buffer is passed to `color_buffer_in` or `depth_buffer_in`. Allocator needs to be capable to allocate device memory.
  - type: `std::shared_ptr<Allocator>`
- **`alloc_width`**: Width of the render buffer to allocate when no pre-allocated buffers are provided.
  - type: `uint32_t`
- **`alloc_height`**: Height of the render buffer to allocate when no pre-allocated buffers are provided.
  - type: `uint32_t`

##### Inputs

All inputs are optional.

- **`volume_pose`**: Transform the volume.
  - type: `nvidia::gxf::Pose3D`
- **`crop_box`**: Volume crop box. Each `nvidia::gxf::Vector2f` contains the min and max values in range `[0, 1]` of the x, y and z axes of the volume.
  - type: `std::array<nvidia::gxf::Vector2f, 3>`
- **`depth_range`**: The distance to the near and far frustum planes.
  - type: `nvidia::gxf::Vector2f`
- **`left_camera_pose`**: Camera pose for the left camera when rendering in stereo mode.
  - type: `nvidia::gxf::Pose3D`
- **`right_camera_pose`**: Camera pose for the right camera when rendering in stereo mode.
  - type: `nvidia::gxf::Pose3D`
- **`left_camera_model`**: Camera model for the left camera when rendering in stereo mode.
  - type: `nvidia::gxf::CameraModel`
- **`right_camera_model`**: Camera model for the right camera when rendering in stereo mode.
  - type: `nvidia::gxf::CameraModel`
- **`camera_matrix`**: Camera pose when not rendering in stereo mode.
  - type: `std::array<float, 16>`
- **`color_buffer_in`**: Buffer to store the rendered color data to, format needs to be 8 bit per component RGBA and buffer needs to be in device memory.
  - type: `nvidia::gxf::VideoBuffer`
- **`depth_buffer_in`**: Buffer to store the rendered depth data to, format needs to be 32 bit float single component buffer needs to be in device memory.
  - type: `nvidia::gxf::VideoBuffer`
- **`density_volume`**: Density volume data. Needs to be a 3D single component array. Supported data types are signed|unsigned 8|16|32 bit integer and 32 bit floating point.
  - type: `nvidia::gxf::Tensor`
- **`density_spacing`**: Physical size of each density volume element.
  - type: `std::array<float, 3>`
- **`density_permute_axis`**: Density volume axis permutation of data space to world space, e.g. if x and y of a volume is swapped this is {1, 0, 2}.
  - type: `std::array<uint32_t, 3>`
- **`density_flip_axes`**: Density volume axis flipping from data space to world space, e.g. if x is flipped this is {true, false, false}.
  - type: `std::array<bool, 3>`
- **`mask_volume`**: Mask volume data. Needs to be a 3D single component array. Supported data types are signed|unsigned 8|16|32 bit integer and 32 bit floating point.
  - type: `nvidia::gxf::Tensor`
- **`mask_spacing`**: Physical size of each mask volume element.
  - type: `std::array<float, 3>`
- **`mask_permute_axis`**: Mask volume axis permutation of data space to world space, e.g. if x and y of a volume is swapped this is {1, 0, 2}.
  - type: `std::array<uint32_t, 3>`
- **`mask_flip_axes`**: Mask volume axis flipping from data space to world space, e.g. if x is flipped this is {true, false, false}.
  - type: `std::array<bool, 3>`

##### Outputs

- **`color_buffer_out`**: Buffer with rendered color data, format is 8 bit per component RGBA and buffer is in device memory.
  - type: `nvidia::gxf::VideoBuffer`
- **`depth_buffer_out`**: Buffer with rendered depth data, format is be 32 bit float single component and buffer is in device memory.
  - type: `nvidia::gxf::VideoBuffer`
