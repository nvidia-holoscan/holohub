# Volume Renderer

The `volume_renderer` operator renders a volume using ClaraViz (https://github.com/NVIDIA/clara-viz).

## `holoscan::ops::VolumeRenderer`

Operator class to render a volume.

### Parameters

- **`config_file`**: Config file path. The content of the file is passed to `clara::viz::JsonInterface::SetSettings()` at initialization time. See [Configuration](#configuration) for details.
  - type: `std::string`
- **`allocator`**: Allocator used to allocate render buffer outputs when no pre-allocated color or depth buffer is passed to `color_buffer_in` or `depth_buffer_in`. Allocator needs to be capable to allocate device memory.
  - type: `std::shared_ptr<Allocator>`
- **`alloc_width`**: Width of the render buffer to allocate when no pre-allocated buffers are provided.
  - type: `uint32_t`
- **`alloc_height`**: Height of the render buffer to allocate when no pre-allocated buffers are provided.
  - type: `uint32_t`

### Inputs

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
- **`camera_pose`**: Camera pose when not rendering in stereo mode.
  - type: `std::array<float, 16>` or `nvidia::gxf::Pose3D`
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

### Outputs

- **`color_buffer_out`**: Buffer with rendered color data, format is 8 bit per component RGBA and buffer is in device memory.
  - type: `nvidia::gxf::VideoBuffer`
- **`depth_buffer_out`**: Buffer with rendered depth data, format is be 32 bit float single component and buffer is in device memory.
  - type: `nvidia::gxf::VideoBuffer`

## Configuration

The renderer accepts a [ClaraViz](https://github.com/NVIDIA/clara-viz) JSON configuration file at startup to control rendering settings, including
- camera parameters;
- transfer functions;
- lighting;
- and more.

The ClaraViz JSON configuration file exists in addition to and independent of a Holoscan SDK `.yaml` configuration file that may be passed to an application.

See the [`volume_rendering_xr` application](../../applications/volume_rendering_xr/configs) for a sample configuration file. Visit the [ClaraViz `render_server.proto` gRPC specification](https://github.com/NVIDIA/clara-viz/blob/main/src/protos/nvidia/claraviz/cinematic/v1/render_server.proto) for insight into configuration file field values.

### Transfer Functions

Usually CT datasets are stored in [Hounsfield scale](https://en.wikipedia.org/wiki/Hounsfield_scale). The renderer maps these values in Hounsfield scale to opacity in order to display the volume. These mappings are called transfer functions. Multiple transfer functions for different input value regions can be defined. Transfer functions also include material properties like diffuse, specular and emissive color. The range of input values the transfer function is applied to is in normalized input range `[0, 1]`.

### Segmentation (Mask) Volume

Different organs often have very similar Hounsfield values, therefore additionally an segmentation volume is supported. The segmentation volume contains an integer index for each element of the volume. Transfer functions can be restricted on specific segmentation indices. The segmentation volume can, for example, be generated using [TotalSegmentator](https://github.com/wasserth/TotalSegmentator).

### Creating a Configuration File

Configuration files are typically specific to a given dataset or modality, and are tailored to a specific voxel intensity range.
It may be necessary to create a new configuration file when working with a new dataset in order to produce a meaningful rendering.

There are two options to create a configuration file for a new dataset:
- Copy from an existing configuration file as a reference and modify parameters manually. An example configuration file is available in the [`volume_rendering_xr` application config folder](../../applications/volume_rendering_xr/configs/).
- Use `VolumeRendererOp` to deduce settings for the input dataset. Follow these steps:
  1. Use the HoloHub [`volume_rendering` app](../../applications/volume_rendering/) or a similar application that will load an input dataset and pass it to `VolumeRendererOp`.
  2. Configure application settings via a Holoscan SDK YAML file or command line settings to run with the following values:
    - Set the `VolumeRendererOp` `config_file` parameter to an empty string to indicate no default config file is present;
    - Set the `VolumeRendererOp` `write_config_file` parameter to the desired output JSON configuration filepath.
  3. Run the application with the desired input volume. The operator will deduce settings and write out the JSON file to reuse on subsequent runs via the `config_file` parameter.
