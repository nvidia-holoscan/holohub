### User interface Render Operator

The `XrTransformRenderOp` renders the mixed reality user interface of the volumetric rendering application. It consumes interface widget state structures as well as render buffers into which to overlay the interface widgets. The operator is application specific and will grow over time to include additional user interface widgets.

#### `holoscan::openxr::XrTransformRenderOp`

##### Parameters 

- **`display_width`**: pixel height of display
  - type: `int`
- **`display_height`**: pixel width of display
  - type: `int`
 
##### Inputs

Camera state for stereo view
- **`left_camera_pose`**: world space pose of the left eye
  - type: `nvidia::gxf::Pose3D`
- **`right_camera_pose`**: world space pose of the right eye
  - type: `nvidia::gxf::Pose3D`
- **`left_camera_model`**: camera model for the left eye
  - type: `nvidia::gxf::CameraModel`
- **`right_camera_model`**: camera model for the right eye
  - type: `nvidia::gxf::CameraModel`
- **`depth_range`**: depth range

User interface widget state structures
- **`ux_box`**: bounding box state structure
  - type: `UxBoundingBox`
- **`ux_cursor`**: cursor state structure
  - type: `UxCursor`

Render buffers to be populated
- **`Collor buffer_in`**: color buffer
  - type: `holoscan::gxf::VideoBuffer`
- **`Depth buffer_in`**: depth buffer
  - type: `holoscan::gxf::VideoBuffer`

##### Outputs

Render buffers including interface widgets
- **`color_buffer_out`**: color buffer
  - type: `holoscan::gxf::VideoBuffer`
- **`depth_buffer_out`**: depth buffer
  - type: `holoscan::gxf::VideoBuffer`
