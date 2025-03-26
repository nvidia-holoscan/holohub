### User interface Control Operator

The `XrTransformControlOp` maintains the state of the mixed reality user interface for the volumetric rendering application. It consumes controller events and produces user interface state structures as well as parameters for the volume rendering operator. The operator is application specific and will grow over time to include additional user interface widgets.

#### `holoscan::openxr::XrTransformControlOp`

##### Inputs
 
Controller state
- **`trigger_click`**: trigger button state
  - type: `bool`
- **`shoulder_click`**: shoulder button state
  - type: `bool`
- **`trackpad_touch`**: trackpad state
  - type: `bool`
- **`trackpad`**: trackpad values [x,y]
  - type: `std::array<float, 2>`
- **`aim_pose`**: world space pose of the controller tip
  - type: `nvidia::gxf::Pose3D`

Device state
- **`head_pose`**: world space head pose of the device
  - type: `nvidia::gxf::Pose3D`

Volume state
- **`extent`**: size of bounding box containing volume
  - type: `std::array<float, 3>`

##### Outputs

User interface widget state structures
- **`ux_box`**: bounding box state structure
  - type: `UxBoundingBox`
- **`ux_cursor`**: cursor state structure
  - type: `UxCursor`

Volume rendering parameters
- **`volume_pose`**: world pose of dataset 
  - type: `nvidia::gxf::Pose3D`
- **`crop_box`**: axis aligned cropping planes in local coordinates
  - type: `std::array<nvidia::gxf::Vector2f, 3>`