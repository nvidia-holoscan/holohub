### NPP Filter

The `npp_filter` operator uses [NPP](https://developer.nvidia.com/npp) to apply a filters to a Tensor or VideBuffer.

#### `holoscan::ops::NppFilter`

Operator class to apply a filter of the [NPP library]() to a Tensor or VideBuffer.

##### Parameters

- **`filter`**: Name of the filter to apply (supported Gauss, SobelHoriz, SobelVert)
  - type: `std::string`
- **`mask_size`**: Filter mask size (supported values 3, 5, 7, 9, 11, 13)
  - type: `uint32_t`
- **`allocator`**: Allocator used to allocate the output data
  - type: `std::shared_ptr<Allocator>`

##### Inputs

- **`input`**: Input frame data
  - type: `nvidia::gxf::Tensor` or `nvidia::gxf::VideoBuffer`

##### Outputs

- **`input`**: Output frame data
  - type: `nvidia::gxf::Tensor` or `nvidia::gxf::VideoBuffer`
