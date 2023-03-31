### GXF Tensor to VideoBuffer Converter

The `tensor_to_video_buffer` converts GXF Tensor to VideoBuffer.

#### `holoscan::ops::TensorToVideoBufferOp`

Operator class to convert GXF Tensor to VideoBuffer. This operator is required
for data transfer  between Holoscan operators that output GXF Tensor and
the other Holoscan Wrapper Operators that understand only VideoBuffer.
It receives GXF Tensor as input and outputs GXF VideoBuffer created from it.

##### Parameters

- **`data_in`**: Data in GXF Tensor format
 - type: `holoscan::IOSpec*`
- **`data_out`**: Data in GXF VideoBuffer format
 - type: `holoscan::IOSpec*`
- **`in_tensor_name`**: Name of the input tensor
  - type: `std::string`
- **`video_format`**: The video format, supported values: "yuv420", "rgb"
  - type: `std::string`
