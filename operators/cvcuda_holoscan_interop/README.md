### CVCUDA Holoscan Interoperability Operators

This directory contains two operators to enable interoperability between the [CVCUDA](https://github.com/CVCUDA/CV-CUDA) and Holoscan
tensors: `holoscan::ops::CvCudaToHoloscan` and `holoscan::ops::HoloscanToCvCuda`.

#### `holoscan::ops::CvCudaToHoloscan`

Operator class to convert a `nvcv::Tensor` to a `holoscan::Tensor`.

##### Inputs

- **`input`**: a CV-CUDA tensor
  - type: `nvcv::Tensor`

##### Outputs

- **`output`**: a Holoscan tensor
  - type: `holoscan::Tensor`

#### `holoscan::ops::HoloscanToCvCuda`

##### Inputs

- **`input`**: a Holoscan tensor map (a `gxf::Entity` is also accepted, and it will be converted to
  a tensor map automatically). A `holoscan::Tensor` could also be passed inside a `holoscan::TensorMap`.
  - type: `holoscan::TensorMap`

##### Outputs

- **`output`**: A CV-CUDA tensor
  - type: `nvcv::Tensor`