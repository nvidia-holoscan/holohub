### CVCUDA Holoscan Interoperability Operators

This directory contains two operators to enable interoperability between the [CVCUDA](https://github.com/CVCUDA/CV-CUDA) and Holoscan
tensors: `holoscan::ops::CvCudaToHoloscan` and `holoscan::ops::HoloscanToCvCuda`.

#### `holoscan::ops::CvCudaToHoloscan`

Operator class to convert a `nvcv::Tensor` to a `holoscan::Tensor`.

##### Inputs

- **`input`**: a CV-CUDA tensor
  - type: `nvcv::Tensor`

##### Outputs

- **`output`**: a Holoscan tensor as `holoscan::Tensor` in `holoscan::TensorMap`
  - type: `holoscan::TensorMap`

#### `holoscan::ops::HoloscanToCvCuda`

##### Inputs

- **`input`**: a `gxf::Entity` containing a Holoscan tensor as `holoscan::Tensor`
  - type: `gxf::Entity`

##### Outputs

- **`output`**: a CV-CUDA tensor
  - type: `nvcv::Tensor`