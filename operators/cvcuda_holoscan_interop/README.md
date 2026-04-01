# CVCUDA Holoscan Interoperability Operators

This directory contains two operators to enable interoperability between the [CVCUDA](https://github.com/CVCUDA/CV-CUDA) and Holoscan
tensors: `holoscan::ops::CvCudaToHoloscan` and `holoscan::ops::HoloscanToCvCuda`.

## `holoscan::ops::CvCudaToHoloscan`

Operator class to convert a `nvcv::Tensor` to a `holoscan::Tensor`.

### Inputs (CvCudaToHoloscan)

- **`input`**: a CV-CUDA tensor
  - type: `nvcv::Tensor`

### Outputs (CvCudaToHoloscan)

- **`output`**: a Holoscan tensor as `holoscan::Tensor` in `holoscan::TensorMap`
  - type: `holoscan::TensorMap`

## `holoscan::ops::HoloscanToCvCuda`

Operator class to convert Holoscan tensor to CV-CUDA.

### Inputs (HoloscanToCvCuda)

- **`input`**: a `gxf::Entity` containing a Holoscan tensor as `holoscan::Tensor`
  - type: `gxf::Entity`

### Outputs (HoloscanToCvCuda)

- **`output`**: a CV-CUDA tensor
  - type: `nvcv::Tensor`
