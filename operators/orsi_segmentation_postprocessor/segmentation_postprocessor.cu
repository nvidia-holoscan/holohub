/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "segmentation_postprocessor.cuh"

namespace holoscan::ops::orsi {
namespace segmentation_postprocessor {

__forceinline__ __device__ uint32_t hwc_to_index(Shape shape, uint32_t y, uint32_t x, uint32_t c) {
  return (y * shape.width + x) * shape.channels + c;
}

__forceinline__ __device__ uint32_t nchw_to_index(Shape shape, uint32_t y, uint32_t x, uint32_t c) {
  return (c * shape.height + y) * shape.width + x;
}

template <enum DataFormat>
__forceinline__ __device__ uint32_t data_format_to_index(Shape shape, uint32_t y, uint32_t x,
                                                         uint32_t c) {}

template <>
__forceinline__ __device__ uint32_t data_format_to_index<DataFormat::kHWC>(Shape shape, uint32_t y,
                                                                           uint32_t x, uint32_t c) {
  return hwc_to_index(shape, y, x, c);
}

template <>
__forceinline__ __device__ uint32_t data_format_to_index<DataFormat::kNHWC>(Shape shape, uint32_t y,
                                                                            uint32_t x,
                                                                            uint32_t c) {
  return hwc_to_index(shape, y, x, c);
}

template <>
__forceinline__ __device__ uint32_t data_format_to_index<DataFormat::kNCHW>(Shape shape, uint32_t y,
                                                                            uint32_t x,
                                                                            uint32_t c) {
  return nchw_to_index(shape, y, x, c);
}

__forceinline__ __device__ uint32_t hw1_to_index(Shape shape, uint32_t y, uint32_t x) {
  return y * shape.width + x;
}

template <enum NetworkOutputType network_output_type, enum DataFormat data_format>
__global__ void postprocessing_kernel(Shape shape, const float* input, output_type_t* output) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x >= shape.width) || (y >= shape.height)) { return; }

  uint8_t max_index = 0;

  switch (network_output_type) {
    case NetworkOutputType::kSigmoid: {
      const float value = input[data_format_to_index<data_format>(shape, y, x, 0)];
      max_index = value >= 0.5f ? 1 : 0;
    } break;
    case NetworkOutputType::kSoftmax: {
      float max_value = 0.0f;
      for (uint32_t c = 0; c < shape.channels; c++) {
        const float value = input[data_format_to_index<data_format>(shape, y, x, c)];
        if (value > max_value) {
          max_value = value;
          max_index = c;
        }
      }
    } break;
  }

  output[hw1_to_index(shape, y, x)] = max_index;
}

uint16_t ceil_div(uint16_t numerator, uint16_t denominator) {
  uint32_t accumulator = numerator + denominator - 1;
  return accumulator / denominator;
}

template <enum DataFormat data_format>
__global__ void threshold_kernel(Shape shape, output_type_t* input, output_type_t* output) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  const uint32_t n = shape.width*shape.height;
  if (((x >= shape.width) || (y >= shape.height)) && n>(blockDim.x*blockDim.y)) { return; }

  
  const float value = input[data_format_to_index<data_format>(shape, y, x, 0)];
  output[hw1_to_index(shape, y, x)]  = value >= 0.5f ? 1 : 0;
}

__forceinline__ __device__ double sigmoid (double a)
{
    return 1.0 / (1.0 + exp(-a));
}

template <enum DataFormat data_format>
__global__ void sigmoid_kernel (Shape shape, const float * __restrict__ src, 
                                output_type_t * __restrict__ dst)
{

  const float value = src[0];
  
  dst[0] = (sigmoid(value)>= 0.5f) ? 1 : 0;
}

void cuda_postprocess(enum NetworkOutputType network_output_type, enum DataFormat data_format,
                      Shape shape, const float* input, output_type_t* output,
                      cudaStream_t cuda_stream) {
  dim3 block(32, 32, 1);
  dim3 grid(ceil_div(shape.width, block.x), ceil_div(shape.height, block.y), 1);
  switch (network_output_type) {
    case NetworkOutputType::kSigmoid:
      switch (data_format) {
        case DataFormat::kNCHW:
          postprocessing_kernel<NetworkOutputType::kSigmoid, DataFormat::kNCHW>
              <<<grid, block, 0, cuda_stream>>>(shape, input, output);
          break;
        case DataFormat::kHWC:
          postprocessing_kernel<NetworkOutputType::kSigmoid, DataFormat::kHWC>
              <<<grid, block, 0, cuda_stream>>>(shape, input, output);
          break;
        case DataFormat::kNHWC:
          postprocessing_kernel<NetworkOutputType::kSigmoid, DataFormat::kNHWC>
              <<<grid, block, 0, cuda_stream>>>(shape, input, output);
          break;
      }
      break;
    case NetworkOutputType::kSoftmax:
      switch (data_format) {
        case DataFormat::kNCHW:
          postprocessing_kernel<NetworkOutputType::kSoftmax, DataFormat::kNCHW>
              <<<grid, block, 0, cuda_stream>>>(shape, input, output);
          break;
        case DataFormat::kHWC:
          postprocessing_kernel<NetworkOutputType::kSoftmax, DataFormat::kHWC>
              <<<grid, block, 0, cuda_stream>>>(shape, input, output);
          break;
        case DataFormat::kNHWC:
          postprocessing_kernel<NetworkOutputType::kSoftmax, DataFormat::kNHWC>
              <<<grid, block, 0, cuda_stream>>>(shape, input, output);
          break;
      }
      break;
      case NetworkOutputType::kRawValues:
      switch (data_format) {
        case DataFormat::kNCHW:
          sigmoid_kernel<DataFormat::kNCHW><<<grid, block>>>(shape, input, output);
          break;
        case DataFormat::kHWC:
          sigmoid_kernel<DataFormat::kHWC>
                <<<grid, block>>>(shape, input, output);
          break;
        case DataFormat::kNHWC:
          sigmoid_kernel<DataFormat::kNHWC>
                <<<grid, block>>>(shape, input, output);
          break;
      }
      break;
  }
}


__global__ void resizing_kernel(Shape input_shape, Shape output_shape, const uint8_t* input, uint8_t* output, int32_t offset_x, int32_t offset_y) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x >= input_shape.width) || (y >= input_shape.height)) { return; }

  output[hw1_to_index(output_shape, y + offset_y, x  + offset_x)] = input[hw1_to_index(input_shape, y, x)];
}

void cuda_resize(Shape input_shape, Shape output_shape, const uint8_t* input, uint8_t* output, int32_t offset_x, int32_t offset_y) {
  dim3 block(32, 32, 1);
  dim3 grid(ceil_div(input_shape.width, block.x), ceil_div(input_shape.height, block.y), 1);
  
  resizing_kernel<<<grid, block>>>(input_shape, output_shape, input, output, offset_x, offset_y);
}

}  // namespace segmentation_postprocessor
}  // namespace holoscan::ops
