/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "cvcuda_utils.hpp"
#include "gxf_utils.hpp"

#include "holoscan/core/resources/gxf/allocator.hpp"

#include <fmt/format.h>
#include <gxf/std/tensor.hpp>

namespace holoscan {

nvcv::DataType dldatatype_to_nvcvdatatype(DLDataType dtype, int num_channels) {
  nvcv::DataType type;
  uint8_t bits = dtype.bits;
  uint16_t channels = (num_channels == 0) ? dtype.lanes : num_channels;

  switch (dtype.code) {
    case kDLInt:
      switch (bits) {
        case 8:
          switch (channels) {
            case 1:
              type = nvcv::TYPE_S8;
              break;
            case 2:
              type = nvcv::TYPE_2S8;
              break;
            case 3:
              type = nvcv::TYPE_3S8;
              break;
            case 4:
              type = nvcv::TYPE_4S8;
              break;
            default:
              throw std::runtime_error(
                  fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                              dtype.code,
                              dtype.bits,
                              channels));
          }
          break;
        case 16:
          switch (channels) {
            case 1:
              type = nvcv::TYPE_S16;
              break;
            case 2:
              type = nvcv::TYPE_2S16;
              break;
            case 3:
              type = nvcv::TYPE_3S16;
              break;
            case 4:
              type = nvcv::TYPE_4S16;
              break;
            default:
              throw std::runtime_error(
                  fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                              dtype.code,
                              dtype.bits,
                              channels));
          }
          break;
        case 32:
          switch (channels) {
            case 1:
              type = nvcv::TYPE_S32;
              break;
            case 2:
              type = nvcv::TYPE_2S32;
              break;
            case 3:
              type = nvcv::TYPE_3S32;
              break;
            case 4:
              type = nvcv::TYPE_4S32;
              break;
            default:
              throw std::runtime_error(
                  fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                              dtype.code,
                              dtype.bits,
                              channels));
          }
          break;
        case 64:
          switch (channels) {
            case 1:
              type = nvcv::TYPE_S64;
              break;
            case 2:
              type = nvcv::TYPE_2S64;
              break;
            case 3:
              type = nvcv::TYPE_3S64;
              break;
            case 4:
              type = nvcv::TYPE_4S64;
              break;
            default:
              throw std::runtime_error(
                  fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                              dtype.code,
                              dtype.bits,
                              channels));
          }
          break;
        default:
          throw std::runtime_error(
              fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                          dtype.code,
                          dtype.bits,
                          channels));
      }
      break;
    case kDLUInt:
      switch (bits) {
        case 8:
          switch (channels) {
            case 1:
              type = nvcv::TYPE_U8;
              break;
            case 2:
              type = nvcv::TYPE_2U8;
              break;
            case 3:
              type = nvcv::TYPE_3U8;
              break;
            case 4:
              type = nvcv::TYPE_4U8;
              break;
            default:
              throw std::runtime_error(
                  fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                              dtype.code,
                              dtype.bits,
                              channels));
          }
          break;
        case 16:
          switch (channels) {
            case 1:
              type = nvcv::TYPE_U16;
              break;
            case 2:
              type = nvcv::TYPE_2U16;
              break;
            case 3:
              type = nvcv::TYPE_3U16;
              break;
            case 4:
              type = nvcv::TYPE_4U16;
              break;
            default:
              throw std::runtime_error(
                  fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                              dtype.code,
                              dtype.bits,
                              channels));
          }
          break;
        case 32:
          switch (channels) {
            case 1:
              type = nvcv::TYPE_U32;
              break;
            case 2:
              type = nvcv::TYPE_2U32;
              break;
            case 3:
              type = nvcv::TYPE_3U32;
              break;
            case 4:
              type = nvcv::TYPE_4U32;
              break;
            default:
              throw std::runtime_error(
                  fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                              dtype.code,
                              dtype.bits,
                              channels));
          }
          break;
        case 64:
          switch (channels) {
            case 1:
              type = nvcv::TYPE_U64;
              break;
            case 2:
              type = nvcv::TYPE_2U64;
              break;
            case 3:
              type = nvcv::TYPE_3U64;
              break;
            case 4:
              type = nvcv::TYPE_4U64;
              break;
            default:
              throw std::runtime_error(
                  fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                              dtype.code,
                              dtype.bits,
                              channels));
          }
          break;
        default:
          throw std::runtime_error(
              fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                          dtype.code,
                          dtype.bits,
                          channels));
      }
      break;
    case kDLFloat:
      switch (bits) {
        case 16:
          switch (channels) {
            case 1:
              type = nvcv::TYPE_F16;
              break;
            case 2:
              type = nvcv::TYPE_2F16;
              break;
            case 3:
              type = nvcv::TYPE_3F16;
              break;
            case 4:
              type = nvcv::TYPE_4F16;
              break;
            default:
              throw std::runtime_error(
                  fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                              dtype.code,
                              dtype.bits,
                              channels));
          }
          break;
        case 32:
          switch (channels) {
            case 1:
              type = nvcv::TYPE_F32;
              break;
            case 2:
              type = nvcv::TYPE_2F32;
              break;
            case 3:
              type = nvcv::TYPE_3F32;
              break;
            case 4:
              type = nvcv::TYPE_4F32;
              break;
            default:
              throw std::runtime_error(
                  fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                              dtype.code,
                              dtype.bits,
                              channels));
          }
          break;
        case 64:
          switch (channels) {
            case 1:
              type = nvcv::TYPE_F64;
              break;
            case 2:
              type = nvcv::TYPE_2F64;
              break;
            case 3:
              type = nvcv::TYPE_3F64;
              break;
            case 4:
              type = nvcv::TYPE_4F64;
              break;
            default:
              throw std::runtime_error(
                  fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                              dtype.code,
                              dtype.bits,
                              channels));
          }
          break;
        default:
          throw std::runtime_error(
              fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                          dtype.code,
                          dtype.bits,
                          channels));
      }
      break;
    case kDLComplex:
      switch (bits) {
        case 64:
          switch (channels) {
            case 1:
              type = nvcv::TYPE_C64;
              break;
            case 2:
              type = nvcv::TYPE_2C64;
              break;
            case 3:
              type = nvcv::TYPE_3C64;
              break;
            case 4:
              type = nvcv::TYPE_4C64;
              break;
            default:
              throw std::runtime_error(
                  fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                              dtype.code,
                              dtype.bits,
                              channels));
          }
          break;
        case 128:
          switch (channels) {
            case 1:
              type = nvcv::TYPE_C128;
              break;
            case 2:
              type = nvcv::TYPE_2C128;
              break;
            default:
              throw std::runtime_error(
                  fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                              dtype.code,
                              dtype.bits,
                              channels));
          }
          break;
        default:
          throw std::runtime_error(
              fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                          dtype.code,
                          dtype.bits,
                          channels));
      }
      break;
    default:
      throw std::runtime_error(
          fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                      dtype.code,
                      dtype.bits,
                      channels));
  }
  return type;
}

nvidia::gxf::PrimitiveType nvcvdatatype_to_gxfprimitivetype(nvcv::DataType dtype) {
  nvidia::gxf::PrimitiveType type;
  switch (dtype) {
    case nvcv::TYPE_U8:
    case nvcv::TYPE_2U8:
    case nvcv::TYPE_3U8:
    case nvcv::TYPE_4U8:
      type = nvidia::gxf::PrimitiveType::kUnsigned8;
      break;
    case nvcv::TYPE_U16:
    case nvcv::TYPE_2U16:
    case nvcv::TYPE_3U16:
    case nvcv::TYPE_4U16:
      type = nvidia::gxf::PrimitiveType::kUnsigned16;
      break;
    case nvcv::TYPE_U32:
    case nvcv::TYPE_2U32:
    case nvcv::TYPE_3U32:
    case nvcv::TYPE_4U32:
      type = nvidia::gxf::PrimitiveType::kUnsigned32;
      break;
    case nvcv::TYPE_U64:
    case nvcv::TYPE_2U64:
    case nvcv::TYPE_3U64:
    case nvcv::TYPE_4U64:
      type = nvidia::gxf::PrimitiveType::kUnsigned64;
      break;
    case nvcv::TYPE_S8:
    case nvcv::TYPE_2S8:
    case nvcv::TYPE_3S8:
    case nvcv::TYPE_4S8:
      type = nvidia::gxf::PrimitiveType::kInt8;
      break;
    case nvcv::TYPE_S16:
    case nvcv::TYPE_2S16:
    case nvcv::TYPE_3S16:
    case nvcv::TYPE_4S16:
      type = nvidia::gxf::PrimitiveType::kInt16;
      break;
    case nvcv::TYPE_S32:
    case nvcv::TYPE_2S32:
    case nvcv::TYPE_3S32:
    case nvcv::TYPE_4S32:
      type = nvidia::gxf::PrimitiveType::kInt32;
      break;
    case nvcv::TYPE_S64:
    case nvcv::TYPE_2S64:
    case nvcv::TYPE_3S64:
    case nvcv::TYPE_4S64:
      type = nvidia::gxf::PrimitiveType::kInt64;
      break;
    case nvcv::TYPE_F32:
    case nvcv::TYPE_2F32:
    case nvcv::TYPE_3F32:
    case nvcv::TYPE_4F32:
      type = nvidia::gxf::PrimitiveType::kFloat32;
      break;
    case nvcv::TYPE_F64:
    case nvcv::TYPE_2F64:
    case nvcv::TYPE_3F64:
    case nvcv::TYPE_4F64:
      type = nvidia::gxf::PrimitiveType::kFloat64;
      break;
    // Can uncomment the complex types below for Holoscan v1.0, but they are unsupported in v0.6.
    // case nvcv::TYPE_C64:
    // case nvcv::TYPE_2C64:
    // case nvcv::TYPE_3C64:
    // case nvcv::TYPE_4C64:
    //   type = nvidia::gxf::PrimitiveType::kComplex64;
    //   break;
    // case nvcv::TYPE_C128:
    // case nvcv::TYPE_2C128:
    //   type = nvidia::gxf::PrimitiveType::kComplex128;
    //   break;
    default:
      throw std::runtime_error("nvcv::DataType does not have a corresponding GXF primitive type");
  }
  return type;
}

nvcv::TensorDataStridedCuda::Buffer nhwc_buffer_from_holoscan_tensor(
    std::shared_ptr<holoscan::Tensor> tensor, std::shared_ptr<void*>& holoscan_tensor_data) {
  auto ndim = tensor->ndim();

  nvcv::TensorDataStridedCuda::Buffer in_buffer;
  auto in_strides = tensor->strides();
  if (ndim == 4) {
    // assume tensor has NHWC layout
    // copy strides from in_tensor->strides()
    for (auto d = 0; d < ndim; d++) { in_buffer.strides[d] = in_strides[d]; }
  } else if (ndim == 3) {
    // assume tensor has HWC layout
    // stride for batch dimension
    in_buffer.strides[0] = in_strides[0] * tensor->shape()[0];
    // remaining strides match in_tensor->strides()
    for (auto d = 0; d < ndim; d++) { in_buffer.strides[d + 1] = in_strides[d]; }
  } else if (ndim == 2) {
    // assume tensor has HW layout
    // stride for batch dimension
    in_buffer.strides[0] = in_strides[0] * tensor->shape()[0];
    // remaining strides match in_tensor->strides()
    for (auto d = 0; d < ndim; d++) { in_buffer.strides[d + 1] = in_strides[d]; }
    in_buffer.strides[3] = in_buffer.strides[2];
  } else {
    throw std::runtime_error(
        "expected a tensor with (height, width) or (height, width, channels) or "
        "(batch, height, width, channels) dimensions");
  }
  // cudaMalloc(&in_buffer.basePtr, tensor->nbytes());
  holoscan_tensor_data = get_custom_shared_ptr(tensor->nbytes());
  in_buffer.basePtr = static_cast<NVCVByte*>(*holoscan_tensor_data);
  cudaMemcpy(in_buffer.basePtr, tensor->data(), tensor->nbytes(), cudaMemcpyDeviceToDevice);
  return in_buffer;
}

std::pair<nvidia::gxf::Entity, std::shared_ptr<void*>> create_out_message_with_tensor_like(
    gxf_context_t context, nvcv::Tensor reference_nhwc_tensor,
    std::shared_ptr<Allocator> allocator) {
  // Create an out_message entity containing a single GXF tensor corresponding to the output.
  // We will then reuse the same data pointer to initialize the CV-CUDA output tensor.
  auto element_type = nvcvdatatype_to_gxfprimitivetype(reference_nhwc_tensor.dtype());
  auto shape = reference_nhwc_tensor.shape();
  if (shape.size() != 4) { throw std::runtime_error("expected 4D tensor (NHWC format)"); }
  nvidia::gxf::Shape out_shape;
  int n = shape[0];
  int h = shape[1];
  int w = shape[2];
  int c = shape[3];
  if (shape[0] == 1) {
    // note: omit singleton batch size since, e.g. HolovizOp expects HWC, not NHWC
    if (shape[3] == 1) {
      // note: omit singleton channel size
      out_shape = nvidia::gxf::Shape{h, w};
    } else {
      out_shape = nvidia::gxf::Shape{h, w, c};
    }
  } else {
    HOLOSCAN_LOG_DEBUG("Batched CVCUDA to Holoscan tensor with batch size of {}", n);
    out_shape = nvidia::gxf::Shape{n, h, w, c};
  }
  auto storage_type = nvidia::gxf::MemoryStorageType::kDevice;
  auto in_strided_data = reference_nhwc_tensor.exportData<nvcv::TensorDataStridedCuda>();
  return create_out_message_with_tensor(context,
                                        out_shape,
                                        element_type,
                                        storage_type,
                                        allocator,
                                        static_cast<void*>(in_strided_data->basePtr()));
}

void validate_cvcuda_tensor(nvcv::Tensor tensor) {
  auto number_dim = tensor.rank();
  if (number_dim > 4) {
    throw std::runtime_error("CV-CUDA tensors of more than four dimensions are not supported");
  }
}

}  // namespace holoscan
