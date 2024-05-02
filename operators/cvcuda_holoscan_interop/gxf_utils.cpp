/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gxf_utils.hpp"

#include <gxf/std/tensor.hpp>
#include "holoscan/holoscan.hpp"

#include <cuda_runtime.h>
#include <fmt/format.h>
#include <holoscan/core/domain/tensor.hpp>
#include <iostream>
#include <memory>
#include <nvcv/Tensor.hpp>

#if __has_include("gxf/std/dlpack_utils.hpp")
  #define GXF_HAS_DLPACK_SUPPORT 1
#else
  #define GXF_HAS_DLPACK_SUPPORT 0
#endif

namespace holoscan {

void validate_holoscan_tensor(std::shared_ptr<holoscan::Tensor> in_tensor) {
  auto ndim_in = in_tensor->ndim();

  if (ndim_in > 4) {
    throw std::runtime_error("Holoscan tensors of more than four dimensions are not supported");
  }

  // raise error if tensor data is not on the device
  DLDevice dev = in_tensor->device();
  if (dev.device_type != kDLCUDA) {
    throw std::runtime_error("expected input tensor to be on a CUDA device");
  }
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
#if GXF_HAS_DLPACK_SUPPORT
    // GXF_HAS_DLPACK_SUPPORT is only true for Holoscan >=2.0 (GXF 4.0) so it is safe to define
    // these in that case. Complex type support in GXF was previously disabled in
    // Holoscan <= v0.6. Once Holohub drops support of Holoscan v0.6 we can enable these
    // unconditionally.
    case nvcv::TYPE_C64:
    case nvcv::TYPE_2C64:
    case nvcv::TYPE_3C64:
    case nvcv::TYPE_4C64:
      type = nvidia::gxf::PrimitiveType::kComplex64;
      break;
    case nvcv::TYPE_C128:
    case nvcv::TYPE_2C128:
      type = nvidia::gxf::PrimitiveType::kComplex128;
      break;
#endif
    default:
      throw std::runtime_error("nvcv::DataType does not have a corresponding GXF primitive type");
  }
  return type;
}

std::pair<nvidia::gxf::Entity, std::shared_ptr<void*>> create_out_message_with_tensor(
    gxf_context_t context, nvcv::Tensor reference_nhwc_tensor) {
  // Create an out_message entity containing a single GXF tensor corresponding to the output.
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

  int element_size = nvidia::gxf::PrimitiveTypeSize(element_type);
  size_t nbytes = shape.size() * element_size;

  std::shared_ptr<void*> pointer;

  switch (storage_type) {
    case nvidia::gxf::MemoryStorageType::kDevice: {
      pointer = get_custom_shared_ptr(nbytes, storage_type);
      *pointer = static_cast<void*>(in_strided_data->basePtr());
    } break;
    case nvidia::gxf::MemoryStorageType::kHost:
    case nvidia::gxf::MemoryStorageType::kSystem:
    default:
      throw std::runtime_error("storage_type out of range");
  }

  // Holoscan Tensor doesn't support direct memory allocation.
  // Thus, create an Entity and use GXF tensor to wrap the CUDA memory.
  auto out_message = nvidia::gxf::Entity::New(context);
  auto gxf_tensor = out_message.value().add<nvidia::gxf::Tensor>("image");
  gxf_tensor.value()->wrapMemory(out_shape,
                                 element_type,
                                 element_size,
                                 nvidia::gxf::ComputeTrivialStrides(out_shape, element_size),
                                 nvidia::gxf::MemoryStorageType::kDevice,
                                 *pointer,
                                 [orig_pointer = pointer](void*) mutable {
                                   orig_pointer.reset();  // decrement ref count
                                   return nvidia::gxf::Success;
                                 });

  return std::make_pair(out_message.value(), pointer);
}
}  // namespace holoscan
