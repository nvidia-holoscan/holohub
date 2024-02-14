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

#include "gxf_utils.hpp"

#include <cuda_runtime.h>
#include <fmt/format.h>

#include <iostream>
#include <memory>

#include <gxf/std/tensor.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>

namespace holoscan {

void validate_holoscan_tensor(std::shared_ptr<holoscan::Tensor> in_tensor) {
  auto ndim_in = in_tensor->ndim();

  // assume 2D + channels without batch dimension
  if (ndim_in > 4) {
    throw std::runtime_error("Holoscan tensors of more than four dimensions are not supported");
  }

  // raise error if tensor data is not on the device
  DLDevice dev = in_tensor->device();
  if (dev.device_type != kDLCUDA) {
    throw std::runtime_error("expected input tensor to be on a CUDA device");
  }
}

std::pair<nvidia::gxf::Entity, std::shared_ptr<void*>> create_out_message_with_tensor(
    gxf_context_t context, nvidia::gxf::Shape shape, nvidia::gxf::PrimitiveType element_type,
    nvidia::gxf::MemoryStorageType storage_type, std::shared_ptr<Allocator> allocator, void* data) {
  int element_size = nvidia::gxf::PrimitiveTypeSize(element_type);
  size_t nbytes = shape.size() * element_size;

  std::shared_ptr<void*> pointer;

  switch (storage_type) {
    case nvidia::gxf::MemoryStorageType::kDevice: {
      pointer = get_custom_shared_ptr(nbytes, storage_type);
      *pointer = data;
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
  gxf_tensor.value()->wrapMemory(shape,
                                 element_type,
                                 element_size,
                                 nvidia::gxf::ComputeTrivialStrides(shape, element_size),
                                 nvidia::gxf::MemoryStorageType::kDevice,
                                 *pointer,
                                 [orig_pointer = pointer](void*) mutable {
                                   orig_pointer.reset();  // decrement ref count
                                   return nvidia::gxf::Success;
                                 });

  return std::make_pair(out_message.value(), pointer);
}
}  // namespace holoscan
