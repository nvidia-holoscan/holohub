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
#include <cuda_runtime.h>

#include <memory>

#include <gxf/std/tensor.hpp>
#include <holoscan/core/domain/tensor.hpp>

namespace holoscan {

std::pair<nvidia::gxf::Entity, std::shared_ptr<void*>> create_out_message_with_tensor(
    gxf_context_t context, nvidia::gxf::Shape shape, nvidia::gxf::PrimitiveType element_type,
    nvidia::gxf::MemoryStorageType storage_type) {
  int element_size = nvidia::gxf::PrimitiveTypeSize(element_type);
  size_t nbytes = shape.size() * element_size;
  // Create a shared pointer for the CUDA memory with a custom deleter.
  auto pointer = std::shared_ptr<void*>(new void*, [](void** pointer) {
    if (pointer != nullptr) {
      if (*pointer != nullptr) { cudaFree(*pointer); }
      delete pointer;
    }
  });

  // Allocate the CUDA memory (don't need to explicitly initialize)
  cudaError_t err = cudaMalloc(pointer.get(), nbytes);
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
