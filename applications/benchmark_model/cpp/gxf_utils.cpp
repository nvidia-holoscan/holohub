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
#include <fmt/format.h>

#include <memory>

#include <gxf/std/tensor.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <iostream>

namespace holoscan {

std::pair<nvidia::gxf::Entity, std::shared_ptr<void*>> create_out_message_with_tensor(
    gxf_context_t context, nvidia::gxf::Shape shape, nvidia::gxf::PrimitiveType element_type,
    nvidia::gxf::MemoryStorageType storage_type) {
  int element_size = nvidia::gxf::PrimitiveTypeSize(element_type);
  size_t nbytes = shape.size() * element_size;

  std::shared_ptr<void*> pointer;

  switch (storage_type) {
    case nvidia::gxf::MemoryStorageType::kHost: {
      // Page-locked memory on the host

      // Create a shared pointer for the page-locked memory with a custom deleter.
      pointer = std::shared_ptr<void*>(new void*, [](void** pointer) {
        if (pointer != nullptr) {
          if (*pointer != nullptr) { cudaFreeHost(*pointer); }
          delete pointer;
        }
      });

      // Allocate the page-locked memory (don't need to explicitly initialize)
      const cudaError_t err = cudaMallocHost(pointer.get(), nbytes);
      if (err != cudaSuccess) {
        throw std::runtime_error(
            fmt::format("Failure in cudaMallocHost. cuda_error: {}, error_str: {}",
                        cudaGetErrorName(err),
                        cudaGetErrorString(err)));
      }
    } break;
    case nvidia::gxf::MemoryStorageType::kDevice: {
      // Create a shared pointer for the CUDA memory with a custom deleter.
      pointer = std::shared_ptr<void*>(new void*, [](void** pointer) {
        if (pointer != nullptr) {
          if (*pointer != nullptr) { cudaFree(*pointer); }
          delete pointer;
        }
      });

      // Allocate the CUDA memory (don't need to explicitly initialize)
      // const cudaError_t err = cudaMalloc(pointer.get(), nbytes);
      const cudaError_t err = cudaMalloc(pointer.get(), nbytes);
      if (err != cudaSuccess) {
        throw std::runtime_error(fmt::format("Failure in cudaMalloc. cuda_error: {}, error_str: {}",
                                             cudaGetErrorName(err),
                                             cudaGetErrorString(err)));
      }
      std::cout << "nvidia::gxf::MemoryStorageType::kDevice" << std::endl;
    } break;
    case nvidia::gxf::MemoryStorageType::kSystem: {
      // system memory (via new/delete)

      // Create a shared pointer for system memory with a custom deleter.
      pointer = std::shared_ptr<void*>(new void*, [](void** pointer) {
        if (pointer != nullptr) {
          if (*pointer != nullptr) { ::operator delete(*pointer); }
          delete pointer;
        }
      });

      // Allocate system memory.
      *pointer = ::operator new(nbytes);
    } break;
    default:
      throw std::runtime_error("storage_type out of range");
  }

  // Holoscan Tensor doesn't support direct memory allocation.
  // Thus, create an Entity and use GXF tensor to wrap the CUDA memory.
  auto out_message = nvidia::gxf::Entity::New(context);
  auto gxf_tensor = out_message.value().add<nvidia::gxf::Tensor>("image");
  std::cout << "gxf_tensor.value()->wrapMemory" << std::endl;
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
