/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#ifndef HOLOHUB_OPERATORS_CVCUDA_HOLOSCAN_INTEROP_GXF_UTILS_HPP
#define HOLOHUB_OPERATORS_CVCUDA_HOLOSCAN_INTEROP_GXF_UTILS_HPP

#include <memory>

#include <gxf/std/tensor.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>
#include <iostream>

namespace holoscan {

inline std::shared_ptr<void*> get_custom_shared_ptr(
    int64_t nbytes,
    nvidia::gxf::MemoryStorageType storage_type = nvidia::gxf::MemoryStorageType::kDevice) {
  void* ptr = nullptr;

  if (storage_type != nvidia::gxf::MemoryStorageType::kDevice) {
    throw std::runtime_error(
        "Only device memory is supported for interoperability between CVCUDA and Holoscan "
        "tensors.");
  }
  cudaMalloc(&ptr, nbytes);

  std::shared_ptr<void*> pointer(new void*(ptr), [](void** pointer) {
    if (pointer != nullptr) {
      if (*pointer != nullptr) {
        cudaFree(*pointer);
        // std::cout << "Freeing memory" << std::endl;
      }
      delete pointer;
    }
  });
  return pointer;
}
/**
 * @brief Generate output message containing a single tensor
 *
 * @param context The GXF context.
 * @param shape The element type of the tensor.
 * @param element_type The element type of the tensor.
 * @param storage_type The storage type of the tensor.
 * @return Pair containing the GXF entity as well as a std::shared_ptr<void*> corresponding to
 * the tensor data.
 */
std::pair<nvidia::gxf::Entity, std::shared_ptr<void*>> create_out_message_with_tensor(
    gxf_context_t context, nvidia::gxf::Shape shape, nvidia::gxf::PrimitiveType element_type,
    nvidia::gxf::MemoryStorageType storage_type, std::shared_ptr<Allocator> allocator, void* data);

/**
 * @brief Validate the Holoscan tensor has less than or equal to four dimensions and  is on the
 * device
 *
 * @param tensormap A shared pointer to holoscan::Tensor
 */
void validate_holoscan_tensor(std::shared_ptr<holoscan::Tensor> in_tensor);

}  // namespace holoscan

#endif /* HOLOHUB_OPERATORS_CVCUDA_HOLOSCAN_INTEROP_GXF_UTILS_HPP */
