/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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
#include <nvcv/Tensor.hpp>

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
        *pointer = nullptr;
      }
      delete pointer;
      pointer = nullptr;
    }
  });
  return pointer;
}

/**
 * @brief Return the GXF tensor primitive type corresponding to a CV-CUDA dtype.
 *
 * The GXF primitive type does not include information on the number of interleaved channels.
 *
 * @param dtype CV-CUDA data type.
 * @return The GXF tensor primitive type
 */
nvidia::gxf::PrimitiveType nvcvdatatype_to_gxfprimitivetype(nvcv::DataType dtype);

/**
 * @brief Generate output message containing a single Holoscan tensor corresponding to the CVCUDA
 * reference_nhwc_tensor
 *
 * Note: If dimensions N or C have size 1, the corresponding dimensions will be dropped from
 * the output tensor.
 *
 * @param context The GXF context.
 * @param reference_nhwc_tensor The reference CV-CUDA tensor.
 * @return A GXF entity containing a single tensor like reference_nhwc_tensor.
 */
std::pair<nvidia::gxf::Entity, std::shared_ptr<void*>> create_out_message_with_tensor(
    gxf_context_t context, nvcv::Tensor reference_nhwc_tensor);

/**
 * @brief Validate the Holoscan tensor has less than or equal to four dimensions and  is on the
 * device
 *
 * @param tensormap A shared pointer to holoscan::Tensor
 */
void validate_holoscan_tensor(std::shared_ptr<holoscan::Tensor> in_tensor);

}  // namespace holoscan

#endif /* HOLOHUB_OPERATORS_CVCUDA_HOLOSCAN_INTEROP_GXF_UTILS_HPP */
