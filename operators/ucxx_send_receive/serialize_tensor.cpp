/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "serialize_tensor.hpp"

#include <array>
#include <cstring>
#include <cuda_runtime.h>

namespace holoscan::ops::ucxx {

holoscan::expected<size_t, holoscan::RuntimeError> serializeTensor(
    const nvidia::gxf::Tensor& tensor,
    uint8_t* buffer,
    size_t buffer_size,
    holoscan::Allocator* allocator) {

  const size_t tensor_size = tensor.element_count() * tensor.bytes_per_element();
  const size_t required_size = sizeof(TensorHeader) + tensor_size;

  if (buffer == nullptr || buffer_size < required_size) {
    HOLOSCAN_LOG_ERROR("Buffer size %zu is too small for tensor serialization (need %zu)",
                       buffer_size, required_size);
    return holoscan::make_unexpected(holoscan::RuntimeError("Buffer size too small"));
  }

  // Write header to buffer.
  TensorHeader* header = reinterpret_cast<TensorHeader*>(buffer);
  header->storage_type = tensor.storage_type();
  header->element_type = tensor.element_type();
  header->bytes_per_element = tensor.bytes_per_element();
  header->rank = tensor.rank();

  for (size_t i = 0; i < Shape::kMaxRank; i++) {
    header->dims[i] = tensor.shape().dimension(i);
    header->strides[i] = tensor.stride(i);
  }

  // Data pointer is immediately after the header.
  uint8_t* tensor_data = buffer + sizeof(TensorHeader);

  switch (tensor.storage_type()) {
    case MemoryStorageType::kHost:
    case MemoryStorageType::kSystem:
    case MemoryStorageType::kCudaManaged:
      std::memcpy(tensor_data, tensor.pointer(), tensor_size);
      break;
    case MemoryStorageType::kDevice: {
      // Use allocator to get a staging buffer (pinned host memory) for faster transfer
      auto staging_buffer = allocator->allocate(tensor_size, holoscan::MemoryStorageType::kHost);
      if (!staging_buffer) {
        HOLOSCAN_LOG_ERROR("Failed to allocate staging buffer for device memory copy");
        return holoscan::make_unexpected(
            holoscan::RuntimeError("Failed to allocate staging buffer"));
      }

      // Copy from device memory to staging buffer
      const cudaError_t error = cudaMemcpy(staging_buffer, tensor.pointer(), tensor_size,
                                           cudaMemcpyDeviceToHost);
      if (error != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("Failure in CudaMemcpy. cuda_error: %s, error_str: %s",
                           cudaGetErrorName(error), cudaGetErrorString(error));
        allocator->free(staging_buffer);
        return holoscan::make_unexpected(
            holoscan::RuntimeError("Failed to copy data to staging buffer"));
      }

      // Copy from staging buffer to output buffer
      std::memcpy(tensor_data, staging_buffer, tensor_size);

      // Free the staging buffer
      allocator->free(staging_buffer);
      break;
    }
    default:
      HOLOSCAN_LOG_ERROR("Invalid memory storage type %d specified for tensor storage",
                         static_cast<int>(tensor.storage_type()));
      return holoscan::make_unexpected(holoscan::RuntimeError("Invalid memory storage type"));
  }

  return required_size;
}

holoscan::expected<nvidia::gxf::Tensor, holoscan::RuntimeError> deserializeTensor(
    const uint8_t* buffer, size_t buffer_size,
    gxf_context_t context,
    holoscan::Allocator* allocator) {
  if (buffer == nullptr || buffer_size < sizeof(TensorHeader)) {
    return holoscan::make_unexpected(holoscan::RuntimeError("Invalid buffer or buffer size"));
  }

  // Read header from buffer.
  const TensorHeader* header = reinterpret_cast<const TensorHeader*>(buffer);

  // Safety checks for header fields.
  if (sizeof(header->dims) > Shape::kMaxRank * sizeof(int32_t)) {
    HOLOSCAN_LOG_ERROR("Header size exceeds limit of %lu.",
                  Shape::kMaxRank * sizeof(int32_t));
    return holoscan::make_unexpected(holoscan::RuntimeError("Header size exceeds limit"));
  }
  if (sizeof(header->strides) > Shape::kMaxRank * sizeof(int64_t)) {
    HOLOSCAN_LOG_ERROR("Header size exceeds limit of %lu.",
                  Shape::kMaxRank * sizeof(int64_t));
    return holoscan::make_unexpected(holoscan::RuntimeError("Header size exceeds limit"));
  }

  std::array<int32_t, Shape::kMaxRank> dims;
  std::memcpy(dims.data(), header->dims, sizeof(header->dims));
  nvidia::gxf::Tensor::stride_array_t strides;
  std::memcpy(strides.data(), header->strides, sizeof(header->strides));

  // Convert holoscan::Allocator to nvidia::gxf::Handle<nvidia::gxf::Allocator>
  auto gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
      context, allocator->gxf_cid());
  if (!gxf_allocator) {
    HOLOSCAN_LOG_ERROR("Failed to create GXF allocator handle");
    return holoscan::make_unexpected(
        holoscan::RuntimeError("Failed to create GXF allocator handle"));
  }

  nvidia::gxf::Tensor tensor;
  auto result = tensor.reshapeCustom(Shape(dims, header->rank),
                                     header->element_type, header->bytes_per_element, strides,
                                     header->storage_type, gxf_allocator.value());
  if (!result) {
    return holoscan::make_unexpected(holoscan::RuntimeError("Failed to reshape tensor"));
  }

  const size_t tensor_size = tensor.element_count() * tensor.bytes_per_element();

  // Data pointer is immediately after the header.
  const uint8_t* tensor_data = buffer + sizeof(TensorHeader);
  size_t remaining = buffer_size - sizeof(TensorHeader);
  if (remaining < tensor_size) {
    HOLOSCAN_LOG_ERROR("Buffer size %zu is too small to contain tensor data of size %zu",
                  buffer_size, tensor_size);
    return holoscan::make_unexpected(holoscan::RuntimeError("Buffer size is too small"));
  }

  switch (tensor.storage_type()) {
    case MemoryStorageType::kHost:
    case MemoryStorageType::kSystem:
    case MemoryStorageType::kCudaManaged:
      std::memcpy(tensor.pointer(), tensor_data, tensor_size);
      break;
    case MemoryStorageType::kDevice: {
      // Use allocator to get a staging buffer (pinned host memory) for faster transfer
      auto staging_buffer = allocator->allocate(tensor_size, holoscan::MemoryStorageType::kHost);
      if (!staging_buffer) {
        HOLOSCAN_LOG_ERROR("Failed to allocate staging buffer for device memory copy");
        return holoscan::make_unexpected(
            holoscan::RuntimeError("Failed to allocate staging buffer"));
      }

      // Copy data to staging buffer
      std::memcpy(staging_buffer, tensor_data, tensor_size);

      // Copy from staging buffer to device memory
      const cudaError_t error = cudaMemcpy(tensor.pointer(), staging_buffer, tensor_size,
                                           cudaMemcpyHostToDevice);
      if (error != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("Failure in CudaMemcpy. cuda_error: %s, error_str: %s",
                      cudaGetErrorName(error), cudaGetErrorString(error));
        allocator->free(staging_buffer);
        return holoscan::make_unexpected(
            holoscan::RuntimeError("Failed to copy data to staging buffer"));
      }

      // Free the staging buffer
      allocator->free(staging_buffer);
      break;
    }
    default:
      return holoscan::make_unexpected(holoscan::RuntimeError("Invalid memory storage type"));
  }

  return tensor;
}

}  // namespace holoscan::ops::ucxx
