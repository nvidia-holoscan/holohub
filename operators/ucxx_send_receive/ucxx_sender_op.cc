/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ucxx_sender_op.h"

#include <cuda_runtime.h>
#include <holoscan/holoscan.hpp>

namespace {

using nvidia::gxf::Shape;
using nvidia::gxf::PrimitiveType;
using nvidia::gxf::MemoryStorageType;

#pragma pack(push, 1)
struct TensorHeader {
  MemoryStorageType storage_type;     // CPU or GPU tensor
  PrimitiveType element_type;         // Tensor element type
  uint64_t bytes_per_element;         // Bytes per tensor element
  uint32_t rank;                      // Tensor rank
  int32_t dims[Shape::kMaxRank];      // Tensor dimensions
  uint64_t strides[Shape::kMaxRank];  // Tensor strides
};
#pragma pack(pop)

// Serializes a tensor object into the given buffer.
// 
// This function converts a nvidia::gxf::Tensor into a raw binary buffer
// format consisting of a TensorHeader followed by contiguous tensor data.
// It copies shape/stride information into the header and copies the tensor
// data based on storage type, using a staging buffer for device memory.
//
// Args:
//   tensor: The tensor to serialize.
//   buffer: Pointer to the destination buffer (must be pre-allocated).
//   buffer_size: Size of the destination buffer in bytes.
//   allocator: Pointer to an allocator to use for staging buffers.
//
// Returns:
//   holoscan::expected<size_t> containing the number of bytes written,
//   or an error if serialization fails.
// 
// Mirrors the serializeTensor function in ucxx_receiver_op.cc.
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
        return holoscan::make_unexpected(holoscan::RuntimeError("Failed to allocate staging buffer"));
      }
      
      // Copy from device memory to staging buffer
      const cudaError_t error = cudaMemcpy(staging_buffer, tensor.pointer(), tensor_size,
                                           cudaMemcpyDeviceToHost);
      if (error != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("Failure in CudaMemcpy. cuda_error: %s, error_str: %s",
                           cudaGetErrorName(error), cudaGetErrorString(error));
        allocator->free(staging_buffer);
        return holoscan::make_unexpected(holoscan::RuntimeError("Failed to copy data to staging buffer"));
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

}  // namespace

namespace holoscan::ops {

void UcxxSenderOp::setup(holoscan::OperatorSpec& spec) {
  spec.param(tag_, "tag", "Tag", "UCX tag number", 0ul);
  spec.param(endpoint_, "endpoint", "Endpoint", "UcxxEndpoint resource");
  spec.param(allocator_, "allocator", "Allocator", "Allocator for staging buffers");
  spec.input<holoscan::gxf::Entity>("in");

  // Add the endpoint's is_alive_condition to this operator so that it will only execute when the
  // endpoint is alive.
  for (auto arg : args()) {
    if (arg.name() == "endpoint") {
      auto resource = std::any_cast<std::shared_ptr<holoscan::Resource>>(arg.value());
      auto endpoint = std::dynamic_pointer_cast<UcxxEndpoint>(resource);
      add_arg(endpoint->is_alive_condition());
      break;
    }
  }
}

void UcxxSenderOp::compute(holoscan::InputContext& input, holoscan::OutputContext&,
                           holoscan::ExecutionContext&) {
  auto in_message = input.receive<holoscan::gxf::Entity>("in").value();
  auto maybe_tensor = in_message.get<holoscan::Tensor>("");
  if (!maybe_tensor) {
    HOLOSCAN_LOG_ERROR("Failed to get tensor from input message");
    return;
  }

  // Convert holoscan::Tensor to nvidia::gxf::Tensor for serialization
  auto gxf_tensor = std::make_shared<nvidia::gxf::Tensor>(maybe_tensor->dl_ctx());
  if (!gxf_tensor) {
    HOLOSCAN_LOG_ERROR("Failed to convert holoscan::Tensor to nvidia::gxf::Tensor");
    return;
  }

  // Calculate required buffer size for serialization
  const size_t tensor_size = gxf_tensor->element_count() * gxf_tensor->bytes_per_element();
  const size_t buffer_size = sizeof(TensorHeader) + tensor_size;

  // Create a send request with pre-allocated buffer
  SendRequest& send = requests_.emplace_back();
  send.buffer.resize(buffer_size);

  // Serialize the tensor into the buffer
  auto result = serializeTensor(*gxf_tensor, send.buffer.data(), send.buffer.size(), 
                                allocator_.get().get());
  if (!result.has_value()) {
    HOLOSCAN_LOG_ERROR("Failed to serialize tensor: {}", result.error().what());
    requests_.pop_back();
    return;
  }

  // Send the serialized tensor buffer
  send.request = endpoint_->endpoint()->tagSend(
      send.buffer.data(), result.value(), ucxx::Tag{tag_.get()});

  // Clean up completed requests
  for (auto it = requests_.begin(); it != requests_.end();) {
    if (!it->request->isCompleted()) {
      it++;
      continue;
    }
    if (ucs_status_t status = it->request->getStatus(); status != UCS_OK) {
      HOLOSCAN_LOG_ERROR("Send failed with status: {}", ucs_status_string(status));
    }
    it = requests_.erase(it);
  }
}

}  // namespace holoscan::ops
