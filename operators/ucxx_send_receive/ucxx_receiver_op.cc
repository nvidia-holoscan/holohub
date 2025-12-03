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

#include "ucxx_receiver_op.h"

#include <holoscan/holoscan.hpp>

namespace {

// Deserializes a tensor object from the given buffer.
// 
// This function reconstructs a holoscan::Tensor from a raw binary buffer
// that contains a serialized TensorHeader followed by contiguous tensor data.
// The function performs sanity checks on the header, copies shape/stride
// information, allocates the tensor using the provided holoscan::Allocator,
// and copies the tensor data based on storage type.
//
// Args:
//   buffer: Pointer to the start of the serialized tensor buffer.
//   buffer_size: Size of the buffer in bytes.
//   allocator: Pointer to an allocator to use for tensor memory.
//
// Returns:
//   holoscan::expected<holoscan::Tensor, holoscan::RuntimeError> containing the deserialized tensor,
//   or an error if deserialization fails.
// 
// Based on nvidia::gxf::StdComponentSerializer::deserializeTensor.
holoscan::expected<holoscan::Tensor, holoscan::RuntimeError> deserializeTensor(
    const uint8_t* buffer, size_t buffer_size,
    holoscan::Allocator* allocator) {
  if (buffer == nullptr || buffer_size < sizeof(holoscan::TensorHeader)) {
    return holoscan::make_unexpected(holoscan::RuntimeError("Invalid buffer or buffer size"));
  }

  // Read header from buffer.
  const holoscan::TensorHeader* header = reinterpret_cast<const holoscan::TensorHeader*>(buffer);

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
  Tensor::stride_array_t strides;
  std::memcpy(strides.data(), header->strides, sizeof(header->strides));

  Tensor tensor;
  auto result = tensor.reshapeCustom(Shape(dims, header->rank),
                                     header->element_type, header->bytes_per_element, strides,
                                     header->storage_type, allocator);
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
      auto staging_buffer = allocator->allocate(tensor_size, MemoryStorageType::kHost);
      if (!staging_buffer) {
        HOLOSCAN_LOG_ERROR("Failed to allocate staging buffer for device memory copy");
        return holoscan::make_unexpected(holoscan::RuntimeError("Failed to allocate staging buffer"));
      }
      
      // Copy data to staging buffer
      std::memcpy(staging_buffer.value(), tensor_data, tensor_size);
      
      // Copy from staging buffer to device memory
      const cudaError_t error = cudaMemcpy(tensor.pointer(), staging_buffer.value(), tensor_size,
                                           cudaMemcpyHostToDevice);
      if (error != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("Failure in CudaMemcpy. cuda_error: %s, error_str: %s",
                      cudaGetErrorName(error), cudaGetErrorString(error));
        allocator->free(staging_buffer.value());
        return holoscan::make_unexpected(holoscan::RuntimeError("Failed to copy data to staging buffer"));
      }
      
      // Free the staging buffer
      auto free_result = allocator->free(staging_buffer.value());
      if (!free_result) {
        HOLOSCAN_LOG_ERROR("Failed to free staging buffer");
        return holoscan::make_unexpected(holoscan::RuntimeError("Failed to free staging buffer"));
      }
      break;
    }
    default:
      return holoscan::make_unexpected(holoscan::RuntimeError("Invalid memory storage type"));
  }

  return tensor;
}

}  // namespace

namespace holoscan::ops {

void UcxxReceiverOp::setup(holoscan::OperatorSpec& spec) {
  spec.param(tag_, "tag", "Tag", "UCX tag number", 0ul);
  spec.param(buffer_size_, "buffer_size", "Buffer size", "Receive buffer size", 4 << 10);
  spec.param(endpoint_, "endpoint", "Endpoint", "UcxxEndpoint resource");
  spec.param(allocator_, "allocator", "Allocator", "Allocator for staging buffers");
  spec.output<std::any>("out");

  // Add the endpoint's is_alive_condition to this operator so that it will only execute only when
  // the endpoint is alive.
  for (auto arg : args()) {
    if (arg.name() == "endpoint") {
      auto resource = std::any_cast<std::shared_ptr<holoscan::Resource>>(arg.value());
      auto endpoint = std::dynamic_pointer_cast<UcxxEndpoint>(resource);
      add_arg(endpoint->is_alive_condition());
      break;
    }
  }
}

void UcxxReceiverOp::start() { buffer_.resize(buffer_size_.get()); }

void UcxxReceiverOp::stop() {
  if (request_) {
    request_->cancel();
  }
}

void UcxxReceiverOp::compute([[maybe_unused]] holoscan::InputContext& input,
                             holoscan::OutputContext& output,
                             [[maybe_unused]] holoscan::ExecutionContext& context) {
  // If the receive request is complete, deserialize and emit it.
  if (request_ && request_->isCompleted()) {
    if (auto status = request_->getStatus(); status == UCS_OK) {
      auto tensor = deserializeTensor(buffer_.data(), buffer_.size(), allocator_.get().get());
      if (!tensor) {
        HOLOSCAN_LOG_ERROR("Failed to deserialize tensor");
        return;  
      }
      output.emit(tensor.value(), "out");
    } else {
      HOLOSCAN_LOG_ERROR("Receive request failed with status: {}", ucs_status_string(status));
    }
    request_ = nullptr;
  }

  // Post a new request if none is pending.
  if (!request_) {
    async_condition()->event_state(holoscan::AsynchronousEventState::EVENT_WAITING);
    request_ = endpoint_->endpoint()->tagRecv(
        buffer_.data(), buffer_.size(), ucxx::Tag{tag_.get()}, ucxx::TagMaskFull,
        /*enablePythonFuture=*/false, [this](ucs_status_t, std::shared_ptr<void>) {
          async_condition()->event_state(holoscan::AsynchronousEventState::EVENT_DONE);
        });
  }
}

}  // namespace isaac::os::ops
