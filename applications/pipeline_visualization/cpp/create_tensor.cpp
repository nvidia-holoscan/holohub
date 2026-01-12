/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "create_tensor.hpp"

#include <cstring>
#include <stdexcept>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/utils/cuda_macros.hpp>

namespace pipeline_visualization::flatbuffers {

::flatbuffers::Offset<Tensor> CreateTensor(::flatbuffers::FlatBufferBuilder& fbb,
                                           const std::shared_ptr<holoscan::Tensor>& tensor,
                                           std::optional<cudaStream_t> stream) {
  // Get the DLPack context from the tensor
  const auto& dl_ctx = tensor->dl_ctx();
  if (!dl_ctx) {
    // Return null offset if tensor has no valid DLPack context
    return 0;
  }

  // Serialize tensor shape (or null offset if empty)
  ::flatbuffers::Offset<::flatbuffers::Vector<int64_t>> shape =
      tensor->shape().size() ? fbb.CreateVector(tensor->shape()) : 0;

  // Extract data type information from the tensor
  DLDataType dtype(static_cast<DLDataTypeCode>(tensor->dtype().code),
                   tensor->dtype().bits,
                   tensor->dtype().lanes);

  // Extract device information (e.g., CPU, CUDA GPU)
  DLDevice dl_device(static_cast<DLDeviceType>(tensor->device().device_type),
                     tensor->device().device_id);

  // Get number of dimensions
  uint32_t ndim = tensor->ndim();

  // Serialize strides if present (or null offset if not available)
  // Strides define the memory layout of multi-dimensional arrays
  ::flatbuffers::Offset<::flatbuffers::Vector<int64_t>> strides =
      (dl_ctx->tensor.dl_tensor.strides != nullptr && dl_ctx->tensor.dl_tensor.ndim > 0)
          ? fbb.CreateVector(dl_ctx->tensor.dl_tensor.strides, dl_ctx->tensor.dl_tensor.ndim)
          : 0;
  // Materialize tensor data when enabled via WithTensorMaterialization scope.
  // By default, this is a null offset (no data serialization).
  ::flatbuffers::Offset<::flatbuffers::Vector<uint8_t>> data = 0;
  if (tensor->data()) {
    // Use tensor->nbytes() to get the actual number of bytes, which handles packed data types
    // correctly. This avoids overshooting for packed dtypes like 1-bit bool or 4-bit INT4.
    size_t total_bytes = tensor->nbytes();
    const auto& device = tensor->device();

    // Handle data copying based on device type (GPU vs CPU memory)
    switch (device.device_type) {
      case kDLCUDAManaged:
      case kDLCUDA: {
        // For CUDA device memory, allocate host buffer and copy from GPU
        uint8_t* dst = nullptr;
        data = fbb.CreateUninitializedVector<uint8_t>(total_bytes, &dst);

        // Use appropriate CUDA memcpy type based on whether memory is managed
        auto copy_type =
            device.device_type == kDLCUDAManaged ? cudaMemcpyDefault : cudaMemcpyDeviceToHost;

        // Perform GPU-to-host memory transfer
        if (stream) {
          HOLOSCAN_CUDA_CALL_THROW_ERROR(
              cudaMemcpyAsync(dst, tensor->data(), total_bytes, copy_type, *stream),
              "Failed to copy GPU data to host for tensor materialization");
        } else {
          HOLOSCAN_CUDA_CALL_THROW_ERROR(
              cudaMemcpy(dst, tensor->data(), total_bytes, copy_type),
              "Failed to copy GPU data to host for tensor materialization");
        }
        break;
      }
      case kDLCPU:
      case kDLCUDAHost: {
        // For host-accessible memory (CPU, CUDAHost), directly copy the data.
        uint8_t* dst = nullptr;
        data = fbb.CreateUninitializedVector<uint8_t>(total_bytes, &dst);
        const uint8_t* data_ptr = static_cast<const uint8_t*>(tensor->data());

        // Standard memory copy for host-accessible data
        std::memcpy(dst, data_ptr, total_bytes);
        break;
      }
      default: {
        // Unsupported device types (e.g., OpenCL, Vulkan, etc.)
        throw std::runtime_error("Unsupported device type for tensor materialization");
      }
    }
  }

  // Construct and return the final FlatBuffer Tensor object with all serialized components
  return CreateTensor(fbb, data, shape, &dtype, &dl_device, ndim, strides);
}

}  // namespace pipeline_visualization::flatbuffers
