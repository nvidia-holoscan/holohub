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

// This file defines functions to serialize and deserialize FlatBuffer Tensor messages to and from
// their native type holoscan::Tensor.
//
// This allows application logic to deal directly with holoscan::Tensor while using the FlatBuffer
// object API, while transparently handling serialization to and from a FlatBuffer when required.
//
// These functions have been been omitted from tensor_generated.h via the native_type annotation.

#include <cstring>
#include <stdexcept>
#include <vector>

#include "holoscan/core/domain/tensor.hpp"

#include "tensor_gather.h"
#include "tensor_materialization.h"
#include "tensor_generated.h"

// CUDA headers are already included by holoscan, so we just need the defines
#define HOLOSCAN_CUDA_CALL(x) x
#define HOLOSCAN_LOG_ERROR(msg) throw std::runtime_error(msg)

namespace isaac {
::flatbuffers::Offset<Tensor> CreateTensor(::flatbuffers::FlatBufferBuilder& _fbb,
                                           const holoscan::Tensor* _o,
                                           const ::flatbuffers::rehasher_function_t*) {
  const auto& dl_ctx = const_cast<holoscan::Tensor*>(_o)->dl_ctx();
  if (!dl_ctx) {
    return 0;
  }

  ::flatbuffers::Offset<::flatbuffers::Vector<int64_t>> shape =
      _o->shape().size() ? _fbb.CreateVector(_o->shape()) : 0;
  DLDataType dtype(static_cast<DLDataTypeCode>(_o->dtype().code), _o->dtype().bits,
                   _o->dtype().lanes);
  DLDevice device(static_cast<DLDeviceType>(_o->device().device_type), _o->device().device_id);
  int ndim = _o->ndim();
  ::flatbuffers::Offset<::flatbuffers::Vector<int64_t>> strides =
      (dl_ctx->tensor.dl_tensor.strides != nullptr && dl_ctx->tensor.dl_tensor.ndim > 0)
          ? _fbb.CreateVector(dl_ctx->tensor.dl_tensor.strides, dl_ctx->tensor.dl_tensor.ndim)
          : 0;
  // Determine materialization mode based on active scopes and tensor size.
  ::flatbuffers::Offset<::flatbuffers::Vector<uint8_t>> data = 0;

  bool should_materialize = false;
  if (WithTensorMaterialization::enabled()) {
    // Explicit materialization scope takes precedence.
    should_materialize = true;
  } else if (WithTensorGather::enabled()) {
    // Use gather mode: materialize if tensor is smaller than threshold.
    size_t tensor_bytes = _o->nbytes();
    should_materialize = (tensor_bytes < WithTensorGather::size_threshold_bytes());
  }

  if (should_materialize) {
    if (_o->data()) {
      // Use tensor->nbytes() to get the actual number of bytes, which handles packed data types
      // correctly. This avoids overshooting for packed dtypes like 1-bit bool or 4-bit INT4.
      size_t total_bytes = _o->nbytes();
      const auto& device = _o->device();

      // Check device type before allocating memory.
      switch (device.device_type) {
        case kDLCUDA:
        case kDLCUDAManaged: {
          uint8_t* dst = nullptr;
          data = _fbb.CreateUninitializedVector<uint8_t>(total_bytes, &dst);
          auto copy_type =
              device.device_type == kDLCUDAManaged ? cudaMemcpyDefault : cudaMemcpyDeviceToHost;
          cudaError_t cuda_result =
              HOLOSCAN_CUDA_CALL(cudaMemcpy(dst, _o->data(), total_bytes, copy_type));
          if (cudaSuccess != cuda_result) {
            HOLOSCAN_LOG_ERROR("Failed to copy GPU data to host for tensor materialization");
            data = 0;
          }
          break;
        }
        case kDLCPU:
        case kDLCUDAHost: {
          // For host-accessible memory (CPU, CUDAHost), directly copy the data.
          uint8_t* dst = nullptr;
          data = _fbb.CreateUninitializedVector<uint8_t>(total_bytes, &dst);
          const uint8_t* data_ptr = static_cast<const uint8_t*>(_o->data());
          std::memcpy(dst, data_ptr, total_bytes);
          break;
        }
        default: {
          HOLOSCAN_LOG_ERROR("Unsupported device type for tensor materialization");
          break;
        }
      }
    }
  }

  return isaac::CreateTensor(_fbb, data, shape, &dtype, &device, ndim, strides);
}

void Tensor::UnPackTo(holoscan::Tensor* _o, const ::flatbuffers::resolver_function_t*) const {
  auto context = std::make_shared<holoscan::DLManagedTensorContext>();

  if (!shape()) {
    throw std::runtime_error("Attempted to unpack a Tensor with no shape");
  }
  context->dl_shape.insert(context->dl_shape.end(), shape()->begin(), shape()->end());
  context->tensor.dl_tensor.shape = context->dl_shape.data();

  if (!dtype()) {
    throw std::runtime_error("Attempted to unpack a Tensor with no dtype");
  }
  context->tensor.dl_tensor.dtype = ::DLDataType{
      .code = dtype()->code(),
      .bits = dtype()->bits(),
      .lanes = dtype()->lanes(),
  };

  // Reading data back into the same device as the original tensor.
  if (!device()) {
    throw std::runtime_error("Attempted to unpack a Tensor with no device");
  }
  context->tensor.dl_tensor.device = ::DLDevice{
      .device_type = static_cast<::DLDeviceType>(device()->device_type()),
      .device_id = device()->device_id(),
  };
  context->tensor.dl_tensor.ndim = ndim();

  if (strides()) {
    context->dl_strides.insert(context->dl_strides.end(), strides()->begin(), strides()->end());
    context->tensor.dl_tensor.strides = context->dl_strides.data();
  } else {
    context->tensor.dl_tensor.strides = nullptr;
  }

  // Copy data into the tensor if available.
  if (data() && data()->size() > 0) {
    const auto& target_device = context->tensor.dl_tensor.device;
    const size_t data_size = data()->size();

    switch (target_device.device_type) {
      case kDLCUDA:
      case kDLCUDAManaged: {
        // Allocate GPU memory and copy data from host to device.
        void* gpu_ptr = nullptr;
        cudaError_t cuda_result = HOLOSCAN_CUDA_CALL(cudaMalloc(&gpu_ptr, data_size));
        if (cudaSuccess != cuda_result) {
          HOLOSCAN_LOG_ERROR("Failed to allocate GPU memory for tensor deserialization");
          context->tensor.dl_tensor.data = nullptr;
          break;
        }

        // Copy data from host (FlatBuffer) to GPU.
        auto copy_type = target_device.device_type == kDLCUDAManaged ? cudaMemcpyDefault
                                                                     : cudaMemcpyHostToDevice;
        cuda_result = HOLOSCAN_CUDA_CALL(cudaMemcpy(gpu_ptr, data()->data(), data_size, copy_type));
        if (cudaSuccess != cuda_result) {
          HOLOSCAN_CUDA_CALL(cudaFree(gpu_ptr));
          HOLOSCAN_LOG_ERROR("Failed to copy data to GPU for tensor deserialization");
          context->tensor.dl_tensor.data = nullptr;
          break;
        }

        // Store GPU pointer and set up proper cleanup via DLPack deleter.
        context->tensor.dl_tensor.data = gpu_ptr;

        // Set up deleter callback to free CUDA memory when tensor is destroyed.
        // TODO(mingxinz): Run this in valgrind to verify that the deleter is actually called.
        context->tensor.deleter = [](::DLManagedTensor* self) {
          if (self->dl_tensor.data != nullptr) {
            cudaFree(self->dl_tensor.data);
          }
          delete static_cast<holoscan::DLManagedTensorContext*>(self->manager_ctx);
        };
        context->tensor.manager_ctx = context.get();
        break;
      }
      case kDLCPU:
      case kDLCUDAHost:
      default: {
        // For CPU and host-accessible memory, copy to host memory.
        auto data_vector = std::make_shared<std::vector<uint8_t>>(data()->begin(), data()->end());
        context->memory_ref = data_vector;
        context->tensor.dl_tensor.data = data_vector->data();

        // TODO(mingxinz): Run this in valgrind to verify that the deleter is actually called.
        context->tensor.deleter = [](::DLManagedTensor* self) {
          auto context = static_cast<holoscan::DLManagedTensorContext*>(self->manager_ctx);
          context->memory_ref.reset();  // Explicitly release data reference.
          delete context;               // Clean up the context.
        };
        context->tensor.manager_ctx = context.get();
        break;
      }
    }
  } else {
    // No data available, set data pointer to nullptr.
    context->tensor.dl_tensor.data = nullptr;
  }

  _o->dl_ctx() = std::move(context);
}

}  // namespace isaac
