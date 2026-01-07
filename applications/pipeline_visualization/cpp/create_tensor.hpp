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

#ifndef PIPELINE_VISUALIZATION_CPP_CREATE_TENSOR_HPP
#define PIPELINE_VISUALIZATION_CPP_CREATE_TENSOR_HPP

#include <memory>
#include <optional>

#include <flatbuffers/tensor_generated.h>
#include <cuda_runtime.h>

namespace holoscan {
// Forward declaration of Holoscan Tensor class
class Tensor;
}  // namespace holoscan

namespace pipeline_visualization::flatbuffers {

/**
 * @brief Creates a FlatBuffers Tensor from a Holoscan Tensor
 *
 * This function serializes a Holoscan Tensor into a FlatBuffers format for efficient
 * data transmission and storage. It handles tensor metadata (shape, data type, device info)
 * and optionally materializes the tensor data depending on the device type.
 *
 * The function supports tensors on different device types:
 * - CPU tensors: Direct memory copy
 * - CUDA device tensors: Device-to-host memory copy using CUDA APIs
 * - CUDA managed memory: Uses cudaMemcpyDefault for optimal transfer
 * - CUDA host memory: Direct memory copy for pinned host memory
 *
 * @param fbb FlatBufferBuilder instance used to construct the serialized tensor
 * @param tensor Shared pointer to the Holoscan Tensor to be serialized
 * @param stream Optional CUDA stream for GPU operations
 * @return FlatBuffers offset to the created Tensor object, or 0 if tensor is invalid
 *
 * @throws std::runtime_error If an unsupported device type is encountered
 * @throws CUDA runtime errors If CUDA memory operations fail
 *
 * @note Data materialization only occurs when tensor->data() is available and
 *       WithTensorMaterialization scope is enabled
 */
::flatbuffers::Offset<Tensor> CreateTensor(::flatbuffers::FlatBufferBuilder& fbb,
                                           const std::shared_ptr<holoscan::Tensor>& tensor,
                                           std::optional<cudaStream_t> stream);

}  // namespace pipeline_visualization::flatbuffers

#endif  // PIPELINE_VISUALIZATION_CPP_CREATE_TENSOR_HPP
