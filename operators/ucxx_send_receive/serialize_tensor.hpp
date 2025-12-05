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

#pragma once

#include <cstdint>
#include <cstddef>

#include "holoscan/holoscan.hpp"
#include "gxf/std/tensor.hpp"

namespace holoscan::ops::ucxx {

using nvidia::gxf::MemoryStorageType;
using nvidia::gxf::PrimitiveType;
using nvidia::gxf::Shape;

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
// Based on nvidia::gxf::StdComponentSerializer::serializeTensor.
holoscan::expected<size_t, holoscan::RuntimeError> serializeTensor(
    const nvidia::gxf::Tensor& tensor,
    uint8_t* buffer,
    size_t buffer_size,
    holoscan::Allocator* allocator);

// Deserializes a tensor object from the given buffer.
//
// This function reconstructs a nvidia::gxf::Tensor from a raw binary buffer
// that contains a serialized TensorHeader followed by contiguous tensor data.
// The function performs sanity checks on the header, copies shape/stride
// information, allocates the tensor using the provided holoscan::Allocator,
// and copies the tensor data based on storage type.
//
// Args:
//   buffer: Pointer to the start of the serialized tensor buffer.
//   buffer_size: Size of the buffer in bytes.
//   context: GXF context for creating allocator handles.
//   allocator: Pointer to an allocator to use for tensor memory.
//
// Returns:
//   holoscan::expected<nvidia::gxf::Tensor, holoscan::RuntimeError> containing
//   the deserialized tensor, or an error if deserialization fails.
//
// Based on nvidia::gxf::StdComponentSerializer::deserializeTensor.
holoscan::expected<nvidia::gxf::Tensor, holoscan::RuntimeError> deserializeTensor(
    const uint8_t* buffer,
    size_t buffer_size,
    gxf_context_t context,
    holoscan::Allocator* allocator);

}  // namespace holoscan::ops::ucxx
