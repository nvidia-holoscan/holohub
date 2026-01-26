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

namespace holoscan::ops::ucxx {

namespace {

// Map DLDataType to nvidia::gxf::PrimitiveType
PrimitiveType dlTypeToPrimitive(DLDataType dtype) {
  if (dtype.lanes != 1) { return PrimitiveType::kCustom; }
  switch (dtype.code) {
    case kDLUInt:
      if (dtype.bits == 8) return PrimitiveType::kUnsigned8;
      if (dtype.bits == 16) return PrimitiveType::kUnsigned16;
      if (dtype.bits == 32) return PrimitiveType::kUnsigned32;
      if (dtype.bits == 64) return PrimitiveType::kUnsigned64;
      break;
    case kDLInt:
      if (dtype.bits == 8) return PrimitiveType::kInt8;
      if (dtype.bits == 16) return PrimitiveType::kInt16;
      if (dtype.bits == 32) return PrimitiveType::kInt32;
      if (dtype.bits == 64) return PrimitiveType::kInt64;
      break;
    case kDLFloat:
      if (dtype.bits == 16) return PrimitiveType::kFloat16;
      if (dtype.bits == 32) return PrimitiveType::kFloat32;
      if (dtype.bits == 64) return PrimitiveType::kFloat64;
      break;
    default: break;
  }
  return PrimitiveType::kCustom;
}

// Map DLDevice to nvidia::gxf::MemoryStorageType
MemoryStorageType dlDeviceToStorage(DLDevice device) {
  switch (device.device_type) {
    case kDLCUDA:
    case kDLCUDAManaged:
      return MemoryStorageType::kDevice;
    default:
      return MemoryStorageType::kHost;
  }
}

// Compute the memory span (in bytes) from data pointer to end of last element.
// Handles contiguous, padded, and permuted positive-stride layouts.
// Header strides must be in bytes. Returns 0 if any dimension is non-positive.
size_t computeMemorySpan(const TensorHeader& header) {
  if (header.rank == 0) { return header.bytes_per_element; }
  if (header.rank > Shape::kMaxRank) { return 0; }
  size_t span = 0;
  for (uint32_t i = 0; i < header.rank; ++i) {
    if (header.dims[i] <= 0) { return 0; }
    span += static_cast<size_t>(header.dims[i] - 1) * header.strides[i];
  }
  return span + header.bytes_per_element;
}

}  // namespace

TensorHeader buildTensorHeader(const nvidia::gxf::Tensor& tensor) {
  TensorHeader header{};
  header.storage_type = tensor.storage_type();
  header.element_type = tensor.element_type();
  header.bytes_per_element = tensor.bytes_per_element();
  if (tensor.rank() > Shape::kMaxRank) { return header; }
  header.rank = tensor.rank();
  for (uint32_t i = 0; i < header.rank; ++i) {
    header.dims[i] = tensor.shape().dimension(i);
    header.strides[i] = tensor.stride(i);
  }
  return header;
}

TensorHeader buildTensorHeader(const holoscan::Tensor& tensor) {
  TensorHeader header{};
  header.storage_type = dlDeviceToStorage(tensor.device());
  header.element_type = dlTypeToPrimitive(tensor.dtype());
  header.bytes_per_element = (tensor.dtype().bits * tensor.dtype().lanes + 7) / 8;
  if (tensor.ndim() < 0 || static_cast<uint32_t>(tensor.ndim()) > Shape::kMaxRank) {
    return header;
  }
  header.rank = static_cast<uint32_t>(tensor.ndim());
  for (uint32_t i = 0; i < header.rank; ++i) {
    header.dims[i] = static_cast<int32_t>(tensor.shape()[i]);
    // DLPack strides are int64_t; reject negative strides (flipped views not supported)
    if (tensor.strides()[i] < 0) {
      header.dims[0] = 0;
      return header;
    }
    header.strides[i] = static_cast<uint64_t>(tensor.strides()[i]) * header.bytes_per_element;
  }
  return header;
}

TensorHeader buildTensorHeader(const nvidia::gxf::VideoBuffer& video_buffer) {
  const auto& info = video_buffer.video_frame_info();

  uint32_t channels = 0;
  switch (info.color_format) {
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA: channels = 4; break;
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB:  channels = 3; break;
    default: break;  // channels stays 0, computeMemorySpan returns 0 (dims <= 0)
  }

  TensorHeader header{};
  header.storage_type = video_buffer.storage_type();
  header.element_type = PrimitiveType::kUnsigned8;
  header.bytes_per_element = 1;
  header.rank = 3;
  header.dims[0] = static_cast<int32_t>(info.height);
  header.dims[1] = static_cast<int32_t>(info.width);
  header.dims[2] = static_cast<int32_t>(channels);
  const auto& plane = info.color_planes[0];
  header.strides[0] = plane.stride;
  header.strides[1] = channels;  // bytes per pixel
  header.strides[2] = 1;
  return header;
}

std::optional<BufferDescriptor> resolveEntityBuffer(
    holoscan::gxf::Entity& entity, const char* tensor_name) {
  // 1. Try holoscan::Tensor (DLPack, data owned by entity)
  auto maybe_hl_tensor = entity.get<holoscan::Tensor>(tensor_name, false);
  if (maybe_hl_tensor) {
    auto& t = *maybe_hl_tensor;
    TensorHeader header = buildTensorHeader(t);
    size_t size = computeMemorySpan(header);
    if (size == 0) { return std::nullopt; }
    return BufferDescriptor{header, t.data(), size};
  }

  // 2. Try nvidia::gxf::Tensor (owned by entity)
  auto& gxf_entity = static_cast<nvidia::gxf::Entity&>(entity);
  auto maybe_tensor = gxf_entity.get<nvidia::gxf::Tensor>(tensor_name);
  if (maybe_tensor) {
    auto* t = maybe_tensor.value().get();
    TensorHeader header = buildTensorHeader(*t);
    size_t size = computeMemorySpan(header);
    if (size == 0) { return std::nullopt; }
    return BufferDescriptor{header, t->pointer(), size};
  }

  // 3. Try nvidia::gxf::VideoBuffer (packed RGB/RGBA, owned by entity)
  auto maybe_vb = gxf_entity.get<nvidia::gxf::VideoBuffer>();
  if (maybe_vb) {
    auto* vb = maybe_vb.value().get();
    TensorHeader header = buildTensorHeader(*vb);
    size_t size = computeMemorySpan(header);
    if (size == 0) { return std::nullopt; }
    return BufferDescriptor{header, vb->pointer(), size};
  }

  return std::nullopt;
}

}  // namespace holoscan::ops::ucxx
