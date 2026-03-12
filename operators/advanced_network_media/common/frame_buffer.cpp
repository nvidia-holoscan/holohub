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

#include "frame_buffer.h"
#include "adv_network_media_logging.h"

namespace holoscan::ops {

Status VideoFrameBufferBase::validate_frame_parameters(
    uint32_t expected_width, uint32_t expected_height, size_t expected_frame_size,
    nvidia::gxf::VideoFormat expected_format) const {
  if (width_ != expected_width || height_ != expected_height) {
    ANM_LOG_ERROR(
        "Resolution mismatch: {}x{} vs {}x{}", width_, height_, expected_width, expected_height);
    return Status::INVALID_PARAMETER;
  }

  if (frame_size_ != expected_frame_size) {
    ANM_LOG_ERROR("Frame size mismatch: {} vs {}", frame_size_, expected_frame_size);
    return Status::INVALID_PARAMETER;
  }

  return validate_format_compliance(expected_format);
}

VideoBufferFrameBuffer::VideoBufferFrameBuffer(nvidia::gxf::Entity entity) {
  entity_ = std::move(entity);

  auto maybe_video_buffer = entity_.get<nvidia::gxf::VideoBuffer>();
  if (!maybe_video_buffer) throw std::runtime_error("Entity doesn't contain a video buffer");

  buffer_ = maybe_video_buffer.value();
  const auto& info = buffer_->video_frame_info();
  width_ = info.width;
  height_ = info.height;
  src_storage_type_ = buffer_->storage_type();
  memory_location_ = from_gxf_memory_type(src_storage_type_);
  frame_size_ = buffer_->size();
  format_ = info.color_format;
  planes_ = info.color_planes;
}

Status VideoBufferFrameBuffer::validate_format_compliance(
    nvidia::gxf::VideoFormat expected_format) const {
  if (expected_format == nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_709) {
    if (format_ != nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_709 &&
        format_ != nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_YUV420_709) {
      ANM_LOG_ERROR("Invalid NV12_709 format");
      return Status::INVALID_PARAMETER;
    }
  } else if (expected_format == nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB) {
    if (format_ != nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB) {
      ANM_LOG_ERROR("Invalid RGB format");
      return Status::INVALID_PARAMETER;
    }
  } else if (format_ != nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_CUSTOM) {
    ANM_LOG_ERROR(
        "Format mismatch: {} vs {}", static_cast<int>(format_), static_cast<int>(expected_format));
    return Status::INVALID_PARAMETER;
  }

  if (expected_format == nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_709) {
    if (width_ % SMPTE_420_ALIGNMENT != 0 || height_ % SMPTE_420_ALIGNMENT != 0) {
      ANM_LOG_ERROR("Resolution not 4:2:0 aligned");
      return Status::INVALID_PARAMETER;
    }
  }

  for (const auto& plane : planes_) {
    if (plane.stride % SMPTE_STRIDE_ALIGNMENT != 0) {
      ANM_LOG_ERROR("Stride {} not {}-byte aligned", plane.stride, SMPTE_STRIDE_ALIGNMENT);
      return Status::INVALID_PARAMETER;
    }
  }

  return Status::SUCCESS;
}

TensorFrameBuffer::TensorFrameBuffer(nvidia::gxf::Entity entity, nvidia::gxf::VideoFormat format) {
  entity_ = std::move(entity);

  auto maybe_tensor = entity_.get<nvidia::gxf::Tensor>();
  if (!maybe_tensor) throw std::runtime_error("Entity doesn't contain a tensor");
  tensor_ = maybe_tensor.value();

  const auto& shape = tensor_->shape();
  width_ = shape.dimension(1);
  height_ = shape.dimension(0);
  src_storage_type_ = tensor_->storage_type();
  memory_location_ = from_gxf_memory_type(src_storage_type_);
  frame_size_ = tensor_->size();
  format_ = format;
}

Status TensorFrameBuffer::validate_format_compliance(
    nvidia::gxf::VideoFormat expected_format) const {
  const auto& shape = tensor_->shape();
  switch (format_) {
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_709:
      if (shape.rank() != 3 || shape.dimension(2) != 2 ||
          tensor_->element_type() != nvidia::gxf::PrimitiveType::kUnsigned8) {
        ANM_LOG_ERROR("Invalid NV12_709 tensor");
        return Status::INVALID_PARAMETER;
      }
      break;

    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB:
      if (shape.rank() != 3 || shape.dimension(2) != 3) {
        ANM_LOG_ERROR("Invalid RGB tensor");
        return Status::INVALID_PARAMETER;
      }
      break;

    default:
      ANM_LOG_ERROR("Unsupported tensor format: {}", static_cast<int>(format_));
      return Status::INVALID_PARAMETER;
  }
  return Status::SUCCESS;
}

AllocatedVideoBufferFrameBuffer::AllocatedVideoBufferFrameBuffer(
    void* data, size_t size, uint32_t width, uint32_t height, nvidia::gxf::VideoFormat format,
    nvidia::gxf::MemoryStorageType storage_type) {
  data_ = data;
  frame_size_ = size;
  width_ = width;
  height_ = height;
  format_ = format;
  src_storage_type_ = storage_type;
  memory_location_ = from_gxf_memory_type(storage_type);
}

Status AllocatedVideoBufferFrameBuffer::validate_format_compliance(
    nvidia::gxf::VideoFormat expected_format) const {
  if (format_ != expected_format) {
    ANM_LOG_ERROR(
        "Format mismatch: {} vs {}", static_cast<int>(format_), static_cast<int>(expected_format));
    return Status::INVALID_PARAMETER;
  }
  return Status::SUCCESS;
}

nvidia::gxf::Entity AllocatedVideoBufferFrameBuffer::wrap_in_entity(
    void* context, std::function<nvidia::gxf::Expected<void>(void*)> release_func) {
  auto result = nvidia::gxf::Entity::New(context);
  if (!result) { throw std::runtime_error("Failed to allocate entity"); }

  auto buffer = result.value().add<nvidia::gxf::VideoBuffer>();
  if (!buffer) { throw std::runtime_error("Failed to allocate video buffer"); }

  // Set up video buffer based on format
  nvidia::gxf::VideoBufferInfo info;

  switch (format_) {
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB: {
      nvidia::gxf::VideoTypeTraits<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB> video_type;
      nvidia::gxf::VideoFormatSize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB> color_format;
      auto color_planes = color_format.getDefaultColorPlanes(width_, height_, false);
      info = {width_,
              height_,
              video_type.value,
              color_planes,
              nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR};
      break;
    }
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_709: {
      nvidia::gxf::VideoTypeTraits<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_709> video_type;
      nvidia::gxf::VideoFormatSize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_709>
          color_format;
      auto color_planes = color_format.getDefaultColorPlanes(width_, height_, false);
      info = {width_,
              height_,
              video_type.value,
              color_planes,
              nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR};
      break;
    }
    default:
      throw std::runtime_error("Unsupported video format");
  }

  buffer.value()->wrapMemory(info, frame_size_, src_storage_type_, data_, release_func);

  return result.value();
}

AllocatedTensorFrameBuffer::AllocatedTensorFrameBuffer(
    void* data, size_t size, uint32_t width, uint32_t height, uint32_t channels,
    nvidia::gxf::VideoFormat format, nvidia::gxf::MemoryStorageType storage_type) {
  data_ = data;
  frame_size_ = size;
  width_ = width;
  height_ = height;
  channels_ = channels;
  format_ = format;
  src_storage_type_ = storage_type;
  memory_location_ = from_gxf_memory_type(storage_type);
}

Status AllocatedTensorFrameBuffer::validate_format_compliance(
    nvidia::gxf::VideoFormat expected_format) const {
  if (format_ != expected_format) {
    ANM_LOG_ERROR(
        "Format mismatch: {} vs {}", static_cast<int>(format_), static_cast<int>(expected_format));
    return Status::INVALID_PARAMETER;
  }

  // Validate channel count based on format
  if (expected_format == nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB && channels_ != 3) {
    ANM_LOG_ERROR("Invalid channel count for RGB format: {}", channels_);
    return Status::INVALID_PARAMETER;
  } else if (expected_format == nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_709 &&
             channels_ != 2) {
    ANM_LOG_ERROR("Invalid channel count for NV12_709 format: {}", channels_);
    return Status::INVALID_PARAMETER;
  }

  return Status::SUCCESS;
}

nvidia::gxf::Entity AllocatedTensorFrameBuffer::wrap_in_entity(
    void* context, std::function<nvidia::gxf::Expected<void>(void*)> release_func) {
  auto result = nvidia::gxf::Entity::New(context);
  if (!result) { throw std::runtime_error("Failed to allocate entity"); }

  auto tensor = result.value().add<nvidia::gxf::Tensor>();
  if (!tensor) { throw std::runtime_error("Failed to allocate tensor"); }

  // Set up tensor shape based on format
  nvidia::gxf::Shape shape;
  auto element_type = nvidia::gxf::PrimitiveType::kUnsigned8;

  switch (format_) {
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB:
      shape = {static_cast<int32_t>(height_), static_cast<int32_t>(width_), 3};
      break;
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_709:
      // For NV12, we need 2 channels (Y and UV interleaved)
      shape = {static_cast<int32_t>(height_), static_cast<int32_t>(width_), 2};
      break;
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_CUSTOM:
      // For custom format, use the channels_ value provided
      shape = {static_cast<int32_t>(height_),
               static_cast<int32_t>(width_),
               static_cast<int32_t>(channels_)};
      break;
    default:
      throw std::runtime_error("Unsupported tensor format");
  }

  auto element_size = nvidia::gxf::PrimitiveTypeSize(element_type);
  auto strides = nvidia::gxf::ComputeTrivialStrides(shape, element_size);

  tensor.value()->wrapMemory(
      shape, element_type, element_size, strides, src_storage_type_, data_, release_func);

  return result.value();
}

}  // namespace holoscan::ops
