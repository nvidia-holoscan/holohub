/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef OPERATORS_ADVANCED_NETWORK_MEDIA_COMMON_FRAME_BUFFER_H_
#define OPERATORS_ADVANCED_NETWORK_MEDIA_COMMON_FRAME_BUFFER_H_

#include <memory>
#include <functional>

#include "rdk/services/services.h"
#include "advanced_network/common.h"
#include "gxf/multimedia/video.hpp"
#include "gxf/core/entity.hpp"
#include "gxf/core/expected.hpp"
#include <holoscan/holoscan.hpp>
#include "video_parameters.h"

using namespace rivermax::dev_kit::services;
using namespace holoscan::advanced_network;

namespace holoscan::ops {

/**
 * @class FrameBufferBase
 * @brief Base class for frame buffers used in media transmission operations.
 */
class FrameBufferBase : public IFrameBuffer {
 public:
  virtual ~FrameBufferBase() = default;

  size_t get_size() const override { return frame_size_; }
  size_t get_aligned_size() const override { return frame_size_; }
  MemoryLocation get_memory_location() const override { return memory_location_; }

 protected:
  /**
   * @brief Converts GXF memory storage type to MemoryLocation enum.
   *
   * @param storage_type The GXF memory storage type to convert.
   * @return The corresponding MemoryLocation.
   */
  inline MemoryLocation from_gxf_memory_type(nvidia::gxf::MemoryStorageType storage_type) const {
    switch (storage_type) {
      case nvidia::gxf::MemoryStorageType::kHost:
      case nvidia::gxf::MemoryStorageType::kSystem:
        return MemoryLocation::Host;
      case nvidia::gxf::MemoryStorageType::kDevice:
        return MemoryLocation::GPU;
      default:
        return MemoryLocation::Host;
    }
  }

 protected:
  MemoryLocation memory_location_;
  nvidia::gxf::MemoryStorageType src_storage_type_;
  size_t frame_size_;
  nvidia::gxf::Entity entity_;
};

/**
 * @class VideoFrameBufferBase
 * @brief Base class for video frame buffers with common validation functionality.
 */
class VideoFrameBufferBase : public FrameBufferBase {
 public:
  virtual ~VideoFrameBufferBase() = default;

  /**
   * @brief Validates the frame buffer against expected parameters.
   *
   * @param expected_width Expected frame width.
   * @param expected_height Expected frame height.
   * @param expected_frame_size Expected frame size in bytes.
   * @param expected_format Expected video format.
   * @return Status indicating validation result.
   */
  Status validate_frame_parameters(uint32_t expected_width, uint32_t expected_height,
                                   size_t expected_frame_size,
                                   nvidia::gxf::VideoFormat expected_format) const;

 protected:
  /**
   * @brief Implementation-specific validation logic to be defined by derived classes.
   *
   * @param fmt The expected video format.
   * @return Status indicating validation result.
   */
  virtual Status validate_format_compliance(nvidia::gxf::VideoFormat fmt) const = 0;

 protected:
  static constexpr uint32_t SMPTE_STRIDE_ALIGNMENT = 256;
  static constexpr uint32_t SMPTE_420_ALIGNMENT = 2;

  uint32_t width_;
  uint32_t height_;
  nvidia::gxf::VideoFormat format_;
};

/**
 * @class VideoBufferFrameBuffer
 * @brief Frame buffer implementation for VideoBuffer type frames.
 */
class VideoBufferFrameBuffer : public VideoFrameBufferBase {
 public:
  /**
   * @brief Constructs a VideoBufferFrameBuffer from a GXF entity.
   *
   * @param entity The GXF entity containing the video buffer.
   */
  explicit VideoBufferFrameBuffer(nvidia::gxf::Entity entity);
  byte_t* get() const override {
    return (buffer_) ? static_cast<byte_t*>(buffer_->pointer()) : nullptr;
  }

 protected:
  Status validate_format_compliance(nvidia::gxf::VideoFormat fmt) const override;

 private:
  nvidia::gxf::Handle<nvidia::gxf::VideoBuffer> buffer_;
  std::vector<nvidia::gxf::ColorPlane> planes_;
};

/**
 * @class TensorFrameBuffer
 * @brief Frame buffer implementation for Tensor type frames.
 */
class TensorFrameBuffer : public VideoFrameBufferBase {
 public:
  /**
   * @brief Constructs a TensorFrameBuffer from a GXF entity with a specific format.
   *
   * @param entity The GXF entity containing the tensor.
   * @param format The video format to interpret the tensor as.
   */
  TensorFrameBuffer(nvidia::gxf::Entity entity, nvidia::gxf::VideoFormat format);
  virtual ~TensorFrameBuffer() = default;
  byte_t* get() const override {
    return (tensor_) ? static_cast<byte_t*>(tensor_->pointer()) : nullptr;
  }

 protected:
  Status validate_format_compliance(nvidia::gxf::VideoFormat fmt) const override;

 private:
  nvidia::gxf::Handle<nvidia::gxf::Tensor> tensor_;
};

/**
 * @class AllocatedVideoBufferFrameBuffer
 * @brief Frame buffer implementation for pre-allocated memory buffers.
 *
 * Used primarily by the RX operator for receiving frames.
 */
class AllocatedVideoBufferFrameBuffer : public VideoFrameBufferBase {
 public:
  /**
   * @brief Constructs an AllocatedViddeoBufferFrameBuffer from pre-allocated memory.
   *
   * @param data Pointer to the allocated memory
   * @param size Size of the allocated memory in bytes
   * @param width Frame width
   * @param height Frame height
   * @param format Video format
   * @param storage_type Memory storage type (device or host)
   */
  AllocatedVideoBufferFrameBuffer(
      void* data, size_t size, uint32_t width, uint32_t height, nvidia::gxf::VideoFormat format,
      nvidia::gxf::MemoryStorageType storage_type = nvidia::gxf::MemoryStorageType::kDevice);
  virtual ~AllocatedVideoBufferFrameBuffer() = default;

  byte_t* get() const override { return static_cast<byte_t*>(data_); }

  /**
   * @brief Creates a GXF entity containing this frame's data.
   *
   * @param context GXF context for entity creation
   * @param release_func Function to call when the entity is released
   * @return Created GXF entity
   */
  nvidia::gxf::Entity wrap_in_entity(
      void* context, std::function<nvidia::gxf::Expected<void>(void*)> release_func);

 protected:
  Status validate_format_compliance(nvidia::gxf::VideoFormat fmt) const override;

 private:
  void* data_;
};

/**
 * @class AllocatedTensorFrameBuffer
 * @brief Frame buffer implementation for pre-allocated tensor memory buffers.
 *
 * Used primarily by the RX operator for receiving frames in tensor format.
 */
class AllocatedTensorFrameBuffer : public VideoFrameBufferBase {
 public:
  /**
   * @brief Constructs an AllocatedTensorFrameBuffer from pre-allocated memory.
   *
   * @param data Pointer to the allocated memory
   * @param size Size of the allocated memory in bytes
   * @param width Frame width
   * @param height Frame height
   * @param channels Number of channels (typically 3 for RGB)
   * @param format Video format
   * @param storage_type Memory storage type (device or host)
   */
  AllocatedTensorFrameBuffer(
      void* data, size_t size, uint32_t width, uint32_t height, uint32_t channels,
      nvidia::gxf::VideoFormat format,
      nvidia::gxf::MemoryStorageType storage_type = nvidia::gxf::MemoryStorageType::kDevice);
  virtual ~AllocatedTensorFrameBuffer() = default;

  byte_t* get() const override { return static_cast<byte_t*>(data_); }

  /**
   * @brief Creates a GXF entity containing this frame's data as a tensor.
   *
   * @param context GXF context for entity creation
   * @param release_func Function to call when the entity is released
   * @return Created GXF entity
   */
  nvidia::gxf::Entity wrap_in_entity(
      void* context, std::function<nvidia::gxf::Expected<void>(void*)> release_func);

 protected:
  Status validate_format_compliance(nvidia::gxf::VideoFormat fmt) const override;

 private:
  void* data_;
  uint32_t channels_;
};

}  // namespace holoscan::ops

#endif  // OPERATORS_ADVANCED_NETWORK_MEDIA_COMMON_FRAME_BUFFER_H_
