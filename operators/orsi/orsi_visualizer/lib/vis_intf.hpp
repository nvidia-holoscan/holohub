/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

// clang-format off
#define GLFW_INCLUDE_NONE 1
#include <GL/glew.h>
#include <GLFW/glfw3.h>  // NOLINT(build/include_order)
// clang-format on

#include <unordered_map>

#include "holoscan/core/operator_spec.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/allocator.hpp"

namespace holoscan::orsi::vis {

// Copied from Holoscan SDK's Holoviz operator
/// Buffer information, can be initialized either with a tensor or a video buffer
struct BufferInfo {
  enum Type {
    INVALID,
    VIDEO,
    TENSOR
  };

  /**
   * Initialize with tensor
   *
   * @returns error code
   */
  gxf_result_t init(const nvidia::gxf::Handle<nvidia::gxf::Tensor>& tensor) {
    type = TENSOR;
    this->tensor = tensor;

    rank = tensor->rank();
    shape = tensor->shape();
    element_type = tensor->element_type();
    name = tensor.name();
    buffer_ptr = tensor->pointer();
    storage_type = tensor->storage_type();
    bytes_size = tensor->bytes_size();
    for (uint32_t i = 0; i < rank; ++i) {
      stride[i] = tensor->stride(i);
    }

    return GXF_SUCCESS;
  }

  /**
   * Initialize with video buffer
   *
   * @returns error code
   */
  gxf_result_t init(const nvidia::gxf::Handle<nvidia::gxf::VideoBuffer>& video) {
    // NOTE: VideoBuffer::moveToTensor() converts VideoBuffer instance to the Tensor instance
    // with an unexpected shape:  [width, height] or [width, height, num_planes].
    // And, if we use moveToTensor() to convert VideoBuffer to Tensor, we may lose the original
    // video buffer when the VideoBuffer instance is used in other places. For that reason, we
    // directly access internal data of VideoBuffer instance to access Tensor data.
    const auto& buffer_info = video->video_frame_info();

    int32_t channels;
    switch (buffer_info.color_format) {
      case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY:
        element_type = nvidia::gxf::PrimitiveType::kUnsigned8;
        channels = 1;
        break;
      case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY16:
        element_type = nvidia::gxf::PrimitiveType::kUnsigned16;
        channels = 1;
        break;
      case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY32:
        element_type = nvidia::gxf::PrimitiveType::kUnsigned32;
        channels = 1;
        break;
      case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB:
        element_type = nvidia::gxf::PrimitiveType::kUnsigned8;
        channels = 3;
        break;
      case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA:
        element_type = nvidia::gxf::PrimitiveType::kUnsigned8;
        channels = 4;
        break;
      default:
        HOLOSCAN_LOG_ERROR("Unsupported input format");
        return GXF_FAILURE;
    }

    rank = 3;
    shape = nvidia::gxf::Shape{static_cast<int32_t>(buffer_info.height),
                               static_cast<int32_t>(buffer_info.width),
                               channels};
    name = video.name();
    buffer_ptr = video->pointer();
    storage_type = video->storage_type();
    bytes_size = video->size();
    stride[0] = buffer_info.color_planes[0].stride;
    stride[1] = channels;
    stride[2] = PrimitiveTypeSize(element_type);

    video_buffer = video;
    type = VIDEO;
    return GXF_SUCCESS;
  }


  uint32_t rank;
  nvidia::gxf::Shape shape;
  nvidia::gxf::PrimitiveType element_type;
  std::string name;
  const uint8_t * buffer_ptr;
  nvidia::gxf::MemoryStorageType storage_type;
  uint64_t bytes_size;
  nvidia::gxf::Tensor::stride_array_t stride;

  Type type = INVALID;
  nvidia::gxf::Handle<nvidia::gxf::VideoBuffer> video_buffer;
  nvidia::gxf::Handle<nvidia::gxf::Tensor> tensor;
};

class VisIntf {
 public:
    virtual ~VisIntf() = default;

    // Main Holsocan Operator interface
    virtual void setup(holoscan::OperatorSpec& spec) = 0;
    virtual void initialize() = 0;
    virtual void start() = 0;
    virtual void compute(const std::unordered_map<std::string, BufferInfo>& input_buffers) = 0;
    virtual void stop() = 0;

    // GLFW window related callbacks
    virtual void onFramebufferSizeCallback(GLFWwindow* wnd, int width, int height) { }
    virtual void onWindowFocusCallback(GLFWwindow* wnd, int focused) { }
    virtual void onChar(GLFWwindow* wnd, unsigned int codepoint) { }
    virtual void onEnter(GLFWwindow* wnd, int entered) { }
    virtual void onMouseMove(GLFWwindow* wnd, double x, double y) { }
    virtual void onMouseButtonCallback(GLFWwindow* wnd, int button, int action, int mods) { }
    virtual void onScrollCallback(GLFWwindow* wnd, double x, double y) { }
    virtual void onKeyCallback(GLFWwindow* wnd, int key, int scancode, int action, int mods) { }

 protected:
    VisIntf() = default;

    VisIntf(const VisIntf&) = delete;
    VisIntf& operator=(const VisIntf&) = delete;

    VisIntf(const VisIntf&&) = delete;
    VisIntf& operator=(const VisIntf&&) = delete;
};

}  // namespace holoscan::orsi::vis

