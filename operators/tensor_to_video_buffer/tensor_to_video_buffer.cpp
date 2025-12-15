/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include <utility>

// If GXF has gxf/std/dlpack_utils.hpp it has DLPack support
#if __has_include("gxf/std/dlpack_utils.hpp")
  #define GXF_HAS_DLPACK_SUPPORT 1
  #include "gxf/std/tensor.hpp"
#else
  #define GXF_HAS_DLPACK_SUPPORT 0
  #include "holoscan/core/gxf/gxf_tensor.hpp"
#endif

#include "gxf/multimedia/video.hpp"

#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator_spec.hpp"

#include "tensor_to_video_buffer.hpp"

namespace holoscan::ops {

static nvidia::gxf::VideoFormat toVideoFormat(const std::string& str) {
  if (str == "rgb") {
    return nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB;
  } else if (str == "yuv420") {
    return nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_YUV420;
  } else {
    return nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_CUSTOM;
  }
}

void TensorToVideoBufferOp::setup(OperatorSpec& spec) {
  auto& input = spec.input<gxf::Entity>("in_tensor");
  auto& output = spec.output<gxf::Entity>("out_video_buffer");

  spec.param(data_in_, "data_in", "DataIn", "Data in Holoscan format", &input);
  spec.param(in_tensor_name_,
             "in_tensor_name",
             "InputTensorName",
             "Name of the input tensor.",
             std::string(""));
  spec.param(video_format_, "video_format", "VideoFormat", "Video format", std::string(""));
  spec.param(data_out_, "data_out", "DataOut", "Data in GXF format", &output);
}

void TensorToVideoBufferOp::start() {
  video_format_type_ = toVideoFormat(video_format_);
}

void TensorToVideoBufferOp::compute(InputContext& op_input, OutputContext& op_output,
                                    ExecutionContext& context) {
  // Process input message
  // The type of `in_message` is 'holoscan::gxf::Entity'.
  auto in_message = op_input.receive<gxf::Entity>("in_tensor").value();

  const std::string in_tensor_name = in_tensor_name_.get();

  // Get tensor attached to the message
  // The type of `maybe_tensor` is 'std::shared_ptr<holoscan::Tensor>'.
  auto maybe_tensor = in_message.get<Tensor>(in_tensor_name.c_str());
  if (!maybe_tensor) {
    maybe_tensor = in_message.get<Tensor>();
    if (!maybe_tensor) {
      throw std::runtime_error(fmt::format("Tensor '{}' not found in message", in_tensor_name));
    }
  }
  #if GXF_HAS_DLPACK_SUPPORT
    // Get a std::shared_ptr<nvidia::gxf::Tensor> from std::shared_ptr<holoscan::Tensor>'.
    auto in_tensor = std::make_shared<nvidia::gxf::Tensor>(maybe_tensor->dl_ctx());
  #else
    // Get a std::shared_ptr<holoscan::gxf::GXFTensor> from std::shared_ptr<holoscan::Tensor>'.
    auto in_tensor = gxf::GXFTensor::from_tensor(maybe_tensor);
  #endif

  nvidia::gxf::Shape out_shape{0, 0, 0};
  void* in_tensor_data = nullptr;
  nvidia::gxf::PrimitiveType in_primitive_type = nvidia::gxf::PrimitiveType::kCustom;
  nvidia::gxf::MemoryStorageType in_memory_storage_type = nvidia::gxf::MemoryStorageType::kHost;
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  uint16_t in_channels = 0;

  // Get needed information from the GXF tensor
  in_tensor_data = in_tensor->pointer();
  in_primitive_type = in_tensor->element_type();
  in_memory_storage_type = in_tensor->storage_type();
  height_ = in_tensor->shape().dimension(0);
  width_ = in_tensor->shape().dimension(1);
  in_channels = in_tensor->shape().dimension(2);

  if (in_memory_storage_type != nvidia::gxf::MemoryStorageType::kDevice) {
    throw std::runtime_error(
        fmt::format("Tensor '{}' or VideoBuffer is not allocated on device", in_tensor_name));
  }

  // Process image only if the input image is 3 channel image
  if (in_primitive_type != nvidia::gxf::PrimitiveType::kUnsigned8 || in_channels != 3) {
    throw std::runtime_error("Only supports 3 channel input tensor");
  }

  // Create and pass the GXF video buffer downstream.
  auto out_message = nvidia::gxf::Entity::New(context.context());
  if (!out_message) {
    throw std::runtime_error("Failed to allocate message; terminating.");
  }

  auto buffer = out_message.value().add<nvidia::gxf::VideoBuffer>();
  if (!buffer) {
    throw std::runtime_error("Failed to allocate video buffer; terminating.");
  }

  auto in_tensor_ptr = static_cast<uint8_t*>(in_tensor_data);
  switch (video_format_type_) {
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_YUV420: {
      nvidia::gxf::VideoTypeTraits<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_YUV420> video_type;
      nvidia::gxf::VideoFormatSize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_YUV420> color_format;
      auto color_planes = color_format.getDefaultColorPlanes(width_, height_);
      nvidia::gxf::VideoBufferInfo info{
          width_,
          height_,
          video_type.value,
          color_planes,
          nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR};
      auto storage_type = nvidia::gxf::MemoryStorageType::kDevice;
      auto size = width_ * height_ * in_channels;
      buffer.value()->wrapMemory(info, size, storage_type, in_tensor_ptr, nullptr);
      break;
    }
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB: {
      nvidia::gxf::VideoTypeTraits<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB> video_type;
      nvidia::gxf::VideoFormatSize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB> color_format;
      auto color_planes = color_format.getDefaultColorPlanes(width_, height_);
      nvidia::gxf::VideoBufferInfo info{
          width_,
          height_,
          video_type.value,
          color_planes,
          nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR};
      auto storage_type = nvidia::gxf::MemoryStorageType::kDevice;
      auto size = width_ * height_ * in_channels;
      buffer.value()->wrapMemory(info, size, storage_type, in_tensor_ptr, nullptr);
      break;
    }
    default:
      throw std::runtime_error("Unsupported video format");
      break;
  }

  // Transmit the gxf video buffer to target
  auto result = gxf::Entity(std::move(out_message.value()));
  op_output.emit(result);
}

}  // namespace holoscan::ops
