/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/operator_spec.hpp"

#include "tensor_to_video_buffer.hpp"

namespace holoscan::ops {

static nvidia::gxf::VideoFormat toVideoFormat(const std::string& str) {
  if (str == "rgb") {
    return nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB;
  } else if (str == "rgba") {
    return nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA;
  } else if (str == "bgra") {
    return nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_BGRA;
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
  const auto format = video_format_.get();
  video_format_type_ = toVideoFormat(format);
  if (video_format_type_ == nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_CUSTOM) {
    throw std::runtime_error(
        fmt::format("Unsupported video_format '{}'. Expected one of: rgb, rgba, bgra, yuv420",
                    format));
  }
}

void TensorToVideoBufferOp::compute(InputContext& op_input, OutputContext& op_output,
                                    ExecutionContext& /*context*/) {
  auto in_message = op_input.receive<gxf::Entity>("in_tensor").value();

  // Intentionally receive the input CUDA stream.
  // The returned value is not used directly; this call makes the framework track
  // the stream dependency for the zero-copy entity emitted downstream.
  [[maybe_unused]] auto input_cuda_stream = op_input.receive_cuda_stream("in_tensor");

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

  if (in_tensor->shape().rank() < 3) {
    throw std::runtime_error(
        fmt::format("Tensor '{}' rank must be >= 3 for HxWxC access", in_tensor_name));
  }

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

  if (in_primitive_type != nvidia::gxf::PrimitiveType::kUnsigned8) {
    throw std::runtime_error("Only supports uint8 input tensor");
  }

  // Zero-copy: attach a VideoBuffer to the SAME incoming entity so the wrapped
  // tensor memory remains owned/alive for downstream consumers.
  auto& gxf_in_message = static_cast<nvidia::gxf::Entity&>(in_message);
  auto buffer = gxf_in_message.add<nvidia::gxf::VideoBuffer>();
  if (!buffer) { throw std::runtime_error("Failed to allocate video buffer; terminating."); }

  auto in_tensor_ptr = static_cast<uint8_t*>(in_tensor_data);
  switch (video_format_type_) {
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_YUV420: {
      if (in_channels != 3) {
        throw std::runtime_error("YUV420 path expects a 3-channel input tensor view");
      }
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
      if (in_channels != 3) {
        throw std::runtime_error("RGB video format requires 3-channel input tensor");
      }
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
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA: {
      if (in_channels != 4) {
        throw std::runtime_error("RGBA video format requires 4-channel input tensor");
      }
      nvidia::gxf::VideoTypeTraits<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA> video_type;
      nvidia::gxf::VideoFormatSize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA> color_format;
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
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_BGRA: {
      if (in_channels != 4) {
        throw std::runtime_error("BGRA video format requires 4-channel input tensor");
      }
      nvidia::gxf::VideoTypeTraits<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_BGRA> video_type;
      nvidia::gxf::VideoFormatSize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_BGRA> color_format;
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

  // Re-emit the same entity for zero-copy. The input CUDA stream is received
  // above so the framework can track the intended stream dependency/order.
  op_output.emit(in_message, "out_video_buffer");
}

}  // namespace holoscan::ops
