/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "qt_video_op.hpp"

#include <gxf/cuda/cuda_stream_id.hpp>
#include <holoscan/holoscan.hpp>

#include "qt_holoscan_video.hpp"

#define CUDA_TRY(stmt)                                                                        \
  ({                                                                                          \
    cudaError_t _holoscan_cuda_err = stmt;                                                    \
    if (cudaSuccess != _holoscan_cuda_err) {                                                  \
      HOLOSCAN_LOG_ERROR("CUDA Runtime call {} in line {} of file {} failed with '{}' ({}).", \
                         #stmt,                                                               \
                         __LINE__,                                                            \
                         __FILE__,                                                            \
                         cudaGetErrorString(_holoscan_cuda_err),                              \
                         static_cast<int>(_holoscan_cuda_err));                               \
    }                                                                                         \
    _holoscan_cuda_err;                                                                       \
  })

/**
 * Dummy YAML convert function for QtHoloscanVideo*
 */
template <>
struct YAML::convert<QtHoloscanVideo*> {
  static Node encode(const QtHoloscanVideo*& data) {
    holoscan::log_error("YAML conversion not supported");
    return Node();
  }

  static bool decode(const Node& node, QtHoloscanVideo*& data) {
    holoscan::log_error("YAML conversion not supported");
    return false;
  }
};

namespace holoscan::ops {

QtVideoOp::QtVideoOp() {}

void QtVideoOp::initialize() {
  // register a dummy converter to make YAML happy
  register_converter<QtHoloscanVideo*>();

  // add a boolean condition to be able to stop the operator
  add_arg(fragment()->make_condition<holoscan::BooleanCondition>("stop_condition"));

  // call the base class
  Operator::initialize();
}

void QtVideoOp::setup(holoscan::OperatorSpec& spec) {
  spec.input<holoscan::gxf::Entity>("input");

  spec.param(qt_holoscan_video_,
             "qt_holoscan_video",
             "QtHoloscanVideo",
             "Pointer to the QtHoloscanVideo object.",
             (QtHoloscanVideo*)nullptr);

  cuda_stream_handler_.define_params(spec);
}

void QtVideoOp::start() {
  CUDA_TRY(cudaEventCreate(&cuda_event_, cudaEventDisableTiming));
}

void QtVideoOp::stop() {
  if (cuda_event_) {
    CUDA_TRY(cudaEventDestroy(cuda_event_));
    cuda_event_ = nullptr;
  }
}

void QtVideoOp::compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
                        holoscan::ExecutionContext& context) {
  auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("input");
  if (!maybe_entity) { throw std::runtime_error("Failed to receive input"); }

  auto& entity = static_cast<nvidia::gxf::Entity&>(maybe_entity.value());

  // get the CUDA stream from the input message
  gxf_result_t stream_handler_result = cuda_stream_handler_.from_message(context.context(), entity);
  if (stream_handler_result != GXF_SUCCESS) {
    throw std::runtime_error("Failed to get the CUDA stream from incoming messages");
  }

  // record a CUDA event, the render later will synchronize the rendering using that event
  CUDA_TRY(cudaEventRecord(cuda_event_, cuda_stream_handler_.get_cuda_stream(context.context())));

  // Get the input data, both VideoBuffer and Tensor is supported. We collect the buffer info
  // in the `video_buffer_info` structure
  nvidia::gxf::VideoBufferInfo video_buffer_info{};
  void* pointer;

  const auto maybe_video_buffer = entity.get<nvidia::gxf::VideoBuffer>();
  if (maybe_video_buffer) {
    const auto video_buffer = maybe_video_buffer.value();

    if (video_buffer->storage_type() != nvidia::gxf::MemoryStorageType::kDevice) {
      throw std::runtime_error("VideoBuffer must be in device memory");
    }

    video_buffer_info = video_buffer->video_frame_info();

    pointer = video_buffer->pointer();
  } else {
    const auto maybe_tensor = entity.get<nvidia::gxf::Tensor>();
    if (!maybe_tensor) {
      throw std::runtime_error("Neither VideoBuffer not Tensor found in message");
    }

    const auto tensor = maybe_tensor.value();

    if (tensor->storage_type() != nvidia::gxf::MemoryStorageType::kDevice) {
      throw std::runtime_error("Tensor must be in device memory");
    }

    const uint32_t rank = tensor->rank();
    const int32_t components = tensor->shape().dimension(tensor->rank() - 1);

    video_buffer_info.color_planes.resize(1);
    video_buffer_info.color_planes[0].color_space = "RGBA";
    video_buffer_info.color_planes[0].bytes_per_pixel = tensor->bytes_per_element() * components;
    video_buffer_info.color_planes[0].stride = tensor->stride(0);
    if (tensor->rank() > 1) {
      video_buffer_info.width = tensor->shape().dimension(tensor->rank() - 2);
      if (tensor->rank() > 2) {
        video_buffer_info.height = tensor->shape().dimension(tensor->rank() - 3);
      } else {
        video_buffer_info.height = 1;
      }
    } else {
      video_buffer_info.width = 1;
    }

    video_buffer_info.color_planes[0].width = video_buffer_info.width;
    video_buffer_info.color_planes[0].height = video_buffer_info.height;
    video_buffer_info.color_planes[0].size =
        video_buffer_info.color_planes[0].height * video_buffer_info.color_planes[0].stride;

    if ((components == 4) && (tensor->element_type() == nvidia::gxf::PrimitiveType::kUnsigned8)) {
      video_buffer_info.color_format = nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA;
    } else if ((components == 1) &&
               (tensor->element_type() == nvidia::gxf::PrimitiveType::kUnsigned8)) {
      video_buffer_info.color_format = nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY;
    } else if ((components == 1) &&
               (tensor->element_type() == nvidia::gxf::PrimitiveType::kUnsigned16)) {
      video_buffer_info.color_format = nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY16;
    } else if ((components == 1) &&
               (tensor->element_type() == nvidia::gxf::PrimitiveType::kUnsigned32)) {
      video_buffer_info.color_format = nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY32;
    } else if ((components == 1) &&
               (tensor->element_type() == nvidia::gxf::PrimitiveType::kFloat32)) {
      video_buffer_info.color_format = nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY32F;
    } else {
      throw std::runtime_error("Unsupported tensor format");
    }

    pointer = tensor->pointer();
  }

  // Now send the buffer to the QtQuick QtHoloscanVideo element to be displayed
  qt_holoscan_video_->processBuffer(pointer, video_buffer_info, cuda_event_);

  // Synchronize with the event recorded by the renderer
  CUDA_TRY(
      cudaStreamWaitEvent(cuda_stream_handler_.get_cuda_stream(context.context()), cuda_event_));

  // Add the CUDA stream we used to the event to allow synchrinization when freeing the memory
  const auto maybe_stream_id = entity.add<nvidia::gxf::CudaStreamId>();
  if (!maybe_stream_id) {
    throw std::runtime_error("Failed to add CUDA stream id to output message.");
  }
  maybe_stream_id.value()->stream_cid =
      cuda_stream_handler_.get_stream_handle(context.context()).cid();
}

}  // namespace holoscan::ops
