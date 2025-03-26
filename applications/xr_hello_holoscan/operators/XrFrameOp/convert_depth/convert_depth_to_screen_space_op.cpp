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

#include "convert_depth_to_screen_space_op.hpp"

#include "convert_depth_to_screen_space.hpp"
#include "gxf/cuda/cuda_event.hpp"
#include "gxf/multimedia/camera.hpp"
#include "gxf/multimedia/video.hpp"

namespace holoscan::openxr {

void ConvertDepthToScreenSpaceOp::setup(OperatorSpec& spec) {
  // spec.param(session_, "session", "OpenXR Session", "handles to OpenXR and Vulkan context");
  spec.input<holoscan::gxf::Entity>("depth_buffer_in");
  spec.output<holoscan::gxf::Entity>("depth_buffer_out");
  spec.input<nvidia::gxf::Vector2f>("depth_range");
}

void ConvertDepthToScreenSpaceOp::start() {
  if (cudaStreamCreate(&stream_) != cudaSuccess) {
    throw std::runtime_error("cudaStreamCreate failed");
  }
}

void ConvertDepthToScreenSpaceOp::stop() {
  cudaStreamDestroy(stream_);
}

void ConvertDepthToScreenSpaceOp::compute(InputContext& input, OutputContext& output,
                                          ExecutionContext& context) {
  auto depth_message = input.receive<holoscan::gxf::Entity>("depth_buffer_in").value();
  auto depth_range = input.receive<nvidia::gxf::Vector2f>("depth_range");

  nvidia::gxf::Expected<nvidia::gxf::Handle<nvidia::gxf::CudaEvent>> render_done_event =
      static_cast<nvidia::gxf::Entity&>(depth_message).get<nvidia::gxf::CudaEvent>();
  if (render_done_event.has_value() && render_done_event.value()->event().has_value()) {
    if (cudaStreamWaitEvent(stream_, render_done_event.value()->event().value()) != cudaSuccess) {
      throw std::runtime_error("cudaStreamWaitEvent failed");
    }
  }

  nvidia::gxf::Handle<nvidia::gxf::VideoBuffer> depth_buffer =
      static_cast<nvidia::gxf::Entity&>(depth_message).get<nvidia::gxf::VideoBuffer>().value();
  if (depth_buffer->video_frame_info().color_format !=
      nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY32F) {
    throw std::runtime_error("Unsupported depth buffer format");
  }
  convertDepthToScreenSpace(stream_,
                            reinterpret_cast<float*>(depth_buffer->pointer()),
                            depth_buffer->video_frame_info().width,
                            depth_buffer->video_frame_info().height,
                            depth_range->x,
                            depth_range->y);
  nvidia::gxf::Handle<nvidia::gxf::CudaEvent> convert_done_event =
      static_cast<nvidia::gxf::Entity&>(depth_message)
          .add<nvidia::gxf::CudaEvent>("convert_done")
          .value();
  convert_done_event->init();
  if (cudaEventRecord(convert_done_event->event().value(), stream_) != cudaSuccess) {
    throw std::runtime_error("cudaEventRecord failed");
  }

  output.emit(depth_message, "depth_buffer_out");
}

}  // namespace holoscan::openxr
