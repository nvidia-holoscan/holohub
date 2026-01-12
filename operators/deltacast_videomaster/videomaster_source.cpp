/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, DELTACAST.TV. All rights reserved.
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

#include "videomaster_source.hpp"


#include <cuda.h>
#include <cuda_runtime.h>

#include <string>
#include <utility>

#include "VideoMasterHD_ApplicationBuffers.h"
#include "gxf/multimedia/video.hpp"

#include "holoscan/core/gxf/entity.hpp"

#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"

#include "VideoMasterAPIHelper/VideoInformation/core.hpp"

using holoscan::OperatorSpec;
using holoscan::InputContext;
using holoscan::OutputContext;
using holoscan::ExecutionContext;
using holoscan::Arg;
using holoscan::ArgList;

namespace holoscan::ops {

void VideoMasterSourceOp::setup(OperatorSpec& spec) {
  auto& signal = spec.output<gxf::Entity>("signal");

  spec.param(_signal, "signal", "Output", "Output signal.", &signal);
  spec.param(_use_rdma, "rdma", "Use RDMA", "Specifies whether RDMA should be used.", false);
  spec.param(_board_index, "board", "Board", "Index of the Deltacast.TV board to use.", 0u);
  spec.param(_channel_index, "input", "Input", "Index of the input channel to use.", 0u);
  spec.param(_width, "width", "Width", "Width of the video frames to send.", 1920u);
  spec.param(_height, "height", "Height", "Height of the video frames to send.", 1080u);
  spec.param(_progressive,
             "progressive",
             "Progressive",
             "Progressiveness of the video frames to send.",
             true);
  spec.param(_framerate, "framerate", "Framerate", "Framerate of the signal to generate.", 60u);
}

void VideoMasterSourceOp::initialize() {
  Operator::initialize();
  _has_lost_signal = false;
  _video_master_base = std::make_unique<VideoMasterBase>(true, _board_index,
                                                        _channel_index, _use_rdma);
}

void VideoMasterSourceOp::start() {
  HOLOSCAN_LOG_INFO("Starting VideoMaster Source on board {} channel {}",
                     _board_index, _channel_index);
  if (!_video_master_base->configure_board())
    throw std::runtime_error("Failed to configure board");
  if (!_video_master_base->open_stream())
    throw std::runtime_error("Failed to open stream");
  HOLOSCAN_LOG_INFO("VideoMaster Source started successfully");
}

void VideoMasterSourceOp::compute(InputContext& op_input, OutputContext& op_output,
                                   ExecutionContext& context) {
  bool success_b = true;

  if (!_video_master_base->signal_present()) {
    if (!_has_lost_signal)
      HOLOSCAN_LOG_INFO("No signal detected, waiting for input...");

    _has_lost_signal = true;
    return;
  } else if (!(_video_master_base->video_format !=
               Deltacast::Helper::VideoFormat{})) {  // start stream
    if (!_video_master_base->configure_stream())
      throw std::runtime_error("Failed to configure stream");

    auto config_video_format = Deltacast::Helper::VideoFormat{_width, _height,
                                                              _progressive,
                                                              _framerate};
    if (_video_master_base->video_format != config_video_format) {
      VHD_StopStream(*_video_master_base->stream_handle());
      throw std::runtime_error("Input signal does not match configuration");
    }

    if (!_video_master_base->init_buffers())
      throw std::runtime_error("Failed to initialize buffers");
    if (!_video_master_base->start_stream())
      throw std::runtime_error("Failed to start stream");
    _slot_count = 0;
  }

  if (_has_lost_signal) {
    HOLOSCAN_LOG_INFO("Input signal detected, resuming operation");
    _has_lost_signal = false;
  }

  HOLOSCAN_LOG_DEBUG("Videoformat detection");
  auto detected_video_format = _video_master_base->video_information->get_video_format(
      _video_master_base->stream_handle());
  if (detected_video_format &&
      *detected_video_format != _video_master_base->video_format) {
    HOLOSCAN_LOG_INFO("Input signal has changed, exiting");
    VHD_StopStream(*_video_master_base->stream_handle());
    throw std::runtime_error("Input signal has changed");
  }

  HANDLE slot_handle;
  HOLOSCAN_LOG_DEBUG("Waiting for incoming slot");
  ULONG api_result = VHD_WaitSlotFilled(*_video_master_base->stream_handle(),
                                        &slot_handle,
                                        VideoMasterBase::SLOT_TIMEOUT);
  if (api_result != VHDERR_NOERROR) {
    if (api_result == VHDERR_TIMEOUT) {
      HOLOSCAN_LOG_INFO("Timeout");
      return;
    } else {
      throw std::runtime_error("Failed to wait for incoming slot");
    }
  }

  BYTE *buffer = nullptr;
  ULONG buffer_size = 0;

  if (!_video_master_base->holoscan_log_on_error(
      Deltacast::Helper::ApiSuccess{
          VHD_GetSlotBuffer(slot_handle,
                            _video_master_base->video_information->get_buffer_type(),
                            &buffer, &buffer_size)},
      "Failed to get slot buffer")) {
    throw std::runtime_error("Failed to get slot buffer");
  }

  HOLOSCAN_LOG_DEBUG("Transmit slot buffer {} - size: {} bytes",
                     (void*)buffer, buffer_size);
  transmit_buffer_data(buffer, buffer_size, op_output, context);

  HOLOSCAN_LOG_DEBUG("Queue slot");
  VHD_QueueInSlot(slot_handle);
  _slot_count++;
}

void VideoMasterSourceOp::stop() {
  HOLOSCAN_LOG_INFO("Stopping VideoMaster Source");
  _video_master_base->stop_stream();
}

void VideoMasterSourceOp::transmit_buffer_data(void* buffer, uint32_t buffer_size,
                                               OutputContext& op_output,
                                               ExecutionContext& context) {
  if (!_use_rdma) {
    // In non-RDMA mode, copy from system buffer (used by VideoMaster DMA) to GPU buffer
    // (used by processing pipeline)
    void* gpu_buffer = _video_master_base->gpu_buffers[
        _slot_count % VideoMasterBase::NB_SLOTS][
        _video_master_base->video_information->get_buffer_type()];
    cudaError_t cuda_status;
    if (_video_master_base->is_igpu())
      cuda_status = cudaMemcpy(gpu_buffer, buffer, buffer_size, cudaMemcpyHostToHost);
    else
      cuda_status = cudaMemcpy(gpu_buffer, buffer, buffer_size, cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMemcpy failed: ") +
                               cudaGetErrorString(cuda_status));
    }
    buffer = gpu_buffer;
  }
  auto video_output = nvidia::gxf::Entity::New(context.context());
  if (!video_output) {
    throw std::runtime_error("Failed to allocate video output; terminating.");
  }

  auto video_buffer = video_output.value().add<nvidia::gxf::VideoBuffer>();
  if (!video_buffer) {
    throw std::runtime_error("Failed to allocate video buffer; terminating.");
  }

  auto format = _video_master_base->video_information->get_video_format(
      _video_master_base->stream_handle());
  if (!format) throw std::runtime_error("Failed to get video format");

  nvidia::gxf::VideoTypeTraits<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA>
      video_type;
  nvidia::gxf::VideoFormatSize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA>
      color_format;
  auto color_planes = color_format.getDefaultColorPlanes(format->width, format->height);
  nvidia::gxf::VideoBufferInfo info{
      format->width, format->height, video_type.value, color_planes,
      nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR};
  auto storage_type = _video_master_base->is_igpu()
                          ? nvidia::gxf::MemoryStorageType::kHost
                          : nvidia::gxf::MemoryStorageType::kDevice;
  video_buffer.value()->wrapMemory(info, buffer_size, storage_type, buffer, nullptr);

  auto result = nvidia::gxf::Entity(std::move(video_output.value()));

  op_output.emit(result, "signal");
}
}  // namespace holoscan::ops
