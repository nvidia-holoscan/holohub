/*
 * SPDX-FileCopyrightText:  Copyright (c) 2022, DELTACAST.TV. All rights reserved.
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

#include "videomaster_transmitter.hpp"

#include "holoscan/core/gxf/entity.hpp"

#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>
#include <utility>

#include "VideoMasterHD_ApplicationBuffers.h"
#include "VideoMasterHD_Sdi.h"
#include "VideoMasterHD_Keyer.h"
#include "gxf/multimedia/video.hpp"

namespace holoscan::ops {

const std::unordered_map<uint32_t, VHD_GENLOCKSOURCE> id_to_genlock_source = {
    {0, VHD_GENLOCK_RX0}, {1, VHD_GENLOCK_RX1}, {2, VHD_GENLOCK_RX2},   {3, VHD_GENLOCK_RX3},
    {4, VHD_GENLOCK_RX4}, {5, VHD_GENLOCK_RX5}, {6, VHD_GENLOCK_RX6},   {7, VHD_GENLOCK_RX7},
    {8, VHD_GENLOCK_RX8}, {9, VHD_GENLOCK_RX9}, {10, VHD_GENLOCK_RX10}, {11, VHD_GENLOCK_RX11},
};
const std::unordered_map<uint32_t, VHD_KEYERINPUT> id_to_rx_keyer_input = {
    {0, VHD_KINPUT_RX0}, {1, VHD_KINPUT_RX1}, {2, VHD_KINPUT_RX2}, {3, VHD_KINPUT_RX3}};
const std::unordered_map<uint32_t, VHD_KEYERINPUT> id_to_tx_keyer_input = {
    {0, VHD_KINPUT_TX0}, {1, VHD_KINPUT_TX1}, {2, VHD_KINPUT_TX2}, {3, VHD_KINPUT_TX3}};
const std::unordered_map<uint32_t, VHD_KEYEROUTPUT> id_to_rx_keyer_output = {
    {0, VHD_KOUTPUT_RX0}, {1, VHD_KOUTPUT_RX1}, {2, VHD_KOUTPUT_RX2}, {3, VHD_KOUTPUT_RX3}};
const std::unordered_map<uint32_t, VHD_KEYER_BOARDPROPERTY> id_to_keyer_video_output = {
    { 0, VHD_KEYER_BP_VIDEOOUTPUT_TX_0 },
    { 1, VHD_KEYER_BP_VIDEOOUTPUT_TX_1 },
    { 2, VHD_KEYER_BP_VIDEOOUTPUT_TX_2 },
    { 3, VHD_KEYER_BP_VIDEOOUTPUT_TX_3 },
};
const std::unordered_map<uint32_t, VHD_KEYER_BOARDPROPERTY> id_to_keyer_anc_output_prop = {
    { 0, VHD_KEYER_BP_ANCOUTPUT_TX_0 },
    { 1, VHD_KEYER_BP_ANCOUTPUT_TX_1 },
    { 2, VHD_KEYER_BP_ANCOUTPUT_TX_2 },
    { 3, VHD_KEYER_BP_ANCOUTPUT_TX_3 },
};

VideoMasterTransmitterOp::VideoMasterTransmitterOp() : holoscan::Operator(), _has_lost_signal(false), _video_master_base(false, _board_index, _channel_index, _use_rdma) {
}

void VideoMasterTransmitterOp::setup(OperatorSpec& spec) {
  auto& source = spec.input<holoscan::Tensor>("source");

  spec.param(_use_rdma, "rdma", "Use RDMA", "Specifies whether RDMA should be used.", false);
  spec.param(_board_index, "board", "Board", "Index of the Deltacast.TV board to use.", 0u);
  spec.param(_channel_index, "output", "Output", "Index of the output channel to use.", 0u);
  spec.param(_width, "width", "Width", "Width of the video frames to send.", 1920u);
  spec.param(_height, "height", "Height", "Height of the video frames to send.", 1080u);
  spec.param(_progressive,
             "progressive",
             "Progressive",
             "Progressiveness of the video frames to send.",
             true);
  spec.param(_framerate, "framerate", "Framerate", "Framerate of the signal to generate.", 60u);
  spec.param(_source, "source", "Source", "Source data.", &source);
  spec.param(_overlay,
             "enable_overlay",
             "EnableOverlay",
             "Specifies whether the input buffers should be treated as overlay data.",
             false);
}

void VideoMasterTransmitterOp::initialize() {}

void VideoMasterTransmitterOp::start() {
  if(!_video_master_base.configure_board())
    throw std::runtime_error("Failed to configure board");
  if(!_video_master_base.open_stream())
    throw std::runtime_error("Failed to open stream");
}

void VideoMasterTransmitterOp::compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) {

  auto source = op_input.receive<gxf::Entity>("source");
  if (!source || source.value().is_null()) {
    throw std::runtime_error("Failed to receive source");
  }

  auto video = source.value();

  auto video_buffer = holoscan::gxf::get_videobuffer(video);

  if (_overlay) {
    if (!_video_master_base.signal_present()) {
      if (!_has_lost_signal)
        HOLOSCAN_LOG_INFO("No signal detected, waiting for input...");

      _has_lost_signal = true;
      return;
    } else if (!(_video_master_base.video_format() != Deltacast::Helper::VideoFormat{})) {  // stream not started yet

      _video_master_base.video_format() = Deltacast::Helper::VideoFormat{_width, _height, _progressive, _framerate};
      _video_master_base.video_information()->set_video_format(_video_master_base.stream_handle(), _video_master_base.video_format());

      if(!configure_board_for_overlay())
        throw std::runtime_error("Failed to configure board for overlay");
      if(!_video_master_base.configure_stream())
        throw std::runtime_error("Failed to configure stream");
      if(!configure_stream_for_overlay())
        throw std::runtime_error("Failed to configure stream for overlay");
      if(!_video_master_base.init_buffers())
        throw std::runtime_error("Failed to initialize buffers");
      if(!_video_master_base.start_stream())
        throw std::runtime_error("Failed to start stream");

      sleep_ms(200);
      _video_master_base.set_loopback_state(false);
      sleep_ms(200);
    }
  }

  HANDLE slot_handle;
  if (_video_master_base.slot_count() >= _video_master_base.slot_handles().size()) {
    if(!_video_master_base.holoscan_log_on_error(Deltacast::Helper::ApiSuccess{
                                      VHD_WaitSlotSent(*_video_master_base.stream_handle(), &slot_handle, VideoMasterBase::SLOT_TIMEOUT)
                                      }, "Failed to wait for slot to be sent")){
      return;
                                      }
  } else {
    slot_handle = _video_master_base.slot_handles()[_video_master_base.slot_count() % VideoMasterBase::NB_SLOTS];
  }

  BYTE *buffer = nullptr;
  ULONG buffer_size = 0;

  if(!_video_master_base.holoscan_log_on_error(Deltacast::Helper::ApiSuccess{
                                    VHD_GetSlotBuffer(slot_handle, VHD_SDI_BT_VIDEO
                                                      , &buffer, &buffer_size)
                                    }, "Failed to retrieve the video buffer")){
                                      return;
                                    }

  cudaMemcpy(buffer, video_buffer->pointer(), buffer_size,
            (_use_rdma ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost));

  if(!_video_master_base.holoscan_log_on_error(Deltacast::Helper::ApiSuccess{
                                    VHD_QueueOutSlot(slot_handle)
                                    }, "Failed to queue out the video buffer")){
                                      return;
                                    }

  _video_master_base.slot_count()++;

  return;

}

void VideoMasterTransmitterOp::stop() {
  _video_master_base.stop_stream();
}

bool VideoMasterTransmitterOp::configure_board_for_overlay() {
  bool success_b = true;

  success_b = _video_master_base.video_information()->configure_sync(_video_master_base.board_handle(), _channel_index);

  auto keyer_props = _video_master_base.video_information()->get_keyer_properties(_video_master_base.board_handle());

  success_b = success_b & _video_master_base.holoscan_log_on_error(Deltacast::Helper::ApiSuccess{
                                    VHD_SetBoardProperty(*_video_master_base.board_handle()
                                                        , keyer_props.at(VHD_KEYER_BP_INPUT_A)
                                                        , id_to_rx_keyer_input.at(_channel_index))
                                     }, "Could not configure keyer input A");

  success_b = success_b & _video_master_base.holoscan_log_on_error(Deltacast::Helper::ApiSuccess{
                                    VHD_SetBoardProperty(*_video_master_base.board_handle()
                                                        , keyer_props.at(VHD_KEYER_BP_INPUT_B)
                                                        , id_to_tx_keyer_input.at(_channel_index))
                                     }, "Could not configure keyer input B");

  success_b = success_b & _video_master_base.holoscan_log_on_error(Deltacast::Helper::ApiSuccess{
                                    VHD_SetBoardProperty(*_video_master_base.board_handle()
                                                        , keyer_props.at(VHD_KEYER_BP_INPUT_K)
                                                        , id_to_tx_keyer_input.at(_channel_index))
                                     }, "Could not configure keyer input K");

  if(!_video_master_base.holoscan_log_on_error(Deltacast::Helper::ApiSuccess{
                                    VHD_SetBoardProperty(*_video_master_base.board_handle()
                                                       , id_to_keyer_video_output.at(_channel_index)
                                                       , VHD_KOUTPUT_KEYER)
                                     }, "Could not configure keyer video output")){
    return false;
  }

  if(!_video_master_base.holoscan_log_on_error(Deltacast::Helper::ApiSuccess{
                                    VHD_SetBoardProperty(*_video_master_base.board_handle()
                                                        , keyer_props.at(VHD_KEYER_BP_ALPHACLIP_MIN)
                                                        , 0)
                                     }, "Could not configure alphaclip min")){
    return false;
  }

  if(!_video_master_base.holoscan_log_on_error(Deltacast::Helper::ApiSuccess{
                                    VHD_SetBoardProperty(*_video_master_base.board_handle()
                                                        , keyer_props.at(VHD_KEYER_BP_ALPHACLIP_MAX)
                                                        , 1020)
                                     }, "Could not configure alphaclip max")){
                                      return false;
                                     }

  if(!_video_master_base.holoscan_log_on_error(Deltacast::Helper::ApiSuccess{
                                    VHD_SetBoardProperty(*_video_master_base.board_handle()
                                                    , keyer_props.at(VHD_KEYER_BP_ALPHABLEND_FACTOR)
                                                    , 1023)
                                     }, "Could not configure alphablend factor")){
                                      return false;
                                     }

  if(!_video_master_base.holoscan_log_on_error(Deltacast::Helper::ApiSuccess{
                                    VHD_SetBoardProperty(*_video_master_base.board_handle()
                                                        , keyer_props.at(VHD_KEYER_BP_ENABLE)
                                                        , TRUE)
                                     }, "Could not enable keyer")){
                                      return false;
                                     }

  return true;
}

bool VideoMasterTransmitterOp::configure_stream_for_overlay() {
  bool success_b = true;

  if(!_video_master_base.holoscan_log_on_error(Deltacast::Helper::ApiSuccess{
                                    VHD_SetStreamProperty(*_video_master_base.stream_handle()
                                                         , VHD_CORE_SP_BUFFER_PACKING
                                                         , VHD_BUFPACK_VIDEO_RGBA_32)
                                     }, "Could not set buffer packing")){
    return false;
  }

  auto opt_sync_tx_property = _video_master_base.video_information()->get_sync_tx_properties();
  if (opt_sync_tx_property) {
    if(!_video_master_base.holoscan_log_on_error(Deltacast::Helper::ApiSuccess{
                                      VHD_SetStreamProperty(*_video_master_base.stream_handle()
                                                           , *opt_sync_tx_property, TRUE)
                                      }, "Could not set sync tx property")){
      return false;
    }
  }

  return true;
}

}  // namespace holoscan::ops
