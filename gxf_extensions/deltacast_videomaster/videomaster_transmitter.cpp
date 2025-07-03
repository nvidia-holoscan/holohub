/*
 * Copyright (c) 2022, DELTACAST.TV.
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
#include "videomaster_transmitter.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>
#include <utility>

#include "VideoMasterHD_ApplicationBuffers.h"
#include "VideoMasterHD_Sdi.h"
#include "VideoMasterHD_Keyer.h"
#include "gxf/multimedia/video.hpp"

namespace nvidia {
namespace holoscan {
namespace videomaster {

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

VideoMasterTransmitter::VideoMasterTransmitter() : VideoMasterBase(false) {}

gxf_result_t VideoMasterTransmitter::registerInterface(gxf::Registrar *registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(_use_rdma, "rdma", "Use RDMA",
                                 "Specifies whether RDMA should be used.");
  result &= registrar->parameter(_board_index, "board", "Board",
                                 "Index of the Deltacast.TV board to use.");
  result &= registrar->parameter(_channel_index, "output", "Output",
                                 "Index of the output channel to use.");
  result &= registrar->parameter(_width, "width", "Width", "Width of the video frames to send.");
  result &= registrar->parameter(_height, "height", "Height",
                                "Height of the video frames to send.");
  result &= registrar->parameter(_progressive, "progressive", "Progressive",
                                 "Progressiveness of the video frames to send.");
  result &= registrar->parameter(_framerate, "framerate", "Framerate",
                                 "Framerate of the signal to generate.");
  result &= registrar->parameter(_source, "source", "Source", "Source data.");
  result &= registrar->parameter(_pool, "pool", "Pool", "Pool to allocate the buffers.");
  result &= registrar->parameter(_overlay, "enable_overlay", "Overlay",
                "Specifies whether the input buffers should be treated as overlay data.", false);

  return gxf::ToResultCode(result);
}

gxf_result_t VideoMasterTransmitter::start() {
  gxf::Expected<void> result;
  result &= configure_board();
  result &= open_stream();

  if (!_overlay) {
    video_format = Deltacast::Helper::VideoFormat{_width, _height, _progressive, _framerate};
    _video_information->set_video_format(stream_handle(), video_format);

    auto opt_sync_source_property = _video_information->get_sync_source_properties();
    if (opt_sync_source_property)
      VHD_SetBoardProperty(*board_handle(), *opt_sync_source_property, VHD_GENLOCK_LOCAL);

    result &= configure_stream();
    result &= init_buffers();
    result &= start_stream();

    sleep_ms(200);
    set_loopback_state(false);
    sleep_ms(200);
  }
  return gxf::ToResultCode(result);
}

gxf_result_t VideoMasterTransmitter::tick() {
  bool success_b = true;
  gxf::Entity message;
  gxf::Expected<gxf::Entity> maybe_message = _source->receive();
  if (!maybe_message) {
    GXF_LOG_ERROR("No message available.");
    return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE;
  }
  message = std::move(maybe_message.value());

  gxf::Expected<gxf::Handle<gxf::Tensor>> maybe_video = gxf::Unexpected{GXF_UNINITIALIZED_VALUE};
  maybe_video = message.get<gxf::Tensor>();
  if (!maybe_video) {
    GXF_LOG_ERROR("Failed to retrieve VideoBuffer");
    return GXF_FAILURE;
  }
  auto frame = maybe_video.value();

  if (_overlay) {
    if (!signal_present()) {
      if (!_has_lost_signal)
        GXF_LOG_INFO("No signal detected, waiting for input...");

      _has_lost_signal = true;
      return GXF_SUCCESS;
    } else if (!(video_format != Deltacast::Helper::VideoFormat{})) {  // stream not started yet
      gxf::Expected<void> result;

      video_format = Deltacast::Helper::VideoFormat{_width, _height, _progressive, _framerate};
      _video_information->set_video_format(stream_handle(), video_format);

      result &= configure_board_for_overlay();
      result &= configure_stream();
      result &= configure_stream_for_overlay();
      result &= init_buffers();
      result &= start_stream();

      sleep_ms(200);
      set_loopback_state(false);
      sleep_ms(200);

      if (!result)
        return gxf::ToResultCode(result);
    }
  }

  HANDLE slot_handle;
  if (_slot_count >= _slot_handles.size()) {
    success_b = gxf_log_on_error(Deltacast::Helper::ApiSuccess{
                                      VHD_WaitSlotSent(*stream_handle(), &slot_handle, SLOT_TIMEOUT)
                                      }, "Failed to wait for slot to be sent");
    if (!success_b)
      return GXF_FAILURE;
  } else {
    slot_handle = _slot_handles[_slot_count % NB_SLOTS];
  }

  BYTE *buffer = nullptr;
  ULONG buffer_size = 0;

  success_b = gxf_log_on_error(Deltacast::Helper::ApiSuccess{
                                    VHD_GetSlotBuffer(slot_handle, VHD_SDI_BT_VIDEO
                                                      , &buffer, &buffer_size)
                                    }, "Failed to retrieve the video buffer");
  if (!success_b) {
    return GXF_FAILURE;
  }

  cudaMemcpy(buffer, frame->pointer(), buffer_size,
            (_use_rdma ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost));

  success_b = gxf_log_on_error(Deltacast::Helper::ApiSuccess{
                                    VHD_QueueOutSlot(slot_handle)
                                    }, "Failed to queue out the video buffer");
  if (!success_b) {
    return GXF_FAILURE;
  }

  _slot_count++;

  return GXF_SUCCESS;
}

gxf::Expected<void> VideoMasterTransmitter::configure_board_for_overlay() {
  bool success_b = true;

  success_b = _video_information->configure_sync(board_handle(), _channel_index);

  auto keyer_props = _video_information->get_keyer_properties(board_handle());

  success_b = success_b & gxf_log_on_error(Deltacast::Helper::ApiSuccess{
                                    VHD_SetBoardProperty(*board_handle()
                                                        , keyer_props.at(VHD_KEYER_BP_INPUT_A)
                                                        , id_to_rx_keyer_input.at(_channel_index))
                                     }, "Could not configure keyer input A");

  success_b = success_b & gxf_log_on_error(Deltacast::Helper::ApiSuccess{
                                    VHD_SetBoardProperty(*board_handle()
                                                        , keyer_props.at(VHD_KEYER_BP_INPUT_B)
                                                        , id_to_tx_keyer_input.at(_channel_index))
                                     }, "Could not configure keyer input B");

  success_b = success_b & gxf_log_on_error(Deltacast::Helper::ApiSuccess{
                                    VHD_SetBoardProperty(*board_handle()
                                                        , keyer_props.at(VHD_KEYER_BP_INPUT_K)
                                                        , id_to_tx_keyer_input.at(_channel_index))
                                     }, "Could not configure keyer input K");

  success_b = success_b & gxf_log_on_error(Deltacast::Helper::ApiSuccess{
                                    VHD_SetBoardProperty(*board_handle()
                                                       , id_to_keyer_video_output.at(_channel_index)
                                                       , VHD_KOUTPUT_KEYER)
                                     }, "Could not configure keyer video output");

  success_b = success_b & gxf_log_on_error(Deltacast::Helper::ApiSuccess{
                                    VHD_SetBoardProperty(*board_handle()
                                                        , keyer_props.at(VHD_KEYER_BP_ALPHACLIP_MIN)
                                                        , 0)
                                     }, "Could not configure alphaclip min");

  success_b = success_b & gxf_log_on_error(Deltacast::Helper::ApiSuccess{
                                    VHD_SetBoardProperty(*board_handle()
                                                        , keyer_props.at(VHD_KEYER_BP_ALPHACLIP_MAX)
                                                        , 1020)
                                     }, "Could not configure alphaclip max");

  success_b = success_b & gxf_log_on_error(Deltacast::Helper::ApiSuccess{
                                    VHD_SetBoardProperty(*board_handle()
                                                    , keyer_props.at(VHD_KEYER_BP_ALPHABLEND_FACTOR)
                                                    , 1023)
                                     }, "Could not configure alphablend factor");

  success_b = success_b & gxf_log_on_error(Deltacast::Helper::ApiSuccess{
                                    VHD_SetBoardProperty(*board_handle()
                                                        , keyer_props.at(VHD_KEYER_BP_ENABLE)
                                                        , TRUE)
                                     }, "Could not enable keyer");

  return success_b ? gxf::Success : gxf::Unexpected{GXF_FAILURE};
}

gxf::Expected<void> VideoMasterTransmitter::configure_stream_for_overlay() {
  bool success_b = true;

  success_b = success_b & gxf_log_on_error(Deltacast::Helper::ApiSuccess{
                                    VHD_SetStreamProperty(*stream_handle()
                                                         , VHD_CORE_SP_BUFFER_PACKING
                                                         , VHD_BUFPACK_VIDEO_RGBA_32)
                                     }, "Could not set buffer packing");

  auto opt_sync_tx_property = _video_information->get_sync_tx_properties();
  if (opt_sync_tx_property) {
    success_b = success_b & gxf_log_on_error(Deltacast::Helper::ApiSuccess{
                                      VHD_SetStreamProperty(*stream_handle()
                                                           , *opt_sync_tx_property, TRUE)
                                      }, "Could not set sync tx property");
  }

  return success_b ? gxf::Success : gxf::Unexpected{GXF_FAILURE};
}

}  // namespace videomaster
}  // namespace holoscan
}  // namespace nvidia
