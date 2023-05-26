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
#include "videomaster_base.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>
#include <utility>

#include "VideoMasterHD_ApplicationBuffers.h"
#include "gxf/multimedia/video.hpp"
#include "video_information/dv_video_information.hpp"
#include "video_information/sdi_video_information.hpp"

namespace nvidia {
namespace holoscan {
namespace videomaster {

const std::unordered_map<uint32_t, VHD_STREAMTYPE> id_to_rx_stream_type = {
    {0, VHD_ST_RX0}, {1, VHD_ST_RX1}, {2, VHD_ST_RX2}, {3, VHD_ST_RX3},
    {4, VHD_ST_RX4},   {5, VHD_ST_RX5}, {6, VHD_ST_RX6}, {7, VHD_ST_RX7},
    {8, VHD_ST_RX8}, {9, VHD_ST_RX9}, {10, VHD_ST_RX10}, {11, VHD_ST_RX11},
};
const std::unordered_map<uint32_t, VHD_CORE_BOARDPROPERTY> id_to_rx_channel_type_prop = {
    {0, VHD_CORE_BP_RX0_TYPE}, {1, VHD_CORE_BP_RX1_TYPE},
    {2, VHD_CORE_BP_RX2_TYPE},   {3, VHD_CORE_BP_RX3_TYPE},
    {4, VHD_CORE_BP_RX4_TYPE}, {5, VHD_CORE_BP_RX5_TYPE},
    {6, VHD_CORE_BP_RX6_TYPE},   {7, VHD_CORE_BP_RX7_TYPE},
    {8, VHD_CORE_BP_RX8_TYPE}, {9, VHD_CORE_BP_RX9_TYPE},
    {10, VHD_CORE_BP_RX10_TYPE}, {11, VHD_CORE_BP_RX11_TYPE},
};
const std::unordered_map<uint32_t, VHD_STREAMTYPE> id_to_tx_stream_type = {
    {0, VHD_ST_TX0}, {1, VHD_ST_TX1}, {2, VHD_ST_TX2}, {3, VHD_ST_TX3},
    {4, VHD_ST_TX4},   {5, VHD_ST_TX5}, {6, VHD_ST_TX6}, {7, VHD_ST_TX7},
    {8, VHD_ST_TX8}, {9, VHD_ST_TX9}, {10, VHD_ST_TX10}, {11, VHD_ST_TX11},
};
const std::unordered_map<uint32_t, VHD_CORE_BOARDPROPERTY> id_to_tx_channel_type_prop = {
    {0, VHD_CORE_BP_TX0_TYPE}, {1, VHD_CORE_BP_TX1_TYPE}, {2, VHD_CORE_BP_TX2_TYPE},
    {3, VHD_CORE_BP_TX3_TYPE}, {4, VHD_CORE_BP_TX4_TYPE}, {5, VHD_CORE_BP_TX5_TYPE},
    {6, VHD_CORE_BP_TX6_TYPE}, {7, VHD_CORE_BP_TX7_TYPE}, {8, VHD_CORE_BP_TX8_TYPE},
    {9, VHD_CORE_BP_TX9_TYPE}, {10, VHD_CORE_BP_TX10_TYPE}, {11, VHD_CORE_BP_TX11_TYPE},
};
const std::unordered_map<uint32_t, VHD_CORE_BOARDPROPERTY> id_to_passive_loopback_prop = {
    {0, VHD_CORE_BP_BYPASS_RELAY_0},
    {1, VHD_CORE_BP_BYPASS_RELAY_1},
    {2, VHD_CORE_BP_BYPASS_RELAY_2},
    {3, VHD_CORE_BP_BYPASS_RELAY_3}
};
const std::unordered_map<uint32_t, VHD_CORE_BOARDPROPERTY> id_to_active_loopback_prop =
                                                            {{0, VHD_CORE_BP_ACTIVE_LOOPBACK_0}};
const std::unordered_map<uint32_t, VHD_CORE_BOARDPROPERTY> id_to_firmware_loopback_prop =
                                                            {{0, VHD_CORE_BP_FIRMWARE_LOOPBACK_0}};

VideoMasterBase::VideoMasterBase(bool is_input)
    : _board_handle(nullptr),
      _stream_handle(nullptr),
      _is_input(is_input),
      _has_lost_signal(false) {}

gxf::Expected<void> VideoMasterBase::configure_board() {
  ULONG dll_version, nb_boards = 0;
  if (!api_call_success(VHD_GetApiInfo(&dll_version, &nb_boards),
                        "API info could not be retrieved"))
    return gxf::Unexpected{GXF_FAILURE};

  GXF_LOG_INFO("VideoMaster API version: %08x - %u boards detected", dll_version, nb_boards);

  if (nb_boards == 0) {
    GXF_LOG_ERROR("No deltacast boards found");
    return gxf::Unexpected{GXF_FAILURE};
  }

  bool success = api_call_success(VHD_OpenBoardHandle(_board_index, &_board_handle, NULL, 0),
                                                      "Failed to open board handle");

  return success ? gxf::Success : gxf::Unexpected{GXF_FAILURE};
}

gxf_result_t VideoMasterBase::stop() {
  GXF_LOG_INFO("Stopping stream and closing handles");

  VHD_StopStream(_stream_handle);
  VHD_CloseStreamHandle(_stream_handle);

  set_loopback_state(true);

  VHD_CloseBoardHandle(_board_handle);

  free_buffers();

  return GXF_SUCCESS;
}

gxf::Expected<void> VideoMasterBase::open_stream() {
  const auto &id_to_channel_type_prop =
                      _is_input ? id_to_rx_channel_type_prop : id_to_tx_channel_type_prop;
  const auto &id_to_stream_type = _is_input ? id_to_rx_stream_type : id_to_tx_stream_type;
  if (id_to_channel_type_prop.find(_channel_index) == id_to_channel_type_prop.end() ||
      id_to_stream_type.find(_channel_index) == id_to_stream_type.end()) {
    GXF_LOG_ERROR("Invalid stream id (%u)", _channel_index);
    return gxf::Unexpected{GXF_FAILURE};
  }

  if (!api_call_success(VHD_GetBoardProperty(_board_handle,
                                             id_to_channel_type_prop.at(_channel_index),
                                             (ULONG *)&_channel_type),
                                             "Failed to retrieve channel type")) {
    return gxf::Unexpected{GXF_FAILURE};
  }

  switch (_channel_type) {
    case VHD_CHNTYPE_HDSDI:
    case VHD_CHNTYPE_3GSDI:
    case VHD_CHNTYPE_12GSDI:
      _video_information =
        std::unique_ptr<VideoMasterSdiVideoInformation>(new VideoMasterSdiVideoInformation());
      break;
    case VHD_CHNTYPE_HDMI:
    case VHD_CHNTYPE_DISPLAYPORT:
      _video_information =
        std::unique_ptr<VideoMasterDvVideoInformation>(new VideoMasterDvVideoInformation());
      break;
    default:
      break;
  }

  if (!_video_information) {
    GXF_LOG_ERROR("Unsupported channel type");
    return gxf::Unexpected{GXF_FAILURE};
  }

  bool success = api_call_success(VHD_OpenStreamHandle(
                                              _board_handle,
                                              id_to_stream_type.at(_channel_index),
                                              _video_information->get_stream_processing_mode(),
                                              NULL,
                                              &_stream_handle,
                                              NULL), "Failed to open stream handle");
  set_loopback_state(false);

  return success ? gxf::Success : gxf::Unexpected{GXF_FAILURE};
}

gxf::Expected<void> VideoMasterBase::configure_stream() {
  bool success = api_call_success(VHD_SetStreamProperty(_stream_handle,
                                    VHD_CORE_SP_BUFFER_PACKING, VHD_BUFPACK_VIDEO_RGB_32),
                                  "Failed to set buffer packing");
  if (!_video_information->get_video_format()->progressive)
    success = success && api_call_success(VHD_SetStreamProperty(_stream_handle,
                                            VHD_CORE_SP_FIELD_MERGE, TRUE),
                                          "Failed to set field merging");
  success = success && _video_information->configure_stream(_stream_handle);

  return success ? gxf::Success : gxf::Unexpected{GXF_FAILURE};
}

gxf::Expected<void> VideoMasterBase::init_buffers() {
  free_buffers();

  if (_use_rdma || _is_input) {
    for (auto &slot : _rdma_buffers)
      slot.resize(_video_information->get_nb_buffer_types());
  }
  if (!_use_rdma || _is_input) {
    for (auto &slot : _non_rdma_buffers)
      slot.resize(_video_information->get_nb_buffer_types());
  }

  if (!api_call_success(VHD_InitApplicationBuffers(_stream_handle),
                                                    "Failed to init application buffers")) {
    return gxf::Unexpected{GXF_FAILURE};
  }

  for (int buffer_type_index = 0; buffer_type_index < _video_information->get_nb_buffer_types();
      buffer_type_index++) {
    ULONG buffer_size = 0;
    VHD_GetApplicationBuffersSize(_stream_handle, buffer_type_index, &buffer_size);
    if (!buffer_size)
      continue;

    for (int slot_index = 0; slot_index < _rdma_buffers.size(); slot_index++) {
      if ((buffer_type_index == _video_information->get_buffer_type() && _use_rdma) || _is_input) {
        _rdma_buffers[slot_index][buffer_type_index].resize(_pool, buffer_size,
                                                            gxf::MemoryStorageType::kDevice);
      }
      if ((buffer_type_index != _video_information->get_buffer_type() || !_use_rdma) || _is_input) {
        void *allocated_buffer = nullptr;
        posix_memalign(&allocated_buffer, 4096, buffer_size);
        _non_rdma_buffers[slot_index][buffer_type_index] = (BYTE*)allocated_buffer;
      }
    }
  }

  for (int slot_index = 0; slot_index < _slot_handles.size(); slot_index++) {
    std::vector<VHD_APPLICATION_BUFFER_DESCRIPTOR> raw_buffer_pointer;
    for (int buffer_type_index = 0; buffer_type_index < _video_information->get_nb_buffer_types();
         buffer_type_index++) {
      VHD_APPLICATION_BUFFER_DESCRIPTOR desc;
      desc.Size = sizeof(VHD_APPLICATION_BUFFER_DESCRIPTOR);
      desc.pBuffer = ((buffer_type_index == _video_information->get_buffer_type() && _use_rdma)
                                            ? _rdma_buffers[slot_index][buffer_type_index].pointer()
                                : _non_rdma_buffers[slot_index][buffer_type_index]);
      desc.RDMAEnabled = (buffer_type_index == _video_information->get_buffer_type() && _use_rdma);

      raw_buffer_pointer.push_back(desc);
    }

    if (!api_call_success(VHD_CreateSlotEx(_stream_handle, raw_buffer_pointer.data(),
                                           &_slot_handles[slot_index]), "Failed to create slot")) {
      return gxf::Unexpected{GXF_FAILURE};
    }

    if (_is_input) {
      if (!api_call_success(VHD_QueueInSlot(_slot_handles[slot_index]),
                                         "Failed to queue input slot")) {
        return gxf::Unexpected{GXF_FAILURE};
      }
    }
  }

  return gxf::Success;
}

void VideoMasterBase::free_buffers() {
  if (_use_rdma || _is_input) {
    for (auto& slot : _rdma_buffers)
      for (auto& buffer : slot) buffer.freeBuffer();
  }
  if (!_use_rdma || _is_input) {
    for (auto& slot : _non_rdma_buffers)
      for (auto& buffer : slot) free(buffer);
  }
}

gxf::Expected<void> VideoMasterBase::start_stream() {
  _slot_count = 0;

  return api_call_success(VHD_StartStream(_stream_handle), "Failed to start stream")
                          ? gxf::Success : gxf::Unexpected{GXF_FAILURE};
}

bool VideoMasterBase::signal_present() {
  const std::unordered_map<uint32_t, VHD_CORE_BOARDPROPERTY> id_to_rx_status_prop = {
      {0, VHD_CORE_BP_RX0_STATUS}, {1, VHD_CORE_BP_RX1_STATUS},   {2, VHD_CORE_BP_RX2_STATUS},
      {3, VHD_CORE_BP_RX3_STATUS}, {4, VHD_CORE_BP_RX4_STATUS},   {5, VHD_CORE_BP_RX5_STATUS},
      {6, VHD_CORE_BP_RX6_STATUS}, {7, VHD_CORE_BP_RX7_STATUS},   {8, VHD_CORE_BP_RX8_STATUS},
      {9, VHD_CORE_BP_RX9_STATUS}, {10, VHD_CORE_BP_RX10_STATUS}, {11, VHD_CORE_BP_RX11_STATUS},
  };
  ULONG status;
  if (!api_call_success(VHD_GetBoardProperty(_board_handle,
                                             id_to_rx_status_prop.at(_channel_index), &status),
                                             "Failed to check incoming RX status")) {
    return false;
  }

  return !(status & VHD_CORE_RXSTS_UNLOCKED);
}

std::unordered_map<ULONG, ULONG>
VideoMasterBase::get_detected_input_information(uint32_t channel_index) {
  std::unordered_map<ULONG, ULONG> input_information;
  auto board_properties = _video_information->get_board_properties(channel_index);
  auto stream_properties = _video_information->get_stream_properties();
  for (uint32_t i = 0; i < board_properties.size(); i++) {
    ULONG data;
    VHD_GetBoardProperty(_board_handle, board_properties[i], (ULONG*)&data);
    input_information[stream_properties[i]] = data;
  }

  return input_information;
}

std::unordered_map<ULONG, ULONG> VideoMasterBase::get_input_information() {
  std::unordered_map<ULONG, ULONG> input_information;
  for (auto prop : _video_information->get_stream_properties()) {
    ULONG data;
    VHD_GetStreamProperty(_stream_handle, prop, (ULONG*)&data);
    input_information[prop] = data;
  }

  return input_information;
}

bool VideoMasterBase::set_loopback_state(bool state) {
  ULONG has_passive_loopback = FALSE;
  ULONG has_active_loopback = FALSE;
  ULONG has_firmware_loopback = FALSE;

  api_call_success(VHD_GetBoardCapability(_board_handle,
                                          VHD_CORE_BOARD_CAP_PASSIVE_LOOPBACK,
                                          &has_passive_loopback),
                                          "failed to retrieve passive loopback capability");
  api_call_success(VHD_GetBoardCapability(_board_handle,
                                          VHD_CORE_BOARD_CAP_ACTIVE_LOOPBACK,
                                          &has_active_loopback),
                                          "failed to retrieve active loopback capability");
  api_call_success(VHD_GetBoardCapability(_board_handle,
                                           VHD_CORE_BOARD_CAP_FIRMWARE_LOOPBACK,
                                           &has_firmware_loopback),
                                           "failed to retrieve firmware loopback capability");

  if (has_firmware_loopback &&
      id_to_firmware_loopback_prop.find(_channel_index) != id_to_firmware_loopback_prop.end())
    return api_call_success(VHD_SetBoardProperty(_board_handle,
            id_to_firmware_loopback_prop.at(_channel_index), state),
            "failed to set firmware loopback state");
  else if (has_active_loopback &&
           id_to_active_loopback_prop.find(_channel_index) != id_to_active_loopback_prop.end())
    return api_call_success(VHD_SetBoardProperty(_board_handle,
              id_to_active_loopback_prop.at(_channel_index), state),
              "failed to set active loopback state");
  else if (has_passive_loopback &&
           id_to_passive_loopback_prop.find(_channel_index) != id_to_passive_loopback_prop.end())
    return api_call_success(VHD_SetBoardProperty(_board_handle,
              id_to_passive_loopback_prop.at(_channel_index), state),
              "failed to set passive loopback state");
  return true;
}

bool VideoMasterBase::api_call_success(ULONG api_error_code, std::string error_message) {
  if (api_error_code != VHDERR_NOERROR) {
    GXF_LOG_ERROR("%s", error_message.c_str());
    return false;
  }

  return true;
}

}  // namespace videomaster
}  // namespace holoscan
}  // namespace nvidia
