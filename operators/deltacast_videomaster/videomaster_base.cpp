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

#include "holoscan/core/gxf/gxf_utils.hpp"

#include "VideoMasterHD_ApplicationBuffers.h"
#include "VideoMasterHD_String.h"
#include "gxf/multimedia/video.hpp"
#include "VideoMasterAPIHelper/api.hpp"
#include "VideoMasterAPIHelper/api_success.hpp"
#include "VideoMasterAPIHelper/enum_to_string.hpp"
#include "VideoMasterAPIHelper/VideoInformation/dv.hpp"
#include "VideoMasterAPIHelper/VideoInformation/sdi.hpp"

namespace holoscan::ops {

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
const std::unordered_map<uint32_t, VHD_CORE_BOARDPROPERTY> id_to_active_loopback_prop = {
    {0, VHD_CORE_BP_ACTIVE_LOOPBACK_0}
};
const std::unordered_map<uint32_t, VHD_CORE_BOARDPROPERTY> id_to_firmware_loopback_prop = {
    {0, VHD_CORE_BP_FIRMWARE_LOOPBACK_0},
    {1, VHD_CORE_BP_FIRMWARE_LOOPBACK_1}
};

VideoMasterBase::VideoMasterBase(bool is_input, uint32_t board_index, uint32_t channel_index, bool use_rdma)
    : _is_input(is_input), _board_index(board_index), _channel_index(channel_index), _use_rdma(use_rdma) {
        // Initialize CUDA context and check for errors
        cudaError_t cuda_error = cudaSetDevice(0);
        if (cuda_error != cudaSuccess) {
            HOLOSCAN_LOG_ERROR("Failed to set CUDA device: {}", cudaGetErrorString(cuda_error));
            _is_igpu = false;  // Default to discrete GPU behavior
            return;
        }
        
        cudaDeviceProp prop;
        cuda_error = cudaGetDeviceProperties(&prop, 0);
        if (cuda_error != cudaSuccess) {
            HOLOSCAN_LOG_ERROR("Failed to get CUDA device properties: {}", cudaGetErrorString(cuda_error));
            _is_igpu = false;  // Default to discrete GPU behavior
            return;
        }
        
        _is_igpu = prop.integrated;
        HOLOSCAN_LOG_INFO("CUDA device initialized: {} (Integrated: {})", prop.name, _is_igpu);
}

void VideoMasterBase::stop_stream() {
  HOLOSCAN_LOG_INFO("Stopping stream and closing handles");

  if (_stream_handle) {
    _stream_handle.reset();
  }

  sleep_ms(200);
  set_loopback_state(true);
  sleep_ms(200);

  if (_board_handle) {
    _board_handle.reset();
  }

  free_buffers();
  HOLOSCAN_LOG_INFO("VideoMaster stream stopped and cleaned up");
}

bool VideoMasterBase::configure_board() {
  std::string api_version = Deltacast::Helper::get_api_version();
  if (api_version.empty()) {
      HOLOSCAN_LOG_ERROR("Could not retrieve VideoMaster API version");
      return false;
  }

  uint32_t nb_boards = Deltacast::Helper::get_number_of_devices();

  if (nb_boards == 0) {
    HOLOSCAN_LOG_ERROR("No deltacast boards found");
    return false;
  }

  HOLOSCAN_LOG_INFO("VideoMaster API version: {} - {} boards detected", api_version, nb_boards);

  _board_handle = std::move(Deltacast::Helper::get_board_handle(_board_index));

  return _board_handle != nullptr;
}

bool VideoMasterBase::open_stream() {
  HOLOSCAN_LOG_INFO("Opening {} stream on channel {}", _is_input ? "input" : "output", _channel_index);
  Deltacast::Helper::ApiSuccess success;

  const auto &id_to_channel_type_prop =
                      _is_input ? id_to_rx_channel_type_prop : id_to_tx_channel_type_prop;
  const auto &id_to_stream_type = _is_input ? id_to_rx_stream_type : id_to_tx_stream_type;
  if (id_to_channel_type_prop.find(_channel_index) == id_to_channel_type_prop.end() ||
      id_to_stream_type.find(_channel_index) == id_to_stream_type.end()) {
    HOLOSCAN_LOG_ERROR("Invalid stream id ({})", _channel_index);
    return false;
  }

  success = VHD_GetBoardProperty(*board_handle(),
                                id_to_channel_type_prop.at(_channel_index),
                                (ULONG *)&_channel_type);

  if (!success) {
    HOLOSCAN_LOG_ERROR("Failed to retrieve channel type");
    return false;
  }

  switch (_channel_type) {
    case VHD_CHNTYPE_HDSDI:
    case VHD_CHNTYPE_3GSDI:
    case VHD_CHNTYPE_12GSDI:
      _video_information = std::make_unique<Deltacast::Helper::SdiVideoInformation>();
      break;
    case VHD_CHNTYPE_HDMI:
    case VHD_CHNTYPE_DISPLAYPORT:
      _video_information = std::make_unique<Deltacast::Helper::DvVideoInformation>();
      break;
    default:
      break;
  }

  if (!_video_information) {
    HOLOSCAN_LOG_ERROR("Unsupported channel type");
    return false;
  }

  _stream_handle = std::move(Deltacast::Helper::get_stream_handle(board_handle(),
                                                id_to_stream_type.at(_channel_index),
                                                _video_information->get_stream_processing_mode()));

  set_loopback_state(false);

  if (!_stream_handle) {
    HOLOSCAN_LOG_ERROR("Failed to open stream handle");
    return false;
  }

  _video_format = {};

  HOLOSCAN_LOG_INFO("{} stream successfully opened."
    , VHD_STREAMTYPE_ToString(id_to_stream_type.at(_channel_index)));

  return true;
}

bool VideoMasterBase::configure_stream() {
  bool success_b = true;
  success_b = holoscan_log_on_error(Deltacast::Helper::ApiSuccess{
                               VHD_SetStreamProperty(*stream_handle(),
                               VHD_CORE_SP_BUFFER_PACKING, VHD_BUFPACK_VIDEO_RGB_32)
                              },
                              "Failed to set stream type");
  if (!success_b) {
    return false;
  }

  if (!_video_information->get_video_format(stream_handle())->progressive) {
    success_b = holoscan_log_on_error(Deltacast::Helper::ApiSuccess{
                                  VHD_SetStreamProperty(*stream_handle()
                                                        , VHD_CORE_SP_FIELD_MERGE, TRUE)
                                  },
                                  "Failed to set field merge property");
    if (!success_b) {
      return false;
    }
  }

  if (_is_input) {
    auto success_opt = _video_information->set_stream_properties_values(
      stream_handle(),
      _video_information->get_stream_properties_values(stream_handle()));

    if (!success_opt.has_value() || !success_opt.value()) {
      HOLOSCAN_LOG_ERROR("Failed to set stream properties");
      return false;
    }
  }

  const auto &id_to_stream_type = _is_input ? id_to_rx_stream_type : id_to_tx_stream_type;
  _video_format = _video_information->get_video_format(stream_handle()).value();
  HOLOSCAN_LOG_INFO("{} configured in {}x{}@{}"
    , VHD_STREAMTYPE_ToString(id_to_stream_type.at(_channel_index))
    , _video_format.width, _video_format.height, _video_format.framerate);

  return true;
}

bool VideoMasterBase::init_buffers() {
  HOLOSCAN_LOG_INFO("Initializing {} buffers", _is_input ? "input" : "output");
  std::vector<ULONG> buffer_sizes;
  free_buffers();

  for (auto &slot : _buffers)
    slot.resize(_video_information->get_nb_buffer_types());

  Deltacast::Helper::ApiSuccess success;
  success = VHD_InitApplicationBuffers(*stream_handle());
  if (!success) {
    HOLOSCAN_LOG_ERROR("Failed to init application buffers");
    return false;
  }

  buffer_sizes.resize(_video_information->get_nb_buffer_types());
  HOLOSCAN_LOG_DEBUG("Buffer types count: {}, Slots count: {}", _video_information->get_nb_buffer_types(), _buffers.size());

  for (int buffer_type_index = 0; buffer_type_index < _video_information->get_nb_buffer_types();
       buffer_type_index++) {
    VHD_GetApplicationBuffersSize(*stream_handle(),
                                  buffer_type_index,
                                  &buffer_sizes[buffer_type_index]);
    HOLOSCAN_LOG_DEBUG("Buffer type {}: size = {} bytes", buffer_type_index, buffer_sizes[buffer_type_index]);
    if (!buffer_sizes[buffer_type_index])
      continue;

    HOLOSCAN_LOG_DEBUG("Is IGPU: {}", _is_igpu);
    for (int slot_index = 0; slot_index < _buffers.size(); slot_index++) {
      HOLOSCAN_LOG_DEBUG("Allocating buffer for slot {}, buffer type {}", slot_index, buffer_type_index);
      if ((buffer_type_index == _video_information->get_buffer_type() && _use_rdma) || _is_input) {
        cudaError_t cuda_error;
        HOLOSCAN_LOG_DEBUG("Allocating CUDA buffer for slot {}, buffer type {}", slot_index, buffer_type_index);
        if(_is_igpu){
          HOLOSCAN_LOG_DEBUG("Trying to allocate Slot {}, Buffer type {}: CUDA host alloc (iGPU) - {} bytes", slot_index, buffer_type_index, buffer_sizes[buffer_type_index]);
          cuda_error = cudaHostAlloc(&_buffers[slot_index][buffer_type_index], buffer_sizes[buffer_type_index], cudaHostAllocDefault);
          if (cuda_error != cudaSuccess) {
            HOLOSCAN_LOG_ERROR("CUDA host allocation failed: {}", cudaGetErrorString(cuda_error));
            return false;
          }
        } else {
          HOLOSCAN_LOG_DEBUG("Trying to allocate Slot {}, Buffer type {}: CUDA device alloc - {} bytes", slot_index, buffer_type_index, buffer_sizes[buffer_type_index]);
          cuda_error = cudaMalloc(&_buffers[slot_index][buffer_type_index], buffer_sizes[buffer_type_index]);
          if (cuda_error != cudaSuccess) {
            HOLOSCAN_LOG_ERROR("CUDA device allocation failed: {}", cudaGetErrorString(cuda_error));
            return false;
          }
        }
      }
      if ((buffer_type_index != _video_information->get_buffer_type() || !_use_rdma) || _is_input) {
        HOLOSCAN_LOG_DEBUG("Slot {}, Buffer type {}: posix_memalign alloc - {} bytes", slot_index, buffer_type_index, buffer_sizes[buffer_type_index]);
        void *allocated_buffer = nullptr;
        posix_memalign(&allocated_buffer, 4096, buffer_sizes[buffer_type_index]);
        HOLOSCAN_LOG_DEBUG("posix_memalign done, allocated_buffer={}", (void*)allocated_buffer);
        _buffers[slot_index][buffer_type_index] = (BYTE*)allocated_buffer;
        HOLOSCAN_LOG_DEBUG("allocated buffer stored in vector");
        
      }
    }
  }

  for (int slot_index = 0; slot_index < _slot_handles.size(); slot_index++) {
    HOLOSCAN_LOG_DEBUG("Creating slot {}/{}", slot_index + 1, _slot_handles.size());
    std::vector<VHD_APPLICATION_BUFFER_DESCRIPTOR> raw_buffer_pointer;
    for (int buffer_type_index = 0; buffer_type_index < _video_information->get_nb_buffer_types();
         buffer_type_index++) {
      VHD_APPLICATION_BUFFER_DESCRIPTOR desc;
      desc.Size = sizeof(VHD_APPLICATION_BUFFER_DESCRIPTOR);
      desc.pBuffer = (buffer_sizes[buffer_type_index]
                        ? _buffers[slot_index][buffer_type_index]
                        : nullptr);
      desc.RDMAEnabled = (buffer_type_index == _video_information->get_buffer_type() && _use_rdma);
      HOLOSCAN_LOG_DEBUG("  Buffer type {}: ptr={}, RDMA={}", buffer_type_index, (void*)desc.pBuffer, desc.RDMAEnabled);

      raw_buffer_pointer.push_back(desc);
    }

    success = VHD_CreateSlotEx(*stream_handle()
                              , raw_buffer_pointer.data(), &_slot_handles[slot_index]);
    if (!success) {
      HOLOSCAN_LOG_ERROR("Failed to create slot");
      return false;
    }

    if (_is_input) {
      success = VHD_QueueInSlot(_slot_handles[slot_index]);
      if (!success) {
        HOLOSCAN_LOG_ERROR("Failed to queue slot");
        return false;
      }
      HOLOSCAN_LOG_DEBUG("Slot {} queued for input", slot_index);
    }
  }

  HOLOSCAN_LOG_INFO("Buffers initialized successfully ({} slots)", NB_SLOTS);
  return true;
}

void VideoMasterBase::free_buffers() {
  HOLOSCAN_LOG_DEBUG("Freeing buffers");
  
  for (int slot_index = 0; slot_index < _buffers.size(); slot_index++) {
    for (int buffer_type_index = 0; buffer_type_index < _buffers[slot_index].size(); buffer_type_index++) {
      if (_buffers[slot_index][buffer_type_index] != nullptr) {
        // Check if this was a CUDA allocation
        if (_video_information && 
            ((buffer_type_index == _video_information->get_buffer_type() && _use_rdma) || _is_input)) {
          if (_is_igpu) {
            cudaFreeHost(_buffers[slot_index][buffer_type_index]);
            HOLOSCAN_LOG_DEBUG("Freed CUDA host buffer for slot {}, buffer type {}", slot_index, buffer_type_index);
          } else {
            cudaFree(_buffers[slot_index][buffer_type_index]);
            HOLOSCAN_LOG_DEBUG("Freed CUDA device buffer for slot {}, buffer type {}", slot_index, buffer_type_index);
          }
        } else {
          // This was a posix_memalign allocation
          free(_buffers[slot_index][buffer_type_index]);
          HOLOSCAN_LOG_DEBUG("Freed posix buffer for slot {}, buffer type {}", slot_index, buffer_type_index);
        }
        _buffers[slot_index][buffer_type_index] = nullptr;
      }
    }
  }
}

bool VideoMasterBase::start_stream() {

  const auto &id_to_stream_type = _is_input ? id_to_rx_stream_type : id_to_tx_stream_type;

  Deltacast::Helper::ApiSuccess success;
  if (!(success = VHD_StartStream(*stream_handle()))) {
    HOLOSCAN_LOG_ERROR("Could not start stream {}"
                  , VHD_STREAMTYPE_ToString(id_to_stream_type.at(_channel_index)));
    return false;
  }

  HOLOSCAN_LOG_INFO("{} stream successfully started."
              , VHD_STREAMTYPE_ToString(id_to_stream_type.at(_channel_index)));

  return true;
}

bool VideoMasterBase::holoscan_log_on_error(Deltacast::Helper::ApiSuccess result
                                      , const std::string& message) {
  bool result_b = static_cast<bool>(result);
  if (!result_b) {
    auto error_code = result.error_code();
    std::string error_message = message
                                + " - Error: " + Deltacast::Helper::enum_to_string(error_code);
    HOLOSCAN_LOG_ERROR(error_message.c_str());
  }
  return result_b;
}

bool VideoMasterBase::signal_present() {
  const std::unordered_map<uint32_t, VHD_CORE_BOARDPROPERTY> id_to_rx_status_prop = {
      {0, VHD_CORE_BP_RX0_STATUS}, {1, VHD_CORE_BP_RX1_STATUS},   {2, VHD_CORE_BP_RX2_STATUS},
      {3, VHD_CORE_BP_RX3_STATUS}, {4, VHD_CORE_BP_RX4_STATUS},   {5, VHD_CORE_BP_RX5_STATUS},
      {6, VHD_CORE_BP_RX6_STATUS}, {7, VHD_CORE_BP_RX7_STATUS},   {8, VHD_CORE_BP_RX8_STATUS},
      {9, VHD_CORE_BP_RX9_STATUS}, {10, VHD_CORE_BP_RX10_STATUS}, {11, VHD_CORE_BP_RX11_STATUS},
  };
  ULONG status;
  Deltacast::Helper::ApiSuccess success;

  success = VHD_GetBoardProperty(*board_handle(), id_to_rx_status_prop.at(_channel_index), &status);
  if (!success) {
    HOLOSCAN_LOG_ERROR("Failed to retrieve rx status");
    return false;
  }

  return !(status & VHD_CORE_RXSTS_UNLOCKED);
}

bool VideoMasterBase::set_loopback_state(bool state) {
  ULONG has_passive_loopback = FALSE;
  ULONG has_active_loopback = FALSE;
  ULONG has_firmware_loopback = FALSE;
  bool success_b = true;

  success_b = success_b & holoscan_log_on_error(Deltacast::Helper::ApiSuccess{
                                  VHD_GetBoardCapability(*board_handle(),
                                    VHD_CORE_BOARD_CAP_PASSIVE_LOOPBACK,
                                    &has_passive_loopback)
                                  }, "Failed to retrieve passive loopback capability");

  success_b = success_b & holoscan_log_on_error(Deltacast::Helper::ApiSuccess{
                                  VHD_GetBoardCapability(*board_handle(),
                                    VHD_CORE_BOARD_CAP_ACTIVE_LOOPBACK,
                                    &has_active_loopback)
                                  }, "Failed to retrieve active loopback capability");

  success_b = success_b & holoscan_log_on_error(Deltacast::Helper::ApiSuccess{
                                  VHD_GetBoardCapability(*board_handle(),
                                    VHD_CORE_BOARD_CAP_FIRMWARE_LOOPBACK,
                                    &has_firmware_loopback)
                                  }, "Failed to retrieve firmware loopback capability");

  if (has_firmware_loopback &&
      id_to_firmware_loopback_prop.find(_channel_index) != id_to_firmware_loopback_prop.end()) {
    success_b = holoscan_log_on_error(Deltacast::Helper::ApiSuccess{
                                  VHD_SetBoardProperty(*board_handle(),
                                    id_to_firmware_loopback_prop.at(_channel_index), state)
                                  }, "Failed to set firmware loopback state");

    return success_b;
  } else if (has_active_loopback &&
             id_to_active_loopback_prop.find(_channel_index) != id_to_active_loopback_prop.end()) {
    success_b = holoscan_log_on_error(Deltacast::Helper::ApiSuccess{
                                  VHD_SetBoardProperty(*board_handle(),
                                    id_to_active_loopback_prop.at(_channel_index),
                                    state)
                                  }, "Failed to set active loopback state");

    return success_b;
  } else if (has_passive_loopback &&
           id_to_passive_loopback_prop.find(_channel_index) != id_to_passive_loopback_prop.end()) {
    success_b = holoscan_log_on_error(Deltacast::Helper::ApiSuccess{
                                  VHD_SetBoardProperty(*board_handle(),
                                    id_to_passive_loopback_prop.at(_channel_index),
                                    state)
                                  }, "Failed to set passive loopback state");
    return success_b;
  }
  return true;
}

}  // namespace holoscan::ops
