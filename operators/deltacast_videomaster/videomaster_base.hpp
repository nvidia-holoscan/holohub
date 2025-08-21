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
#ifndef HOLOSCAN_OPERATORS_VIDEOMASTER_BASE_HPP_
#define HOLOSCAN_OPERATORS_VIDEOMASTER_BASE_HPP_

#include <array>
#include <string>
#include <vector>
#include <optional>

#include "holoscan/holoscan.hpp"

#include "VideoMasterHD_Core.h"
#include "gxf/std/memory_buffer.hpp"
#include "VideoMasterAPIHelper/handle_manager.hpp"
#include "VideoMasterAPIHelper/VideoInformation/core.hpp"

#include <chrono>
#include <thread>
#define sleep_ms(value) std::this_thread::sleep_for(std::chrono::milliseconds(value))

namespace holoscan::ops {

class VideoMasterBase {
 public:

  VideoMasterBase(bool is_input, uint32_t board_index, uint32_t channel_index, bool use_rdma = false);

  static constexpr uint32_t SLOT_TIMEOUT = 100;
  static constexpr uint32_t NB_SLOTS = 4;

  bool configure_board();
  bool open_stream();
  bool configure_stream();
  bool init_buffers();
  bool start_stream();
  bool holoscan_log_on_error(Deltacast::Helper::ApiSuccess result, const std::string& message);
  void stop_stream();
  bool signal_present();
  bool set_loopback_state(bool state);

  Deltacast::Helper::BoardHandle& board_handle() { return *_board_handle; }
  Deltacast::Helper::StreamHandle& stream_handle() { return *_stream_handle; }
  
  // Getters for private members
  std::unique_ptr<Deltacast::Helper::VideoInformation>& video_information() { return _video_information; }
  const std::unique_ptr<Deltacast::Helper::VideoInformation>& video_information() const { return _video_information; }
  std::array<std::vector<BYTE*>, NB_SLOTS>& buffers() { return _buffers; }
  const std::array<std::vector<BYTE*>, NB_SLOTS>& buffers() const { return _buffers; }
  std::array<HANDLE, NB_SLOTS>& slot_handles() { return _slot_handles; }
  const std::array<HANDLE, NB_SLOTS>& slot_handles() const { return _slot_handles; }
  Deltacast::Helper::VideoFormat& video_format() { return _video_format; }
  const Deltacast::Helper::VideoFormat& video_format() const { return _video_format; }

 private:

  bool _is_input;
  bool _is_igpu = false;
  VHD_CHANNELTYPE _channel_type;
  std::unique_ptr<Deltacast::Helper::VideoInformation> _video_information;
  std::array<std::vector<BYTE*>, NB_SLOTS> _buffers;
  std::array<HANDLE, NB_SLOTS> _slot_handles;
  uint32_t _board_index;
  uint32_t _channel_index;
  bool _use_rdma;

  Deltacast::Helper::VideoFormat _video_format;

  void free_buffers();
  std::unique_ptr<Deltacast::Helper::BoardHandle> _board_handle;
  std::unique_ptr<Deltacast::Helper::StreamHandle> _stream_handle;
};

}  // namespace holoscan::ops

#endif  // NVIDIA_HOLOSCAN_GXF_EXTENSIONS_VIDEOMASTER_BASE_HPP
