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
#ifndef NVIDIA_HOLOSCAN_GXF_EXTENSIONS_VIDEOMASTER_BASE_HPP_
#define NVIDIA_HOLOSCAN_GXF_EXTENSIONS_VIDEOMASTER_BASE_HPP_

#include <array>
#include <string>
#include <vector>
#include <optional>

#include "VideoMasterHD_Core.h"
#include "gxf/std/codelet.hpp"
#include "gxf/std/memory_buffer.hpp"
#include "VideoMasterAPIHelper/handle_manager.hpp"
#include "VideoMasterAPIHelper/VideoInformation/core.hpp"

#include <chrono>
#include <thread>
#define sleep_ms(value) std::this_thread::sleep_for(std::chrono::milliseconds(value))

namespace nvidia {
namespace holoscan {
namespace videomaster {

class VideoMasterBase : public gxf::Codelet {
 public:
  explicit VideoMasterBase(bool is_input);

  gxf_result_t stop() override;

 protected:
  static const uint32_t SLOT_TIMEOUT = 100;
  static const uint32_t NB_SLOTS = 4;
  gxf::Parameter<bool> _use_rdma;
  gxf::Parameter<uint32_t> _board_index;
  gxf::Parameter<uint32_t> _channel_index;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> _pool;

  Deltacast::Helper::BoardHandle& board_handle() { return *_board_handle; }
  Deltacast::Helper::StreamHandle& stream_handle() { return *_stream_handle; }

  bool _is_input;
  VHD_CHANNELTYPE _channel_type;
  bool _has_lost_signal;
  std::unique_ptr<Deltacast::Helper::VideoInformation> _video_information;
  std::array<std::vector<gxf::MemoryBuffer>, NB_SLOTS> _rdma_buffers;
  std::array<std::vector<BYTE*>, NB_SLOTS> _non_rdma_buffers;
  std::array<HANDLE, NB_SLOTS> _slot_handles;
  uint64_t _slot_count;

  gxf::Expected<void> configure_board();
  gxf::Expected<void> open_stream();
  gxf::Expected<void> configure_stream();
  gxf::Expected<void> init_buffers();
  gxf::Expected<void> start_stream();
  bool gxf_log_on_error(Deltacast::Helper::ApiSuccess result, const std::string& message);

  bool signal_present();
  bool set_loopback_state(bool state);

  Deltacast::Helper::VideoFormat video_format;

 private:
  void free_buffers();
  std::unique_ptr<Deltacast::Helper::BoardHandle> _board_handle;
  std::unique_ptr<Deltacast::Helper::StreamHandle> _stream_handle;
};

}  // namespace videomaster
}  // namespace holoscan
}  // namespace nvidia

#endif  // NVIDIA_HOLOSCAN_GXF_EXTENSIONS_VIDEOMASTER_BASE_HPP
