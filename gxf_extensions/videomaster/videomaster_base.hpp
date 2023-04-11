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

#include "VideoMasterHD_Core.h"
#include "gxf/std/codelet.hpp"
#include "gxf/std/memory_buffer.hpp"
#include "video_information/video_information.hpp"

namespace nvidia {
namespace holoscan {
namespace videomaster {

class VideoMasterBase : public gxf::Codelet {
 public:
  VideoMasterBase(bool is_input);

  gxf_result_t stop() override;

 protected:
  static const uint32_t SLOT_TIMEOUT = 100;
  static const uint32_t NB_SLOTS = 4;
  gxf::Parameter<bool> _use_rdma;
  gxf::Parameter<uint32_t> _board_index;
  gxf::Parameter<uint32_t> _channel_index;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> _pool;

  HANDLE _board_handle;
  HANDLE _stream_handle;
  bool _is_input;
  VHD_CHANNELTYPE _channel_type;
  bool _has_lost_signal;
  std::unique_ptr<VideoMasterVideoInformation> _video_information;
  std::array<std::vector<gxf::MemoryBuffer>, NB_SLOTS> _rdma_buffers;
  std::array<std::vector<BYTE*>, NB_SLOTS> _non_rdma_buffers;
  std::array<HANDLE, NB_SLOTS> _slot_handles;
  uint64_t _slot_count;

  gxf::Expected<void> configure_board();
  gxf::Expected<void> open_stream();
  gxf::Expected<void> configure_stream();
  gxf::Expected<void> init_buffers();
  gxf::Expected<void> start_stream();

  bool signal_present();
  bool set_loopback_state(bool state);
  std::unordered_map<ULONG, ULONG> get_detected_input_information(uint32_t channel_index) ;
  std::unordered_map<ULONG, ULONG> get_input_information();

  bool api_call_success(ULONG api_error_code, std::string error_message);

 private:
  void free_buffers();
};

}  // namespace videomaster
}  // namespace holoscan
}  // namespace nvidia

#endif  // NVIDIA_HOLOSCAN_GXF_EXTENSIONS_VIDEOMASTER_BASE_HPP_