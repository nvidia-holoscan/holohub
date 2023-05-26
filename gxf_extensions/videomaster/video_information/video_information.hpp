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
#ifndef NVIDIA_HOLOSCAN_GXF_EXTENSIONS_VIDEOMASTER_VIDEO_INFORMATION_HPP_
#define NVIDIA_HOLOSCAN_GXF_EXTENSIONS_VIDEOMASTER_VIDEO_INFORMATION_HPP_

#include "gxf/std/codelet.hpp"

namespace nvidia {
namespace holoscan {
namespace videomaster {

struct VideoFormat {
  uint32_t width;
  uint32_t height;
  bool progressive;
  uint32_t framerate;
};

struct VideoMasterVideoInformation {
  virtual uint32_t get_buffer_type() = 0;
  virtual uint32_t get_nb_buffer_types() = 0;
  virtual uint32_t get_stream_processing_mode() = 0;
  virtual std::vector<uint32_t> get_board_properties(uint32_t channel_index) = 0;
  virtual std::vector<uint32_t> get_stream_properties() = 0;
  virtual gxf::Expected<VideoFormat> get_video_format() = 0;
  virtual gxf::Expected<void> update_stream_properties_values(VideoFormat video_format) = 0;
  virtual gxf::Expected<void> configure_stream(void* stream_handle) = 0;

  std::unordered_map<uint32_t, uint32_t> stream_properties_values;
};

}  // namespace videomaster
}  // namespace holoscan
}  // namespace nvidia

#endif  // NVIDIA_HOLOSCAN_GXF_EXTENSIONS_VIDEOMASTER_VIDEO_INFORMATION_HPP
