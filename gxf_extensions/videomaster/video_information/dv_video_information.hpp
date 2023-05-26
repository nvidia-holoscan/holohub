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
#ifndef NVIDIA_HOLOSCAN_GXF_EXTENSIONS_VIDEOMASTER_DV_VIDEO_INFORMATION_HPP_
#define NVIDIA_HOLOSCAN_GXF_EXTENSIONS_VIDEOMASTER_DV_VIDEO_INFORMATION_HPP_

#include "video_information.hpp"

namespace nvidia {
namespace holoscan {
namespace videomaster {

struct VideoMasterDvVideoInformation : public VideoMasterVideoInformation {
  uint32_t get_buffer_type() override;
  uint32_t get_nb_buffer_types() override;
  uint32_t get_stream_processing_mode() override;
  std::vector<uint32_t> get_board_properties(uint32_t channel_index) override;
  std::vector<uint32_t> get_stream_properties() override;
  gxf::Expected<VideoFormat> get_video_format() override;
  gxf::Expected<void> update_stream_properties_values(VideoFormat video_format) override;
  gxf::Expected<void> configure_stream(void* stream_handle) override;
};

}  // namespace videomaster
}  // namespace holoscan
}  // namespace nvidia

#endif  // NVIDIA_HOLOSCAN_GXF_EXTENSIONS_VIDEOMASTER_DV_VIDEO_INFORMATION_HPP
