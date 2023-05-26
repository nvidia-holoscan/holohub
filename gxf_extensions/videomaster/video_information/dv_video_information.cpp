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

#include "dv_video_information.hpp"

#include "VideoMasterHD_Core.h"
#include "VideoMasterHD_Dv.h"

namespace nvidia {
namespace holoscan {
namespace videomaster {

uint32_t VideoMasterDvVideoInformation::get_buffer_type() { return VHD_DV_BT_VIDEO; }

uint32_t VideoMasterDvVideoInformation::get_nb_buffer_types() { return NB_VHD_DV_BUFFERTYPE; }

uint32_t VideoMasterDvVideoInformation::get_stream_processing_mode() {
  return VHD_DV_STPROC_DISJOINED_VIDEO; }

std::vector<uint32_t> VideoMasterDvVideoInformation::get_board_properties(uint32_t channel_index) {
  return {};
}

std::vector<uint32_t> VideoMasterDvVideoInformation::get_stream_properties() {
  return {VHD_DV_SP_ACTIVE_WIDTH, VHD_DV_SP_ACTIVE_HEIGHT,
          VHD_DV_SP_INTERLACED, VHD_DV_SP_REFRESH_RATE};
}

gxf::Expected<VideoFormat> VideoMasterDvVideoInformation::get_video_format() {
  if (stream_properties_values.find(VHD_DV_SP_ACTIVE_WIDTH) == stream_properties_values.end() ||
      stream_properties_values.find(VHD_DV_SP_ACTIVE_HEIGHT) == stream_properties_values.end() ||
      stream_properties_values.find(VHD_DV_SP_INTERLACED) == stream_properties_values.end() ||
      stream_properties_values.find(VHD_DV_SP_REFRESH_RATE) == stream_properties_values.end())
    return gxf::Expected<VideoFormat>{VideoFormat{0, 0, false, 0}};

  return gxf::Expected<VideoFormat>{
      VideoFormat{stream_properties_values[VHD_DV_SP_ACTIVE_WIDTH],
                  stream_properties_values[VHD_DV_SP_ACTIVE_HEIGHT],
                  !stream_properties_values[VHD_DV_SP_INTERLACED],
                  stream_properties_values[VHD_DV_SP_REFRESH_RATE]}};
}

gxf::Expected<void>
VideoMasterDvVideoInformation::update_stream_properties_values(VideoFormat video_format) {
  stream_properties_values[VHD_DV_SP_ACTIVE_WIDTH] = video_format.width;
  stream_properties_values[VHD_DV_SP_ACTIVE_HEIGHT] = video_format.height;
  stream_properties_values[VHD_DV_SP_INTERLACED] = !video_format.progressive;
  stream_properties_values[VHD_DV_SP_REFRESH_RATE] = video_format.framerate;

  return gxf::Success;
}

gxf::Expected<void> VideoMasterDvVideoInformation::configure_stream(void* stream_handle) {
  auto return_code =
      VHD_PresetTimingStreamProperties(stream_handle,
                                       VHD_DV_STD_SMPTE,
                                       stream_properties_values[VHD_DV_SP_ACTIVE_WIDTH],
                                       stream_properties_values[VHD_DV_SP_ACTIVE_HEIGHT],
                                       stream_properties_values[VHD_DV_SP_REFRESH_RATE],
                                       stream_properties_values[VHD_DV_SP_INTERLACED]);

  return return_code == VHDERR_NOERROR ? gxf::Success : gxf::Unexpected{GXF_FAILURE};
}

}  // namespace videomaster
}  // namespace holoscan
}  // namespace nvidia
