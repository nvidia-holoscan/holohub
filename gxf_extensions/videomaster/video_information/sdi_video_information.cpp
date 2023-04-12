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

#include "sdi_video_information.hpp"

#include "VideoMasterHD_Core.h"
#include "VideoMasterHD_Sdi.h"

namespace nvidia {
namespace holoscan {
namespace videomaster {

const std::vector<VHD_SDI_BOARDPROPERTY> rx_channel_standard_board_properties = {
  VHD_SDI_BP_RX0_STANDARD,
  VHD_SDI_BP_RX1_STANDARD,
  VHD_SDI_BP_RX2_STANDARD,
  VHD_SDI_BP_RX3_STANDARD,
  VHD_SDI_BP_RX4_STANDARD,
  VHD_SDI_BP_RX5_STANDARD,
  VHD_SDI_BP_RX6_STANDARD,
  VHD_SDI_BP_RX7_STANDARD,
  VHD_SDI_BP_RX8_STANDARD,
  VHD_SDI_BP_RX9_STANDARD,
  VHD_SDI_BP_RX10_STANDARD,
  VHD_SDI_BP_RX11_STANDARD,
};

const std::vector<VHD_SDI_BOARDPROPERTY> rx_channel_interface_board_properties = {
  VHD_SDI_BP_RX0_INTERFACE,
  VHD_SDI_BP_RX1_INTERFACE,
  VHD_SDI_BP_RX2_INTERFACE,
  VHD_SDI_BP_RX3_INTERFACE,
  VHD_SDI_BP_RX4_INTERFACE,
  VHD_SDI_BP_RX5_INTERFACE,
  VHD_SDI_BP_RX6_INTERFACE,
  VHD_SDI_BP_RX7_INTERFACE,
  VHD_SDI_BP_RX8_INTERFACE,
  VHD_SDI_BP_RX9_INTERFACE,
  VHD_SDI_BP_RX10_INTERFACE,
  VHD_SDI_BP_RX11_INTERFACE,
};

const std::unordered_map<VHD_VIDEOSTANDARD, VHD_INTERFACE> default_interfaces = {
    {VHD_VIDEOSTD_S259M_PAL, VHD_INTERFACE_SD_259},
    {VHD_VIDEOSTD_S259M_NTSC_487, VHD_INTERFACE_SD_259},
    {VHD_VIDEOSTD_S259M_NTSC_480, VHD_INTERFACE_SD_259},

    {VHD_VIDEOSTD_S296M_720p_24Hz, VHD_INTERFACE_HD_292_1},
    {VHD_VIDEOSTD_S296M_720p_25Hz, VHD_INTERFACE_HD_292_1},
    {VHD_VIDEOSTD_S296M_720p_30Hz, VHD_INTERFACE_HD_292_1},
    {VHD_VIDEOSTD_S296M_720p_50Hz, VHD_INTERFACE_HD_292_1},
    {VHD_VIDEOSTD_S296M_720p_60Hz, VHD_INTERFACE_HD_292_1},
    {VHD_VIDEOSTD_S274M_1080i_50Hz, VHD_INTERFACE_HD_292_1},
    {VHD_VIDEOSTD_S274M_1080i_60Hz, VHD_INTERFACE_HD_292_1},
    {VHD_VIDEOSTD_S274M_1080p_24Hz, VHD_INTERFACE_HD_292_1},
    {VHD_VIDEOSTD_S274M_1080p_25Hz, VHD_INTERFACE_HD_292_1},
    {VHD_VIDEOSTD_S274M_1080p_30Hz, VHD_INTERFACE_HD_292_1},
    {VHD_VIDEOSTD_S274M_1080psf_24Hz, VHD_INTERFACE_HD_292_1},
    {VHD_VIDEOSTD_S274M_1080psf_25Hz, VHD_INTERFACE_HD_292_1},
    {VHD_VIDEOSTD_S274M_1080psf_30Hz, VHD_INTERFACE_HD_292_1},
    {VHD_VIDEOSTD_S2048M_2048p_24Hz, VHD_INTERFACE_HD_292_1},
    {VHD_VIDEOSTD_S2048M_2048p_25Hz, VHD_INTERFACE_HD_292_1},
    {VHD_VIDEOSTD_S2048M_2048p_30Hz, VHD_INTERFACE_HD_292_1},
    {VHD_VIDEOSTD_S2048M_2048psf_24Hz, VHD_INTERFACE_HD_292_1},
    {VHD_VIDEOSTD_S2048M_2048psf_25Hz, VHD_INTERFACE_HD_292_1},
    {VHD_VIDEOSTD_S2048M_2048psf_30Hz, VHD_INTERFACE_HD_292_1},

    {VHD_VIDEOSTD_S274M_1080p_50Hz, VHD_INTERFACE_3G_A_425_1},
    {VHD_VIDEOSTD_S274M_1080p_60Hz, VHD_INTERFACE_3G_A_425_1},
    {VHD_VIDEOSTD_S2048M_2048p_60Hz, VHD_INTERFACE_3G_A_425_1},
    {VHD_VIDEOSTD_S2048M_2048p_50Hz, VHD_INTERFACE_3G_A_425_1},
    {VHD_VIDEOSTD_S2048M_2048p_48Hz, VHD_INTERFACE_3G_A_425_1},

    {VHD_VIDEOSTD_3840x2160p_24Hz, VHD_INTERFACE_6G_2081_10},
    {VHD_VIDEOSTD_3840x2160p_25Hz, VHD_INTERFACE_6G_2081_10},
    {VHD_VIDEOSTD_3840x2160p_30Hz, VHD_INTERFACE_6G_2081_10},
    {VHD_VIDEOSTD_3840x2160psf_24Hz, VHD_INTERFACE_6G_2081_10},
    {VHD_VIDEOSTD_3840x2160psf_25Hz, VHD_INTERFACE_6G_2081_10},
    {VHD_VIDEOSTD_3840x2160psf_30Hz, VHD_INTERFACE_6G_2081_10},
    {VHD_VIDEOSTD_4096x2160p_24Hz, VHD_INTERFACE_6G_2081_10},
    {VHD_VIDEOSTD_4096x2160p_25Hz, VHD_INTERFACE_6G_2081_10},
    {VHD_VIDEOSTD_4096x2160p_30Hz, VHD_INTERFACE_6G_2081_10},
    {VHD_VIDEOSTD_4096x2160psf_24Hz, VHD_INTERFACE_6G_2081_10},
    {VHD_VIDEOSTD_4096x2160psf_25Hz, VHD_INTERFACE_6G_2081_10},
    {VHD_VIDEOSTD_4096x2160psf_30Hz, VHD_INTERFACE_6G_2081_10},

    {VHD_VIDEOSTD_3840x2160p_50Hz, VHD_INTERFACE_12G_2082_10},
    {VHD_VIDEOSTD_3840x2160p_60Hz, VHD_INTERFACE_12G_2082_10},
    {VHD_VIDEOSTD_4096x2160p_48Hz, VHD_INTERFACE_12G_2082_10},
    {VHD_VIDEOSTD_4096x2160p_50Hz, VHD_INTERFACE_12G_2082_10},
    {VHD_VIDEOSTD_4096x2160p_60Hz, VHD_INTERFACE_12G_2082_10},

    {VHD_VIDEOSTD_7680x4320p_24Hz, VHD_INTERFACE_4X6G_2081_12},
    {VHD_VIDEOSTD_7680x4320p_25Hz, VHD_INTERFACE_4X6G_2081_12},
    {VHD_VIDEOSTD_7680x4320p_30Hz, VHD_INTERFACE_4X6G_2081_12},
    {VHD_VIDEOSTD_8192x4320p_24Hz, VHD_INTERFACE_4X6G_2081_12},
    {VHD_VIDEOSTD_8192x4320p_25Hz, VHD_INTERFACE_4X6G_2081_12},
    {VHD_VIDEOSTD_8192x4320p_30Hz, VHD_INTERFACE_4X6G_2081_12},

    {VHD_VIDEOSTD_7680x4320p_50Hz, VHD_INTERFACE_4X12G_2082_12},
    {VHD_VIDEOSTD_7680x4320p_60Hz, VHD_INTERFACE_4X12G_2082_12},
    {VHD_VIDEOSTD_8192x4320p_48Hz, VHD_INTERFACE_4X12G_2082_12},
    {VHD_VIDEOSTD_8192x4320p_50Hz, VHD_INTERFACE_4X12G_2082_12},
    {VHD_VIDEOSTD_8192x4320p_60Hz, VHD_INTERFACE_4X12G_2082_12},
};

uint32_t VideoMasterSdiVideoInformation::get_buffer_type() { return VHD_SDI_BT_VIDEO; }

uint32_t VideoMasterSdiVideoInformation::get_nb_buffer_types() { return NB_VHD_SDI_BUFFERTYPE; }

uint32_t VideoMasterSdiVideoInformation::get_stream_processing_mode() { return VHD_SDI_STPROC_DISJOINED_VIDEO; }

std::vector<uint32_t> VideoMasterSdiVideoInformation::get_board_properties(uint32_t channel_index) {
  return {(uint32_t)rx_channel_standard_board_properties[channel_index], (uint32_t)rx_channel_interface_board_properties[channel_index]};
}

std::vector<uint32_t> VideoMasterSdiVideoInformation::get_stream_properties() {
  return {VHD_SDI_SP_VIDEO_STANDARD, VHD_SDI_SP_INTERFACE};
}

gxf::Expected<VideoFormat> VideoMasterSdiVideoInformation::get_video_format() {
  ULONG width, height, framerate;
  BOOL32 interlaced;
  if (stream_properties_values.find(VHD_SDI_SP_VIDEO_STANDARD) == stream_properties_values.end())
    return gxf::Expected<VideoFormat>{VideoFormat{0, 0, false, 0}};

  ULONG api_result = VHD_GetVideoCharacteristics((VHD_VIDEOSTANDARD)stream_properties_values[VHD_SDI_SP_VIDEO_STANDARD],
                                                 &width, &height, &interlaced, &framerate);
  if (api_result != VHDERR_NOERROR) {
    GXF_LOG_ERROR("Could not retrieve video characteristics");
    return gxf::Unexpected{GXF_FAILURE};
  }

  return gxf::Expected<VideoFormat>{VideoFormat{width, height, !interlaced, framerate}};
}

gxf::Expected<void> VideoMasterSdiVideoInformation::update_stream_properties_values(VideoFormat video_format) {
  for (uint32_t video_standard = 0; video_standard < NB_VHD_VIDEOSTANDARDS; video_standard++) {
    ULONG width, height, framerate;
    BOOL32 interlaced;
    ULONG api_result =
        VHD_GetVideoCharacteristics((VHD_VIDEOSTANDARD)video_standard, &width, &height, &interlaced, &framerate);
    if (api_result != VHDERR_NOERROR) {
      GXF_LOG_ERROR("Could not retrieve video characteristics");
      return gxf::Unexpected{GXF_FAILURE};
    }

    if (default_interfaces.find((VHD_VIDEOSTANDARD)video_standard) == default_interfaces.end()) {
      GXF_LOG_ERROR("Unsupported video standard");
      return gxf::Unexpected{GXF_FAILURE};
    }

    if (width == video_format.width && height == video_format.height && interlaced == !video_format.progressive &&
        framerate == video_format.framerate) {
      stream_properties_values[VHD_SDI_SP_VIDEO_STANDARD] = video_standard;
      stream_properties_values[VHD_SDI_SP_INTERFACE] = default_interfaces.at((VHD_VIDEOSTANDARD)video_standard);
      return gxf::Success;
    }
  }

  return gxf::Unexpected{GXF_FAILURE};
}

gxf::Expected<void> VideoMasterSdiVideoInformation::configure_stream(void* stream_handle) {
  for (auto& stream_prop : stream_properties_values) {
    auto return_code = VHD_SetStreamProperty(stream_handle, stream_prop.first, stream_prop.second);
    if (return_code != VHDERR_NOERROR) return gxf::Unexpected{GXF_FAILURE};
  }

  return gxf::Success;
}

}  // namespace videomaster
}  // namespace holoscan
}  // namespace nvidia