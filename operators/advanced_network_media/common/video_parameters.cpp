/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "video_parameters.h"
#include "adv_network_media_logging.h"

namespace holoscan::ops {

VideoFormatSampling get_video_sampling_format(const std::string& format) {
  // Convert input format to lowercase for case-insensitive comparison
  std::string format_lower = format;
  std::transform(
      format_lower.begin(), format_lower.end(), format_lower.begin(), [](unsigned char c) {
        return std::tolower(c);
      });

  // Check against lowercase versions of format strings
  if (format_lower == "rgb888" || format_lower == "rgb") return VideoFormatSampling::RGB;

  // YCbCr 4:2:2 / YUV 4:2:2 formats
  if (format_lower == "ycbcr-4:2:2" || format_lower == "yuv422" || format_lower == "yuv-422" ||
      format_lower == "yuv-4:2:2" || format_lower == "ycbcr422")
    return VideoFormatSampling::YCbCr_4_2_2;

  // YCbCr 4:2:0 / YUV 4:2:0 formats
  if (format_lower == "ycbcr-4:2:0" || format_lower == "yuv420" || format_lower == "yuv-420" ||
      format_lower == "yuv-4:2:0" || format_lower == "ycbcr420")
    return VideoFormatSampling::YCbCr_4_2_0;

  // YCbCr 4:4:4 / YUV 4:4:4 formats
  if (format_lower == "ycbcr-4:4:4" || format_lower == "yuv444" || format_lower == "yuv-444" ||
      format_lower == "yuv-4:4:4" || format_lower == "ycbcr444")
    return VideoFormatSampling::YCbCr_4_4_4;

  // Return CUSTOM for any unsupported format
  ANM_CONFIG_LOG("Unsupported video sampling format: {}. Using CUSTOM format.", format);
  return VideoFormatSampling::CUSTOM;
}

VideoColorBitDepth get_color_bit_depth(int bit_depth) {
  switch (bit_depth) {
    case 8:
      return VideoColorBitDepth::_8;
    case 10:
      return VideoColorBitDepth::_10;
    case 12:
      return VideoColorBitDepth::_12;
    default:
      throw std::invalid_argument("Unsupported bit depth: " + std::to_string(bit_depth));
  }
}

nvidia::gxf::VideoFormat get_expected_gxf_video_format(VideoFormatSampling sampling,
                                                       VideoColorBitDepth depth) {
  if (sampling == VideoFormatSampling::RGB && depth == VideoColorBitDepth::_8) {
    return nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB;
  } else if (sampling == VideoFormatSampling::YCbCr_4_2_0 && depth == VideoColorBitDepth::_8) {
    return nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_709;
  } else if (sampling == VideoFormatSampling::CUSTOM) {
    return nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_CUSTOM;
  } else {
    // Return CUSTOM for any unsupported format
    return nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_CUSTOM;
  }
}

size_t calculate_frame_size(uint32_t width, uint32_t height, VideoFormatSampling sampling_format,
                            VideoColorBitDepth bit_depth) {
  using BytesPerPixelRatio = std::pair<uint32_t, uint32_t>;
  using ColorDepthPixelRatioMap =
      std::unordered_map<VideoFormatSampling,
                         std::unordered_map<VideoColorBitDepth, BytesPerPixelRatio>>;

  static const ColorDepthPixelRatioMap COLOR_DEPTH_TO_PIXEL_RATIO = {
      {VideoFormatSampling::RGB,
       {{VideoColorBitDepth::_8, {3, 1}},
        {VideoColorBitDepth::_10, {15, 4}},
        {VideoColorBitDepth::_12, {9, 2}}}},
      {VideoFormatSampling::YCbCr_4_4_4,
       {{VideoColorBitDepth::_8, {3, 1}},
        {VideoColorBitDepth::_10, {15, 4}},
        {VideoColorBitDepth::_12, {9, 2}}}},
      {VideoFormatSampling::YCbCr_4_2_2,
       {{VideoColorBitDepth::_8, {4, 2}},
        {VideoColorBitDepth::_10, {5, 2}},
        {VideoColorBitDepth::_12, {6, 2}}}},
      {VideoFormatSampling::YCbCr_4_2_0,
       {{VideoColorBitDepth::_8, {6, 4}},
        {VideoColorBitDepth::_10, {15, 8}},
        {VideoColorBitDepth::_12, {9, 4}}}}};

  auto format_it = COLOR_DEPTH_TO_PIXEL_RATIO.find(sampling_format);
  if (format_it == COLOR_DEPTH_TO_PIXEL_RATIO.end()) {
    throw std::invalid_argument("Unsupported sampling format");
  }

  auto depth_it = format_it->second.find(bit_depth);
  if (depth_it == format_it->second.end()) { throw std::invalid_argument("Unsupported bit depth"); }

  // Use 64-bit integer arithmetic to avoid float truncation and overflow on large dimensions
  uint64_t total_units = static_cast<uint64_t>(width) * static_cast<uint64_t>(height) *
                         static_cast<uint64_t>(depth_it->second.first);
  uint64_t size_bytes = (total_units + depth_it->second.second - 1) / depth_it->second.second;
  return static_cast<size_t>(size_bytes);
}

uint32_t get_channel_count_for_format(nvidia::gxf::VideoFormat format) {
  switch (format) {
    // RGB formats
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB:
      return 3;
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA:
      return 4;
    // NV12 formats (semi-planar with interleaved UV) - all use 2 channels
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12:
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_ER:
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_709:
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_709_ER:
      return 2;
    // YUV420 formats (multi-planar) - all use 3 planes (Y, U, V)
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_YUV420:
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_YUV420_ER:
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_YUV420_709:
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_YUV420_709_ER:
      return 3;
    default:
      ANM_LOG_WARN("Unknown format {}, assuming 3 channels", static_cast<int>(format));
      return 3;
  }
}

}  // namespace holoscan::ops
