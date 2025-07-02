/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef OPERATORS_ADVANCED_NETWORK_MEDIA_COMMON_VIDEO_PARAMETERS_H_
#define OPERATORS_ADVANCED_NETWORK_MEDIA_COMMON_VIDEO_PARAMETERS_H_

#include <string>
#include <unordered_map>
#include "gxf/multimedia/video.hpp"
#include <holoscan/holoscan.hpp>
#include "advanced_network/common.h"

using namespace holoscan::advanced_network;

namespace holoscan::ops {

/**
 * @enum VideoFormatSampling
 * @brief Enumeration for video sampling formats.
 */
enum class VideoFormatSampling {
  RGB,
  YCbCr_4_4_4,
  YCbCr_4_2_2,
  YCbCr_4_2_0,
  CUSTOM  // Default for unsupported formats
};

/**
 * @enum VideoColorBitDepth
 * @brief Enumeration for video color bit depths.
 */
enum class VideoColorBitDepth { _8, _10, _12 };

/**
 * @brief Converts a string format name to a VideoFormatSampling enum value.
 *
 * Supported formats include RGB888, YCbCr-4:2:2, YCbCr-4:2:0, YCbCr-4:4:4,
 * and simplified notations like yuv422, yuv420, yuv444. The comparison is
 * case-insensitive.
 *
 * @param format String representation of the video format.
 * @return The corresponding VideoFormatSampling enum value.
 *         Returns VideoFormatSampling::CUSTOM for unsupported formats.
 */
VideoFormatSampling get_video_sampling_format(const std::string& format);

/**
 * @brief Converts a bit depth integer to a VideoColorBitDepth enum value.
 *
 * @param bit_depth Integer representation of the bit depth.
 * @return The corresponding VideoColorBitDepth enum value.
 * @throws std::invalid_argument If the bit depth is not supported.
 */
VideoColorBitDepth get_color_bit_depth(int bit_depth);

/**
 * @brief Maps internal video format representation to GXF video format.
 *
 * @param sampling The video sampling format.
 * @param depth The color bit depth.
 * @return The GXF video format corresponding to the given settings.
 */
nvidia::gxf::VideoFormat get_expected_gxf_video_format(VideoFormatSampling sampling,
                                                       VideoColorBitDepth depth);

/**
 * @brief Calculates the frame size based on resolution, sampling format, and bit depth.
 *
 * @param width Frame width in pixels.
 * @param height Frame height in pixels.
 * @param sampling_format The video sampling format.
 * @param bit_depth The color bit depth.
 * @return The calculated frame size in bytes.
 * @throws std::invalid_argument If the sampling format or bit depth is unsupported.
 */
size_t calculate_frame_size(uint32_t width, uint32_t height, VideoFormatSampling sampling_format,
                            VideoColorBitDepth bit_depth);

/**
 * @brief Returns the number of channels required for a given video format
 *
 * @param format The GXF video format
 * @return Number of channels required for this format
 */
uint32_t get_channel_count_for_format(nvidia::gxf::VideoFormat format);

}  // namespace holoscan::ops

#endif  // OPERATORS_ADVANCED_NETWORK_MEDIA_COMMON_VIDEO_PARAMETERS_H_
