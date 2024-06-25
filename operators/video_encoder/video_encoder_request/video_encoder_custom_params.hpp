/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_OPERATORS_VIDEO_ENCODER_REQUEST_VIDEO_ENCODER_CUSTOM_PARAMS
#define HOLOSCAN_OPERATORS_VIDEO_ENCODER_REQUEST_VIDEO_ENCODER_CUSTOM_PARAMS

#include <yaml-cpp/yaml.h>

#include "video_encoder_utils.hpp"

template <>
struct YAML::convert<nvidia::gxf::EncoderInputFormat> {
  static Node encode(const nvidia::gxf::EncoderInputFormat& rhs) {
    Node node;

    std::string ss;
    auto it = std::find_if(std::begin(nvidia::gxf::EncoderInputFormatMapping),
                           std::end(nvidia::gxf::EncoderInputFormatMapping),
                           [&rhs](auto&& p) { return p.second == rhs; });
    if (it != nvidia::gxf::EncoderInputFormatMapping.end()) {
      ss = it->first;
    } else {
      ss = "unsupported";
    }
    node.push_back(ss);
    YAML::Node value_node = node[0];
    return value_node;
  }

  static bool decode(const Node& node, nvidia::gxf::EncoderInputFormat& rhs) {
    if (!node.IsScalar()) return false;

    auto value = node.Scalar();
    auto format = nvidia::gxf::get_encoder_input_format(value);
    return format != nvidia::gxf::EncoderInputFormat::kUnsupported;
  }
};

template <>
struct YAML::convert<nvidia::gxf::EncoderConfig> {
  static Node encode(const nvidia::gxf::EncoderConfig& rhs) {
    Node node;
    std::string ss;
    auto it = std::find_if(std::begin(nvidia::gxf::EncoderConfigMapping),
                           std::end(nvidia::gxf::EncoderConfigMapping),
                           [&rhs](auto&& p) { return p.second == rhs; });
    if (it != nvidia::gxf::EncoderConfigMapping.end()) {
      ss = it->first;
    } else {
      ss = "unsupported";
    }
    node.push_back(ss);
    YAML::Node value_node = node[0];
    return value_node;
  }

  static bool decode(const Node& node, nvidia::gxf::EncoderConfig& rhs) {
    if (!node.IsScalar()) return false;

    auto value = node.Scalar();
    auto config = nvidia::gxf::get_encoder_config(value);
    return config != nvidia::gxf::EncoderConfig::kUnsupported;
  }
};

#endif /* HOLOSCAN_OPERATORS_VIDEO_ENCODER_REQUEST_VIDEO_ENCODER_CUSTOM_PARAMS */
