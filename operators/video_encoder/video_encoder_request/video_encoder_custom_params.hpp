/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    switch (rhs) {
        case nvidia::gxf::EncoderInputFormat::kNV12:
          ss = "nv12";
          break;
        case nvidia::gxf::EncoderInputFormat::kNV24:
          ss = "nv24";
          break;
        case nvidia::gxf::EncoderInputFormat::kYUV420PLANAR:
          ss = "yuv420planar";
          break;
        default:
            ss = "unsupported";
            break;
    }
    node.push_back(ss);
    YAML::Node value_node = node[0];
    return value_node;
  }

  static bool decode(const Node& node, nvidia::gxf::EncoderInputFormat& rhs) {
    if (!node.IsScalar()) return false;

    auto value = node.Scalar();
    if (value == "nv12") {
        rhs = nvidia::gxf::EncoderInputFormat::kNV12;
    } else if (value == "nv24") {
      rhs = nvidia::gxf::EncoderInputFormat::kNV24;
    } else if (value == "yuv420planar") {
        rhs = nvidia::gxf::EncoderInputFormat::kYUV420PLANAR;
    } else {
        rhs = nvidia::gxf::EncoderInputFormat::kUnsupported;
        return false;
    }
    return true;
  }
};

template <>
struct YAML::convert<nvidia::gxf::EncoderConfig> {
  static Node encode(const nvidia::gxf::EncoderConfig& rhs) {
    Node node;
    std::string ss;
    switch (rhs) {
        case nvidia::gxf::EncoderConfig::kIFrameCQP :
          ss = "iframe_cqp";
          break;
        case nvidia::gxf::EncoderConfig::kPFrameCQP:
          ss = "pframe_cqp";
          break;
        case nvidia::gxf::EncoderConfig::kCustom:
          ss = "custom";
          break;
        default:
            ss = "unsupported";
            break;
    }
    node.push_back(ss);
    YAML::Node value_node = node[0];
    return value_node;
  }

  static bool decode(const Node& node, nvidia::gxf::EncoderConfig& rhs) {
    if (!node.IsScalar()) return false;

    auto value = node.Scalar();
    if (value == "iframe_cqp") {
        rhs = nvidia::gxf::EncoderConfig::kIFrameCQP;
    } else if (value == "pframe_cqp") {
      rhs = nvidia::gxf::EncoderConfig::kPFrameCQP;
    } else if (value == "custom") {
        rhs = nvidia::gxf::EncoderConfig::kCustom;
    } else {
        rhs = nvidia::gxf::EncoderConfig::kUnsupported;
        return false;
    }
    return true;
  }
};

#endif /* HOLOSCAN_OPERATORS_VIDEO_ENCODER_REQUEST_VIDEO_ENCODER_CUSTOM_PARAMS */
