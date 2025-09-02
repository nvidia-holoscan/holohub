/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_ROS2_YAML_CONVERTER_HPP
#define HOLOSCAN_ROS2_YAML_CONVERTER_HPP

#include <stdexcept>

#include <yaml-cpp/yaml.h>

#define ROS2_DECLARE_YAML_CONVERTER_UNSUPPORTED(type) \
  template <>                                         \
  struct YAML::convert<type> {                        \
    static Node encode(const type&) {                 \
      throw std::runtime_error("Unsupported");        \
      return Node{};                                  \
    }                                                 \
    static bool decode(const Node&, type&) {          \
      throw std::runtime_error("Unsupported");        \
      return false;                                   \
    }                                                 \
  };

#endif /* HOLOSCAN_ROS2_YAML_CONVERTER_HPP */
