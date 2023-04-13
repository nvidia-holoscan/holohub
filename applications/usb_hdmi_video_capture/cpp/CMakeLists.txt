# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.20)
project(usb_hdmi_video_capture CXX)

find_package(holoscan 0.5 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(usb_hdmi_video_capture
    main.cpp
)

target_link_libraries(usb_hdmi_video_capture
   PRIVATE
   holoscan::core
   holoscan::ops::holoviz
   v4l2_video_capture
)

target_include_directories(usb_hdmi_video_capture
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../operators/v4l2_video_capture
)

# Copy config file
add_custom_target(usb_hdmi_video_capture_yaml
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/usb_hdmi_video_capture.yaml" ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS "usb_hdmi_video_capture.yaml"
    BYPRODUCTS "usb_hdmi_video_capture.yaml"
)
add_dependencies(usb_hdmi_video_capture usb_hdmi_video_capture_yaml)