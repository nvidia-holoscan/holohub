# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(holo_to_gst CXX)

find_package(holoscan REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(holo-to-gst
  ../common/pattern_generator.cpp
  holo_to_gst.cpp
)

target_include_directories(holo-to-gst
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

target_link_libraries(holo-to-gst
  PRIVATE
  holoscan::core
  holoscan::ops::gstreamer_bridge
  holoscan::ops::v4l2
  holoscan::ops::format_converter
)

# Install target
install(TARGETS holo-to-gst DESTINATION bin/holo_to_gst)

