# # SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# # SPDX-License-Identifier: Apache-2.0
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# # http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

cmake_minimum_required(VERSION 3.24)
project(velodyne_lidar_app CXX)

find_package(holoscan 2.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
list(PREPEND CMAKE_PREFIX_PATH "$ENV{DAQIRI_ROOT}" "/opt/daqiri")
find_package(daqiri REQUIRED CONFIG)
include(HoloHubDaqiri)

add_executable(velodyne_lidar_app main.cpp)

target_link_libraries(velodyne_lidar_app
   PRIVATE
   holoscan::core
   holoscan::ops::holoviz
   velodyne_lidar
)
holohub_link_daqiri(velodyne_lidar_app)
