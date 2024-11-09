# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

message(STATUS "PROJECT_NAME: ${PROJECT_NAME}")

enable_language(CUDA)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(rmax_service)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/rmax_mgr_impl)

target_sources(${PROJECT_NAME}
    PRIVATE
        rmax_mgr_impl/adv_network_rmax_mgr.cpp
        rmax_mgr_impl/burst_manager.cpp
        rmax_mgr_impl/rmax_config_manager.cpp
    PUBLIC
        adv_network_rmax_mgr.h
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_compile_options(${PROJECT_NAME} PUBLIC -std=c++17)


target_link_libraries(${PROJECT_NAME}
    PRIVATE
        rmax_service
        rmax-ral-lib
        rmax-ral-build
        holoscan::core
)