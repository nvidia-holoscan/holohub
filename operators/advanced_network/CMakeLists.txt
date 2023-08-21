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
project(advanced_network)

find_package(holoscan 0.5 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(PkgConfig)

enable_language(CUDA)

add_library(advanced_network SHARED
  adv_network_rx.cpp
  adv_network_tx.cpp
  adv_network_common.cpp
  adv_network_kernels.cu
  adv_network_dpdk_mgr.cpp
)

add_library(holoscan::advanced_network ALIAS advanced_network)


######## DPDK Configuration ########
# Set this to wherever your DPDK installation was built if not in a standard location. this
# can be removed if it was installed globally.
#set(DPDK_OUTPUT_PATH /tmp/dpdk-22.11/out)
#set(ENV{PKG_CONFIG_PATH} "${DPDK_OUTPUT_PATH}/lib/aarch64-linux-gnu/pkgconfig")

set(DPDK_EXTRA_LIBS -Wl,--no-whole-archive -lmlx5 -libverbs -pthread -lnuma -ldl)

pkg_check_modules(DPDK REQUIRED libdpdk)

target_include_directories(advanced_network INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(advanced_network PUBLIC ${DPDK_CFLAGS})
target_link_libraries(advanced_network holoscan::core ${DPDK_EXTRA_LIBS} ${DPDK_LIBRARIES})

if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()
