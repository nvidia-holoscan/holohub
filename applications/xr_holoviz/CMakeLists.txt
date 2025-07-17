# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(xr_holoviz)


find_package(holoscan 3.3 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(xr_holoviz
    xr_holoviz.cpp
)

target_link_libraries(xr_holoviz
    holoscan::core
    holoscan::viz
    holoscan::ops::holoviz
    holoscan::ops::xr
    GXF::multimedia
)


# Copy config file to the build tree
add_custom_target(xr_holoviz_yaml
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/config.yaml" ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS "config.yaml"
    BYPRODUCTS "config.yaml"
)
add_dependencies(xr_holoviz xr_holoviz_yaml)