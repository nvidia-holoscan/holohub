# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

include(pybind11_add_holohub_module)
find_package(holoscan 2.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

set(MODULE_NAME advanced_network_media_rx)
set(MODULE_CLASS_NAME "*")

pybind11_add_holohub_module(
    CPP_CMAKE_TARGET advanced_network_media_rx
    CLASS_NAME "AdvNetworkMediaRxOp"
    SOURCES adv_network_media_rx_pybind.cpp
)

target_link_libraries(${MODULE_NAME}_python
PRIVATE
    holoscan::core
    ${MODULE_NAME}
)

set(CMAKE_PYBIND11_HOLOHUB_MODULE_OUT_DIR ${CMAKE_BINARY_DIR}/python/${CMAKE_INSTALL_LIBDIR}/holohub)
set(CMAKE_SUBMODULE_OUT_DIR ${CMAKE_PYBIND11_HOLOHUB_MODULE_OUT_DIR}/${MODULE_NAME})

set_target_properties(${MODULE_NAME}_python PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SUBMODULE_OUT_DIR}
    OUTPUT_NAME _${MODULE_NAME}
)

configure_file(
    ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/pybind11/__init__.py
    ${CMAKE_PYBIND11_HOLOHUB_MODULE_OUT_DIR}/advanced_network_media_rx/__init__.py
)
