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

# BINDINGS

include(pybind11_add_holohub_module)
find_package(holoscan 2.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install") # Remove if holoscan logger dependency removed

set(MODULE_NAME advanced_network_common)
set(MODULE_CLASS_NAME "*")

pybind11_add_module(${MODULE_NAME}_python MODULE adv_network_common_pybind.cpp)
#target_link_libraries(${MODULE_NAME}_python PRIVATE holoscan::core)

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
    ${CMAKE_PYBIND11_HOLOHUB_MODULE_OUT_DIR}/advanced_network_common/__init__.py
)

# UTILS

install(FILES tune_system.py
  DESTINATION bin
  PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
  COMPONENT advanced_network-utils
)
