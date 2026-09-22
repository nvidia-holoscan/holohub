# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

find_package(holoscan 2.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME grpc_endoscopy_tool_tracking_transport_test
           COMMAND "${Python3_EXECUTABLE}" -m pytest -q
                   "${CMAKE_SOURCE_DIR}/operators/grpc_operators/python/server/test_grpc_service.py"
           WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
  set_tests_properties(grpc_endoscopy_tool_tracking_transport_test PROPERTIES
                       LABELS "application;python;integration"
                       TIMEOUT 120)
endif()

install(
  DIRECTORY cloud edge
  DESTINATION bin/grpc_endoscopy_tool_tracking/python
)

install(
  FILES endoscopy_tool_tracking.yaml
  DESTINATION bin/grpc_endoscopy_tool_tracking/python
)

install(
  FILES requirements.txt
  DESTINATION bin/grpc_endoscopy_tool_tracking/python/cloud
)

install(
  FILES requirements.txt
  DESTINATION bin/grpc_endoscopy_tool_tracking/python/edge
)
