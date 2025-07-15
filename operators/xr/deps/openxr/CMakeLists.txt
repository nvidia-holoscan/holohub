# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

include(FetchContent)

# OpenXR SDK
FetchContent_Declare(
  OpenXR-SDK-Source
  GIT_REPOSITORY https://github.com/KhronosGroup/OpenXR-SDK-Source.git
  GIT_TAG dde23f1a37968b4b458d0de71b4f6b67ed16024a # release-1.0.26
)

set(BUILD_CONFORMANCE_TESTS OFF CACHE BOOL "Disable conformance tests")
set(BUILD_TESTS OFF CACHE BOOL "Disable tests")

FetchContent_MakeAvailable(OpenXR-SDK-Source)

add_library(openxr_headers INTERFACE)
target_link_libraries(openxr_headers INTERFACE headers)
target_compile_definitions(openxr_headers INTERFACE XR_USE_TIMESPEC)

# OpenXR HPP
FetchContent_Declare(
  OpenXR-Hpp
  GIT_REPOSITORY https://github.com/KhronosGroup/OpenXR-Hpp.git
  GIT_TAG 5d8262f14476f1b0504a58bdd7e8b89eb36787db # release-1.0.26
)

FetchContent_GetProperties(OpenXR-Hpp)
if(NOT openxr-hpp_POPULATED)
  FetchContent_Populate(OpenXR-Hpp)
endif()

add_custom_command(
  OUTPUT "${openxr-hpp_SOURCE_DIR}/include/openxr/openxr.hpp"
  COMMAND "${CMAKE_COMMAND}" -E env "OPENXR_REPO=${openxr-sdk-source_SOURCE_DIR}"
          "${openxr-hpp_SOURCE_DIR}/generate-openxr-hpp.sh" || true
  WORKING_DIRECTORY "${openxr-hpp_SOURCE_DIR}"
  COMMENT "Generate OpenXR-Hpp headers")

add_custom_target(
  generate_openxr_hpp
  DEPENDS "${openxr-hpp_SOURCE_DIR}/include/openxr/openxr.hpp"
  WORKING_DIRECTORY "${openxr-hpp_SOURCE_DIR}"
  COMMENT "Generate OpenXR-Hpp headers")

# Add the generated OpenXR-Hpp headers to openxr_headers.
add_dependencies(openxr_headers generate_openxr_hpp)
target_include_directories(openxr_headers INTERFACE ${openxr-hpp_SOURCE_DIR}/include)
