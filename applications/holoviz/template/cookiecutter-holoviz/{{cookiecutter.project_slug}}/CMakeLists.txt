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

cmake_minimum_required(VERSION 3.20)

project({{ cookiecutter.project_slug }})

find_package(holoscan {{ cookiecutter.holoscan_version }} REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable({{ cookiecutter.project_slug }}
  {{ cookiecutter.project_slug }}.cpp
)

target_link_libraries({{ cookiecutter.project_slug }}
  PRIVATE
    holoscan::core
    holoscan::ops::holoviz
  )

if(BUILD_TESTING)
  # Add test
  add_test(NAME {{ cookiecutter.project_slug }}_test
           COMMAND {{ cookiecutter.project_slug }}
           --count=10
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties({{ cookiecutter.project_slug }}_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Application has finished running.")
endif()
