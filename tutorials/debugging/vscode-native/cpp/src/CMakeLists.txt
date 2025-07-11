# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

project(holoscan_hello_world_cpp LANGUAGES C CXX)

# Finds the package holoscan
find_package(holoscan REQUIRED CONFIG PATHS ${HOLOSCAN_INSTALL_DIR})

add_executable(hello_world
  hello_world.cpp
)

target_link_libraries(hello_world
  PRIVATE
  holoscan::core
)

# Testing
add_test(NAME EXAMPLE_CPP_HELLO_WORLD_TEST
          COMMAND hello_world
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
set_tests_properties(EXAMPLE_CPP_HELLO_WORLD_TEST PROPERTIES
                      PASS_REGULAR_EXPRESSION "Hello World!")
