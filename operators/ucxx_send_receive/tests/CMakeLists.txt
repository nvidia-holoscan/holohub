# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

find_package(holoscan 3.8 REQUIRED CONFIG)

add_executable(test_ucxx_ops
  test_ucxx_ops.cc
)

target_include_directories(test_ucxx_ops
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

target_link_libraries(test_ucxx_ops
  PRIVATE
    holoscan::core
    holoscan::ops::ucxx
    ucxx
)

add_test(NAME test_ucxx_ops
  COMMAND bash -c "(${CMAKE_CURRENT_BINARY_DIR}/test_ucxx_ops 2>&1 \| tee test_ucxx_ops.log) \
    && (grep -q \"Endpoint connected\" ${CMAKE_CURRENT_BINARY_DIR}/test_ucxx_ops.log || (echo \\\"ERROR: No endpoint connected\\\" && exit 1)) \
    && (grep -q \"Received tensor with rank 3 and shape: \\[1080, 1920, 3\\]\" ${CMAKE_CURRENT_BINARY_DIR}/test_ucxx_ops.log || (echo \"ERROR: No tensor received\" && exit 1)) \
    && (grep -q \"Received tensor with dtype: 1 bits: 8\" ${CMAKE_CURRENT_BINARY_DIR}/test_ucxx_ops.log || (echo \"ERROR: No tensor received\" && exit 1)) \
    && (grep -q \"Endpoint closed\" ${CMAKE_CURRENT_BINARY_DIR}/test_ucxx_ops.log || (echo \"ERROR: No endpoint closed\" && exit 1)) "
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
set_tests_properties(test_ucxx_ops PROPERTIES
  FAIL_REGULAR_EXPRESSION "Endpoint closed with status: Endpoint is not connected"
)