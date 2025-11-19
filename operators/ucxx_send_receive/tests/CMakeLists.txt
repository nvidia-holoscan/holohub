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

add_executable(test_ucxx_ops
  test_ucxx_ops.cc
)

include(isaac_deps/holoscan)
include(isaac_deps/gtest)

target_link_libraries(test_ucxx_ops
  PRIVATE
    GTest::gtest_main
    holoscan::core
    isaac::common::torch_utils
    isaac::messages::reflection
    isaac::messages::schemas
    isaac::ops::ucxx
    torch
    ucxx
)

add_test(NAME test_ucxx_ops COMMAND test_ucxx_ops)
set_tests_properties(test_ucxx_ops PROPERTIES
  LABELS "cpp;operators;gpu"
)
