# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

add_library(ucxx_receiver_op SHARED ucxx_receiver_op.cpp)
add_library(holoscan::ops::ucxx_receiver_op ALIAS ucxx_receiver_op)

target_include_directories(ucxx_receiver_op
  PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(ucxx_receiver_op
  PUBLIC
    holoscan::core
    holoscan::ops::ucxx_common
    ucs
    ucxx
)
