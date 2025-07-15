 # SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


add_executable(test_render_volume_xr
  main.cpp
)

target_link_libraries(test_render_volume_xr
  holoscan::core
  holoscan::ops::xr_transform_op
  holoscan::ops::frame_op
  holoscan::ops::volume_loader
  holoscan::ops::volume_renderer
  GTest::gtest_main
)

target_include_directories(test_render_volume_xr INTERFACE ${CMAKE_SOURCE_DIR}/operators/XrFrameOp)

# Copy config file to the build tree
add_custom_target(test_render_volume_xr_config_yaml
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/app_config.yaml" ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(test_render_volume_xr test_render_volume_xr_config_yaml)
