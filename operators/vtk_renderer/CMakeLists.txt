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
cmake_minimum_required(VERSION 3.20)
project(vtk_renderer LANGUAGES CXX)

find_package(holoscan REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

find_package(VTK REQUIRED COMPONENTS
  CommonColor
  CommonCore
  FiltersSources
  ImagingSources
  InteractionStyle
  RenderingContextOpenGL2
  RenderingCore
  RenderingFreeType
  RenderingGL2PSOpenGL2
  RenderingOpenGL2)

add_library(vtk_renderer SHARED vtk_renderer.cpp vtk_renderer.hpp)
add_library(holoscan::vtk_renderer ALIAS vtk_renderer)

target_link_libraries(vtk_renderer PUBLIC holoscan::core PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS  vtk_renderer
  MODULES ${VTK_LIBRARIES})

target_include_directories(vtk_renderer INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(vtk_renderer INTERFACE VTK_RENDERER)

if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()
