# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rightsorsi_visualizer_lib reserved.
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
project(orsi_visualizer_lib)

find_package(holoscan 0.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)

find_package(VTK COMPONENTS
  CommonColor
  CommonCore
  CommonDataModel
  FiltersSources
  InteractionStyle
  IOGeometry
  IOImage
  RenderingContextOpenGL2
  RenderingCore
  RenderingFreeType
  RenderingGL2PSOpenGL2
  RenderingOpenGL2
  TestingRendering
)

# -------------------------------------------------------------------------------
#
# OpenGL Visualizer library
#

add_library(orsi_visualizer_lib SHARED
  opengl_utils.cpp
  opengl_utils.hpp
  video_frame.hpp
  video_frame.cpp
  vis_orsi.hpp
  vis_orsi.cpp
  vtk_view.hpp
  vtk_view.cpp
  vis_intf.hpp
)

target_include_directories(orsi_visualizer_lib
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(orsi_visualizer_lib
  holoscan::core
  glfw # to be replaced with Holoscan SDK dependency
  GLEW::GLEW  # to be replaced with Holoscan SDK dependency
  ${VTK_LIBRARIES}
)

# vtk_model_autoinit is needed
vtk_module_autoinit(
  TARGETS orsi_visualizer_lib
  MODULES ${VTK_LIBRARIES}
)
