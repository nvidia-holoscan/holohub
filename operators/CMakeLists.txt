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

if(HOLOHUB_BUILD_PYTHON)
  # Define and create a directory to store python operators under the holohub python module
  if(NOT CMAKE_INSTALL_LIBDIR)
    set(CMAKE_INSTALL_LIBDIR lib)
  endif()
  set(HOLOHUB_PYTHON_MODULE_OUT_DIR ${CMAKE_BINARY_DIR}/python/${CMAKE_INSTALL_LIBDIR}/holohub)
  file(MAKE_DIRECTORY ${HOLOHUB_PYTHON_MODULE_OUT_DIR})
endif()

# Add operators (in alphabetical order)
add_holohub_operator(advanced_network)
add_holohub_operator(basic_network)
add_holohub_operator(cvcuda_holoscan_interop)
add_subdirectory(dds)
add_holohub_operator(deltacast_videomaster DEPENDS EXTENSIONS deltacast_videomaster)
add_holohub_operator(emergent_source DEPENDS EXTENSIONS emergent_source)
add_holohub_operator(lstm_tensor_rt_inference DEPENDS EXTENSIONS lstm_tensor_rt_inference)
add_holohub_operator(npp_filter)
add_holohub_operator(openigtlink)
add_holohub_operator(prohawk_video_processing)
add_holohub_operator(qt_video)
add_holohub_operator(realsense_camera)
add_subdirectory(orsi)
add_holohub_operator(tensor_to_video_buffer)
add_holohub_operator(tool_tracking_postprocessor)
add_holohub_operator(velodyne_lidar)
add_holohub_operator(video_encoder)
add_holohub_operator(visualizer_icardio)
add_holohub_operator(volume_loader)
add_holohub_operator(volume_renderer)
add_holohub_operator(vtk_renderer)
add_holohub_operator(xr_basic_render)
add_holohub_operator(XrFrameOp)
add_holohub_operator(XrTransformOp)
add_holohub_operator(yuan_qcap DEPENDS EXTENSIONS yuan_qcap)


# install
install(
    DIRECTORY "${CMAKE_BINARY_DIR}/python/lib/holohub"
    DESTINATION python/lib
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    DIRECTORY_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    PATTERN "__pycache__" EXCLUDE
)
