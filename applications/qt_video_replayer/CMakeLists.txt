# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(qt_video_replayer)

find_package(holoscan 0.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

find_package(Qt6 REQUIRED COMPONENTS Core Gui Qml Quick)

if(COMMAND qt_standard_project_setup)
  qt_standard_project_setup()
else()
  set(CMAKE_AUTOMOC ON)
  set(CMAKE_AUTOUIC ON)
endif()

# Create example
qt_add_executable(qt_video_replayer
  qt_video_replayer.cpp
  qt_holoscan_app.cpp
)

target_link_libraries(qt_video_replayer
  PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Qml
    Qt6::Quick
    holoscan::core
    holoscan::ops::qt_video
    holoscan::ops::format_converter
    holoscan::ops::video_stream_replayer
    holoscan::ops::npp_filter
  )

qt_add_qml_module(qt_video_replayer
    URI Qt
    VERSION 1.0
    QML_FILES
      qt_video_replayer.qml
    RESOURCE_PREFIX /scenegraph/qt_video_replayer
    NO_RESOURCE_TARGET_PATH
  )

# Download the racerx video sample data
if(HOLOHUB_DOWNLOAD_DATASETS)
  include(holoscan_download_data)
  holoscan_download_data(racerx
    URL nvidia/clara-holoscan/holoscan_racerx_video:20231009
    DOWNLOAD_NAME holoscan_racerx_video_20231009.zip
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
    ALL
  )
endif()

if(BUILD_TESTING)
  # Add test
  add_test(NAME qt_video_replayer_test
           COMMAND qt_video_replayer
           --count=100
           --data=${HOLOHUB_DATA_DIR}/racerx
           --basename=racerx
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(qt_video_replayer_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking.")
endif()
