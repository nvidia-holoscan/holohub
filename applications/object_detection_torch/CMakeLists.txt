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
project(object_detection_torch CXX)

find_package(holoscan 0.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(object_detection_torch
    main.cpp
)

target_link_libraries(object_detection_torch
    PRIVATE
    holoscan::core
    holoscan::ops::aja
    holoscan::ops::video_stream_replayer
    holoscan::ops::video_stream_recorder
    holoscan::ops::format_converter
    holoscan::ops::inference
    holoscan::ops::inference_processor
    holoscan::ops::holoviz
)

# Download the cars sample data
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)
if(HOLOHUB_DOWNLOAD_DATASETS)
  include(holoscan_download_data)
  holoscan_download_data(object_detection_torch
    URL https://api.ngc.nvidia.com/v2/resources/nvidia/clara-holoscan/holoscan_cars_video/versions/20230725/zip
    DOWNLOAD_NAME holoscan_cars_video_20230725.zip
    URL_MD5 a0a510bc01cee0580a280d792da9ac5f
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
    GENERATE_GXF_ENTITIES
    GXF_ENTITIES_HEIGHT 1080
    GXF_ENTITIES_WIDTH 1920
    GXF_ENTITIES_CHANNELS 3
    GXF_ENTITIES_FRAMERATE 30
  )
  add_dependencies(object_detection_torch object_detection_torch_data)

  # Download the model
  add_custom_target(download_model
    COMMAND python3 "${CMAKE_CURRENT_SOURCE_DIR}/generate_resnet_model.py"  "${HOLOHUB_DATA_DIR}/object_detection_torch/frcnn_resnet50_t.pt"
    BYPRODUCTS "${HOLOHUB_DATA_DIR}/object_detection_torch/frcnn_resnet50_t.pt"
  )
  add_dependencies(object_detection_torch download_model)
endif()

# Copy config file
add_custom_target(object_detection_torch_yaml
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/object_detection_torch.yaml"
                                                  "${CMAKE_CURRENT_BINARY_DIR}/object_detection_torch.yaml"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/postprocessing.yaml"
                                                  "${HOLOHUB_DATA_DIR}/object_detection_torch/postprocessing.yaml"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/frcnn_resnet50_t.yaml"
                                                  "${HOLOHUB_DATA_DIR}/object_detection_torch/frcnn_resnet50_t.yaml"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/labels.txt"
                                                  "${HOLOHUB_DATA_DIR}/object_detection_torch/labels.txt"
    DEPENDS "object_detection_torch.yaml" "postprocessing.yaml" "frcnn_resnet50_t.yaml" "labels.txt"
    BYPRODUCTS "object_detection_torch.yaml" "postprocessing.yaml" "frcnn_resnet50_t.yaml" "labels.txt"
)
add_dependencies(object_detection_torch object_detection_torch_yaml)

# Add testing
if(BUILD_TESTING)

  set(RECORDING_DIR ${CMAKE_CURRENT_BINARY_DIR}/recording_output)
  set(SOURCE_VIDEO_BASENAME object_detection_torch_output)
  set(VALIDATION_FRAMES_DIR ${CMAKE_SOURCE_DIR}/applications/object_detection_torch/testing/)

  file(MAKE_DIRECTORY ${RECORDING_DIR})

  # Configure the yaml file for testing
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/object_detection_torch.yaml" CONFIG_FILE)
  string(REPLACE "count: 0" "count: 10" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "record_type: \"none\"" "record_type: \"visualizer\"" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "directory: \"/tmp\"" "directory: \"${RECORDING_DIR}\"" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "basename: \"tensor\"" "basename: \"${SOURCE_VIDEO_BASENAME}\"" CONFIG_FILE ${CONFIG_FILE})

  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/object_detection_torch_testing.yaml" ${CONFIG_FILE})

  # Add test
  add_test(NAME object_detection_torch_test
           COMMAND object_detection_torch ${CMAKE_CURRENT_BINARY_DIR}/object_detection_torch_testing.yaml
                   --data "${HOLOHUB_DATA_DIR}/object_detection_torch"
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  set_tests_properties(object_detection_torch_test PROPERTIES
                PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking.;"
                FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")

  # Add a test to check the validity of the frames
  add_test(NAME object_detection_torch_render_test
    COMMAND python3 ${CMAKE_SOURCE_DIR}/utilities/video_validation.py
    --source_video_dir ${RECORDING_DIR}
    --source_video_basename ${SOURCE_VIDEO_BASENAME}
    --output_dir ${RECORDING_DIR}
    --validation_frames_dir ${VALIDATION_FRAMES_DIR}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  set_tests_properties(object_detection_torch_render_test PROPERTIES
    DEPENDS object_detection_torch_test
    PASS_REGULAR_EXPRESSION "Valid video output!"
  )
endif()
