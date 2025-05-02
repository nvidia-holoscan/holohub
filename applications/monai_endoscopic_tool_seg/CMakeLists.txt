# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(monai_endoscopic_tool_seg NONE)

find_package(holoscan 0.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

              # Default to download datasets
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)

# Download the endoscopy sample data
if(HOLOHUB_DOWNLOAD_DATASETS)
  include(holoscan_download_data)
  holoscan_download_data(endoscopy
    URL nvidia/clara-holoscan/holoscan_endoscopy_sample_data:20230222
    DOWNLOAD_NAME holoscan_endoscopy_sample_data_20230222.zip
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
    GENERATE_GXF_ENTITIES
    GXF_ENTITIES_HEIGHT 480
    GXF_ENTITIES_WIDTH 854
    GXF_ENTITIES_CHANNELS 3
    GXF_ENTITIES_FRAMERATE 30
    ALL
  )

  # MONAI endoscopic tool segmentation model
  holoscan_download_data(monai_tool_seg_model
    URL nvidia/clara-holoscan/monai_endoscopic_tool_segmentation_model:v0.1
    DOWNLOAD_NAME monai_endoscopic_tool_segmentation_model_v0.1.zip
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
    ALL
  )
endif()

# Add testing
if(BUILD_TESTING)

  set(RECORDING_DIR ${CMAKE_CURRENT_BINARY_DIR}/recording_output)
  set(SOURCE_VIDEO_BASENAME python_monai_endoscopic_tool_seg_output)
  set(VALIDATION_FRAMES_DIR ${CMAKE_SOURCE_DIR}/applications/monai_endoscopic_tool_seg/testing/)

  file(MAKE_DIRECTORY ${RECORDING_DIR})

  # Configure the yaml file for testing
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/tool_segmentation.yaml" CONFIG_FILE)
  string(REPLACE "count: 0" "count: 10" CONFIG_FILE ${CONFIG_FILE})
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/tool_segmentation_testing.yaml ${CONFIG_FILE})

  # Patch the current example to enable recording the rendering window
  add_custom_command(OUTPUT tool_segmentation_test.py
    COMMAND patch -u -o tool_segmentation_test.py ${CMAKE_CURRENT_SOURCE_DIR}/tool_segmentation.py
        ${CMAKE_SOURCE_DIR}/applications/monai_endoscopic_tool_seg/testing/tool_segmentation.patch
    COMMAND sed -i "s#RECORDING_DIR#${RECORDING_DIR}#g" tool_segmentation_test.py
    COMMAND sed -i "s#SOURCE_VIDEO_BASENAME#${SOURCE_VIDEO_BASENAME}#g" tool_segmentation_test.py
  )

  add_custom_target(tool_segmentation_test ALL
    DEPENDS "tool_segmentation_test.py"
  )
  # Add test
  add_test(NAME monai_endoscopic_tool_seg_python_test
    COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/tool_segmentation_test.py
                   --config ${CMAKE_CURRENT_BINARY_DIR}/tool_segmentation_testing.yaml
                   --data "${HOLOHUB_DATA_DIR}"
                   --source replayer
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_property(TEST monai_endoscopic_tool_seg_python_test PROPERTY ENVIRONMENT
               "PYTHONPATH=${GXF_LIB_DIR}/../python/lib:${CMAKE_BINARY_DIR}/python/lib")

  set_tests_properties(monai_endoscopic_tool_seg_python_test
                PROPERTIES PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking.;"
                FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")

  # Add a test to check the validity of the frames
  add_test(NAME monai_endoscopic_tool_seg_python_render_test
    COMMAND python3 ${CMAKE_SOURCE_DIR}/utilities/video_validation.py
    --source_video_dir ${RECORDING_DIR}
    --source_video_basename ${SOURCE_VIDEO_BASENAME}
    --output_dir ${RECORDING_DIR}
    --validation_frames_dir ${VALIDATION_FRAMES_DIR}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  set_tests_properties(monai_endoscopic_tool_seg_python_render_test PROPERTIES
    DEPENDS monai_endoscopic_tool_seg_python_test
    PASS_REGULAR_EXPRESSION "Valid video output!"
  )
endif()
