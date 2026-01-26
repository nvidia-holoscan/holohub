# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(ColonoscopySegmentation NONE)

# Default to download datasets
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)

# Download the associated dataset if needed
if(HOLOHUB_DOWNLOAD_DATASETS)
  include(holoscan_download_data)
  holoscan_download_data(colonoscopy_segmentation
    URL nvidia/clara-holoscan/holoscan_colonoscopy_sample_data:20230222
    DOWNLOAD_NAME holoscan_colonoscopy_sample_data_20230222.zip
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
    GENERATE_GXF_ENTITIES
    GXF_ENTITIES_HEIGHT 576
    GXF_ENTITIES_WIDTH 720
    GXF_ENTITIES_CHANNELS 3
    GXF_ENTITIES_FRAMERATE 30
    ALL
    )
endif()

# Add testing
if(BUILD_TESTING)

  set(RECORDING_DIR ${CMAKE_CURRENT_BINARY_DIR}/recording_output)
  set(SOURCE_VIDEO_BASENAME python_colonoscopy_segmentation_output)
  set(VALIDATION_FRAMES_DIR ${CMAKE_SOURCE_DIR}/applications/colonoscopy_segmentation/testing/)

  file(MAKE_DIRECTORY ${RECORDING_DIR})

  # Configure the yaml file for testing
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/colonoscopy_segmentation.yaml" CONFIG_FILE)
  string(REGEX REPLACE "source:[^\n]*" "source: replayer" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "count: 0" "count: 10" CONFIG_FILE ${CONFIG_FILE})
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/colonoscopy_segmentation_testing.yaml ${CONFIG_FILE})

    # Patch the current example to enable recording the rendering window
    add_custom_command(OUTPUT colonoscopy_segmentation_test.py
    COMMAND patch -u -o colonoscopy_segmentation_test.py ${CMAKE_CURRENT_SOURCE_DIR}/colonoscopy_segmentation.py
        ${CMAKE_SOURCE_DIR}/applications/colonoscopy_segmentation/testing/colonoscopy_segmentation.patch
    COMMAND sed -i "s#RECORDING_DIR#${RECORDING_DIR}#g" colonoscopy_segmentation_test.py
    COMMAND sed -i "s#SOURCE_VIDEO_BASENAME#${SOURCE_VIDEO_BASENAME}#g" colonoscopy_segmentation_test.py
  )

  add_custom_target(colonoscopy_segmentation_test ALL
    DEPENDS "colonoscopy_segmentation_test.py"
  )

  # Add test
  add_test(NAME colonoscopy_segmentation_python_test
           COMMAND ${PYTHON_EXECUTABLE_TEST} ${CMAKE_CURRENT_BINARY_DIR}/colonoscopy_segmentation_test.py
                   --config ${CMAKE_CURRENT_BINARY_DIR}/colonoscopy_segmentation_testing.yaml
                   --data "${HOLOHUB_DATA_DIR}/colonoscopy_segmentation"
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  set_tests_properties(colonoscopy_segmentation_python_test
                PROPERTIES PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking.;"
                FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed"
                TIMEOUT 1000)

  # Add a test to check the validity of the frames
  add_test(NAME colonoscopy_segmentation_python_render_test
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/utilities/video_validation.py
    --source_video_dir ${RECORDING_DIR}
    --source_video_basename ${SOURCE_VIDEO_BASENAME}
    --output_dir ${RECORDING_DIR}
    --validation_frames_dir ${VALIDATION_FRAMES_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  set_tests_properties(colonoscopy_segmentation_python_render_test PROPERTIES
    DEPENDS colonoscopy_segmentation_python_test
    PASS_REGULAR_EXPRESSION "Valid video output!"
  )

endif()
