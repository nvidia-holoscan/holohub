# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(body_pose_estimation NONE)

find_package(holoscan 1.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Download the yolov8 pose model if it doesn't exist
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/body_pose_estimation/yolov8l-pose.onnx"
    COMMAND yolo export model=yolov8l-pose.pt format=onnx
    COMMAND mkdir -p "${HOLOHUB_DATA_DIR}/body_pose_estimation"
    COMMAND mv yolov8l-pose.onnx "${HOLOHUB_DATA_DIR}/body_pose_estimation"
    BYPRODUCTS "${HOLOHUB_DATA_DIR}/body_pose_estimation/yolov8l-pose.onnx"
    )

# Download the testing video
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/body_pose_estimation/twirl.mp4"
    COMMAND mkdir -p "${HOLOHUB_DATA_DIR}/body_pose_estimation"
    COMMAND curl -S -o "${HOLOHUB_DATA_DIR}/body_pose_estimation/twirl.mp4"
              -L "https://www.pexels.com/download/video/5385885/?fps=25.0&h=1920&w=1080"
    BYPRODUCTS "${HOLOHUB_DATA_DIR}/body_pose_estimation/twirl.mp4"
    VERBATIM
    )

# Convert the video file
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/body_pose_estimation/twirl.gxf_index"
                          "${HOLOHUB_DATA_DIR}/body_pose_estimation/twirl.gxf_entities"
    COMMAND ffmpeg -i "${HOLOHUB_DATA_DIR}/body_pose_estimation/twirl.mp4" -pix_fmt rgb24 -f rawvideo pipe:1 |
            python3 "${CMAKE_SOURCE_DIR}/utilities/convert_video_to_gxf_entities.py"
            --directory "${HOLOHUB_DATA_DIR}/body_pose_estimation" --basename twirl --width 1080 --height 1920 --framerate 25
    BYPRODUCTS "${HOLOHUB_DATA_DIR}/body_pose_estimation/twirl.gxf_index"
               "${HOLOHUB_DATA_DIR}/body_pose_estimation/twirl.gxf_entities"
    DEPENDS "${HOLOHUB_DATA_DIR}/body_pose_estimation/twirl.mp4"
    )

add_custom_target(body_pose_estimation_data ALL
                  DEPENDS
                  "${HOLOHUB_DATA_DIR}/body_pose_estimation/yolov8l-pose.onnx"
                  "${HOLOHUB_DATA_DIR}/body_pose_estimation/twirl.gxf_index")


# Copy qos_profiles.xml from the dds_video application to the binary directory.
# While this application could define its own QoS profiles, they would need to
# be compatible with the other publishers or subscribers that are reading or
# writing the video stream. Since the primary use case for this application is
# to share video frames with the dds_video application, it's easiest to just
# take the QoS profiles directly from dds_video to guarantee compatibility.
if(dds_video_subscriber IN_LIST HOLOHUB_BUILD_OPERATORS OR
   dds_video_publisher IN_LIST HOLOHUB_BUILD_OPERATORS)
  add_custom_target(body_pose_estimation_qos_profiles_xml ALL
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../dds_video/qos_profiles.xml" ${CMAKE_BINARY_DIR}
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../dds_video/qos_profiles.xml"
  )
endif()

# Add testing
if(BUILD_TESTING)
  # Configure the yaml file for testing
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/body_pose_estimation.yaml" CONFIG_FILE)
  string(REPLACE "count: 0" "count: 10" CONFIG_FILE ${CONFIG_FILE})
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/body_pose_estimation_testing.yaml ${CONFIG_FILE})

  add_custom_target(body_pose_estimation_test ALL
    DEPENDS "body_pose_estimation.py"
  )

  # Add test
  add_test(NAME body_pose_estimation_python_test
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/body_pose_estimation.py
                   --config ${CMAKE_CURRENT_BINARY_DIR}/body_pose_estimation_testing.yaml
                   --data "${HOLOHUB_DATA_DIR}/body_pose_estimation"
                   --source replayer
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_property(TEST body_pose_estimation_python_test PROPERTY ENVIRONMENT
               "PYTHONPATH=${GXF_LIB_DIR}/../python/lib")

  set_tests_properties(body_pose_estimation_python_test
                PROPERTIES PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking.;"
                FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")

endif()
