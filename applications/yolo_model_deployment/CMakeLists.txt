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
project(yolo_model_deployment NONE)

message(STATUS "Starting YOLO model deployment setup...")

find_package(holoscan 1.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Download the yolov8 model, exporting ONNX model, apply graph surgery for input size conversion
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/yolo_model_deployment/yolov8s.onnx"
    COMMAND git clone https://github.com/triple-Mu/YOLOv8-TensorRT.git || true
    COMMAND wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
    COMMAND mv yolov8s.pt YOLOv8-TensorRT
    COMMAND python3 YOLOv8-TensorRT/export-det.py --weights yolov8s.pt --iou-thres 0.65 --conf-thres 0.25 --topk 100 --opset 11 --sim --input-shape 1 3 640 640 --device cuda:0
    COMMAND wget https://raw.githubusercontent.com/nvidia-holoscan/holoscan-sdk/refs/heads/main/scripts/graph_surgeon.py
    COMMAND python ./graph_surgeon.py yolov8s.onnx yolov8s.onnx
    COMMAND mkdir -p "${HOLOHUB_DATA_DIR}/yolo_model_deployment"
    COMMAND mv yolov8s.onnx "${HOLOHUB_DATA_DIR}/yolo_model_deployment"
    )


# Download the testing video and trim
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/yolo_model_deployment/meeting.mp4"
    COMMAND mkdir -p "${HOLOHUB_DATA_DIR}/yolo_model_deployment"
    COMMAND curl -S -o "${HOLOHUB_DATA_DIR}/yolo_model_deployment/video.mp4"
              -L "https://www.pexels.com/download/video/3252919/?fps=25.0&h=720&w=1280"
    COMMAND ffmpeg -i "${HOLOHUB_DATA_DIR}/yolo_model_deployment/video.mp4" -t 6.4
              -c:v libx264 "${HOLOHUB_DATA_DIR}/yolo_model_deployment/meeting.mp4"
    COMMAND rm -rf "${HOLOHUB_DATA_DIR}/yolo_model_deployment/video.mp4"
    VERBATIM
    )

# Convert the video file
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/yolo_model_deployment/meeting.gxf_index"
                          "${HOLOHUB_DATA_DIR}/yolo_model_deployment/meeting.gxf_entities"
    COMMAND ffmpeg -i "${HOLOHUB_DATA_DIR}/yolo_model_deployment/meeting.mp4" -pix_fmt rgb24 -f rawvideo pipe:1 |
            python3 "${CMAKE_SOURCE_DIR}/utilities/convert_video_to_gxf_entities.py"
            --directory "${HOLOHUB_DATA_DIR}/yolo_model_deployment" --basename meeting --width 1280 --height 720 --framerate 25
    DEPENDS "${HOLOHUB_DATA_DIR}/yolo_model_deployment/meeting.mp4"
    )

add_custom_target(yolo_model_deployment ALL
                  DEPENDS
                  "${HOLOHUB_DATA_DIR}/yolo_model_deployment/yolov8s.onnx"
                  "${HOLOHUB_DATA_DIR}/yolo_model_deployment/meeting.gxf_index")

# Add testing
if(BUILD_TESTING)
  # Configure the yaml file for testing
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/yolo_detection.yaml" CONFIG_FILE)
  string(REPLACE "count: 0" "count: 10" CONFIG_FILE ${CONFIG_FILE})
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/yolo_detection_testing.yaml ${CONFIG_FILE})

  add_custom_target(yolo_detection_test ALL
    DEPENDS "yolo_detection.py"
  )

  # Add test
  add_test(NAME yolo_detection_python_test
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/yolo_detection.py
                   --config ${CMAKE_CURRENT_BINARY_DIR}/yolo_detection_testing.yaml
                   --data "${HOLOHUB_DATA_DIR}/yolo_model_deployment"
                   --source replayer
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_property(TEST yolo_detection_python_test PROPERTY ENVIRONMENT
               "PYTHONPATH=${GXF_LIB_DIR}/../python/lib")

  set_tests_properties(yolo_detection_python_test
                PROPERTIES PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking.;"
                FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")

endif()
