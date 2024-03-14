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

find_package(holoscan 1.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_custom_target(yolov8_onnx ALL
  COMMAND yolo export model=yolov8l-pose.pt format=onnx
  COMMAND mkdir -p "${HOLOHUB_DATA_DIR}/body_pose_estimation"
  COMMAND mv yolov8l-pose.onnx "${HOLOHUB_DATA_DIR}/body_pose_estimation"
  BYPRODUCTS "yolov8l-pose.onnx"
)
