# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Download the model
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/video_deidentification/resnet34_peoplenet_int8.onnx"
    COMMAND mkdir -p "${HOLOHUB_DATA_DIR}/video_deidentification"
    COMMAND curl -S -o "${HOLOHUB_DATA_DIR}/video_deidentification/resnet34_peoplenet_int8.onnx"
              -L "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/pruned_quantized_decrypted_v2.3.3/files?redirect=true&path=resnet34_peoplenet_int8.onnx"
    VERBATIM
)

# Download the testing video
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/video_deidentification/tourist.mp4"
    COMMAND mkdir -p "${HOLOHUB_DATA_DIR}/video_deidentification"
    COMMAND curl -S -o "${HOLOHUB_DATA_DIR}/video_deidentification/tourist.mp4"
              -L "https://www.pexels.com/download/video/5271997/?fps=25.0&h=540&w=960"
    VERBATIM
)


# Convert the video file
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/video_deidentification/tourist.gxf_index"
                          "${HOLOHUB_DATA_DIR}/video_deidentification/tourist.gxf_entities"
    COMMAND ffmpeg -i "${HOLOHUB_DATA_DIR}/video_deidentification/tourist.mp4" -pix_fmt rgb24 -f rawvideo pipe:1 |
            python3 "${CMAKE_SOURCE_DIR}/utilities/convert_video_to_gxf_entities.py"
            --directory "${HOLOHUB_DATA_DIR}/video_deidentification" --basename tourist --width 960 --height 540 --framerate 24
    DEPENDS "${HOLOHUB_DATA_DIR}/video_deidentification/tourist.mp4"
)

add_custom_target(video_deidentification_data ALL
    DEPENDS
    "${HOLOHUB_DATA_DIR}/video_deidentification/resnet34_peoplenet_int8.onnx"
    "${HOLOHUB_DATA_DIR}/video_deidentification/tourist.gxf_entities"
)
