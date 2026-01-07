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

cmake_minimum_required(VERSION 3.5)
project(holoscan_ros2_vb1940)

find_package(rclcpp REQUIRED HINTS /opt/ros/jazzy)
find_package(sensor_msgs REQUIRED)

find_package(holoscan 3.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Enable CUDA language
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# Hololink paths (adjust if needed)
set(HOLOLINK_ROOT "/deps/holoscan-sensor-bridge")
set(HOLOLINK_BUILD_DIR "${HOLOLINK_ROOT}/build")
set(HOLOLINK_INCLUDE_DIR "${HOLOLINK_ROOT}/src")

# Define hololink library directories
set(HOLOLINK_LIB_DIRS
    "${HOLOLINK_BUILD_DIR}/src/hololink/core"
    "${HOLOLINK_BUILD_DIR}/src/hololink/common"
    "${HOLOLINK_BUILD_DIR}/src/hololink/sensors"
    "${HOLOLINK_BUILD_DIR}/src/hololink/sensors/camera"
    "${HOLOLINK_BUILD_DIR}/src/hololink/sensors/camera/vb1940"
    "${HOLOLINK_BUILD_DIR}/src/hololink/operators"
    "${HOLOLINK_BUILD_DIR}/src/hololink/operators/csi_to_bayer"
    "${HOLOLINK_BUILD_DIR}/src/hololink/operators/roce_receiver"
    "${HOLOLINK_BUILD_DIR}/src/hololink/operators/image_processor"
)

link_directories(${HOLOLINK_LIB_DIRS})

add_library(hololink STATIC IMPORTED)
set_target_properties(hololink PROPERTIES
    IMPORTED_LOCATION "${HOLOLINK_BUILD_DIR}/src/hololink/common/libhololink.a"
    INTERFACE_LINK_LIBRARIES "CUDA::nvrtc"
)
add_library(hololink::hololink ALIAS hololink)
add_library(csi_to_bayer STATIC IMPORTED)
set_target_properties(csi_to_bayer PROPERTIES
    IMPORTED_LOCATION "${HOLOLINK_BUILD_DIR}/src/hololink/operators/csi_to_bayer/libcsi_to_bayer.a"
    INTERFACE_LINK_LIBRARIES "hololink::hololink"
)
add_library(hololink::csi_to_bayer ALIAS csi_to_bayer)

# Publisher executable
add_executable(holoscan_ros2_vb1940_publisher
    vb1940_publisher.cpp
    convert_16bit_to_8bit_kernel.cu
)

target_link_libraries(holoscan_ros2_vb1940_publisher PRIVATE
    holoscan::core
    holoscan::ops::holoscan_ros2
    holoscan::ops::bayer_demosaic
    ${rclcpp_LIBRARIES}
    ${sensor_msgs_LIBRARIES}
    # CUDA libraries
    CUDA::cudart
    CUDA::cuda_driver
    # Hololink libraries
    hololink::hololink
    hololink::csi_to_bayer
    hololink_core
    hololink_sensor
    hololink_camera_sensor
    native_vb1940_camera_sensor
    base_receiver_op
    roce_receiver
    image_processor
    # System libraries needed by hololink
    ibverbs
    fmt
)

target_include_directories(holoscan_ros2_vb1940_publisher PRIVATE
    ${rclcpp_INCLUDE_DIRS}
    ${sensor_msgs_INCLUDE_DIRS}
    ${HOLOLINK_INCLUDE_DIR}
)

target_compile_definitions(holoscan_ros2_vb1940_publisher PRIVATE HOLOLINK_AVAILABLE=1)

# Subscriber executable
add_executable(holoscan_ros2_vb1940_subscriber vb1940_subscriber.cpp)
target_link_libraries(holoscan_ros2_vb1940_subscriber
    holoscan::core
    holoscan::ops::holoscan_ros2
    holoscan::ops::holoviz
    ${rclcpp_LIBRARIES}
    ${sensor_msgs_LIBRARIES}
    CUDA::cuda_driver
    # Hololink libraries for argument parsing
    hololink_core
    hololink::hololink
)
target_include_directories(holoscan_ros2_vb1940_subscriber PRIVATE
    ${rclcpp_INCLUDE_DIRS}
    ${sensor_msgs_INCLUDE_DIRS}
    ${HOLOLINK_INCLUDE_DIR}
)
