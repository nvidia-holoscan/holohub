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

# holoscan_ros2 related applications
# In each subdirectory, there is a set of related applications.
# For example, pubsub directory contains applications that publish and subscribe to ROS2 topics.
# vb1940 directory contains applications that publish and subscribe images from the vb1940 sensor.
add_holohub_application(pubsub DEPENDS OPERATORS holoscan_ros2)
add_holohub_application(vb1940 DEPENDS OPERATORS holoscan_ros2)
