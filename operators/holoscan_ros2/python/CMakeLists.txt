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

# Files to include in our module
list(APPEND holoscan_ros2_python_files
    holoscan_ros2/__init__.py
    holoscan_ros2/bridge.py
    holoscan_ros2/operator.py
    holoscan_ros2/operators/__init__.py
    holoscan_ros2/operators/publisher.py
    holoscan_ros2/operators/subscriber.py
)

# Define paths to copy in the holohub python module
list(TRANSFORM holoscan_ros2_python_files
    PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/
    OUTPUT_VARIABLE holoscan_ros2_python_src_files
)
set(holoscan_ros2_python_out_dir ${HOLOHUB_PYTHON_MODULE_OUT_DIR}/holoscan_ros2)
file(MAKE_DIRECTORY ${holoscan_ros2_python_out_dir})
file(MAKE_DIRECTORY ${holoscan_ros2_python_out_dir}/operators)
list(TRANSFORM holoscan_ros2_python_files
    PREPEND ${holoscan_ros2_python_out_dir}/
    OUTPUT_VARIABLE holoscan_ros2_python_out_files
)

# Target to copy
add_custom_target(holoscan_ros2_python ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/holoscan_ros2 ${holoscan_ros2_python_out_dir}
    DEPENDS ${holoscan_ros2_python_src_files}
    BYPRODUCTS ${holoscan_ros2_python_out_files}
)
