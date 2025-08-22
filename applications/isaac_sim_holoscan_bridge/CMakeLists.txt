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

cmake_minimum_required(VERSION 3.20)

project(isaac_sim_holoscan_bridge LANGUAGES NONE)

# Add testing
if(BUILD_TESTING)
  add_custom_target(isaac_sim_holoscan_bridge_test ALL
    DEPENDS "app.py"
  )

  # Add test
  add_test(NAME isaac_sim_holoscan_bridge_python_test
    COMMAND ${CMAKE_COMMAND} -E env python3 ${CMAKE_CURRENT_SOURCE_DIR}/app.py --headless --frame-count 10
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  set_tests_properties(isaac_sim_holoscan_bridge_python_test
    PROPERTIES PASS_REGULAR_EXPRESSION "Starting simulator;Starting Holoscan pipeline;Simulation App Startup Complete;Simulator finished;Transformer application finished"
    # Omniverse Kit prints the error:
    #   1: 2025-06-12 08:15:39 [9,453ms] [Error] [carb.graphics-vulkan.plugin] Could not get NGX parameters block because NGX isn't enabled.
    #   1: 2025-06-12 08:15:39 [9,453ms] [Error] [carb.graphics-vulkan.plugin] Failed to create NGX context.
    # ignore NGX errors.
    FAIL_REGULAR_EXPRESSION "^(?!.*NGX).*([^a-z]Error;error"
    # Omniverse Kit seg-faults on exit, ignore return code
    SKIP_RETURN_CODE 1)

endif()
