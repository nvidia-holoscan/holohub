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
project(surgical_scene_recon LANGUAGES NONE)

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------

# Holoscan
find_package(holoscan 3.7 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")


# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------

if(BUILD_TESTING)
    # Test Surgical Scene Reconstruction Training
    add_test(NAME surgical_scene_recon_training_test
            COMMAND ./holohub run surgical_scene_recon verify_train
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    set_tests_properties(surgical_scene_recon_training_test PROPERTIES
        TIMEOUT 600
        PASS_REGULAR_EXPRESSION "SUCCESS!"
    )

    # Test Surgical Scene Reconstruction Inference
    add_test(NAME surgical_scene_recon_inference_test
            COMMAND ./holohub run surgical_scene_recon verify_inference
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    set_tests_properties(surgical_scene_recon_inference_test PROPERTIES
        TIMEOUT 600
        PASS_REGULAR_EXPRESSION "Dynamic visualization completed successfully!"
    )
endif()
