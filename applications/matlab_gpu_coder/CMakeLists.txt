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

cmake_minimum_required(VERSION 3.20)
project(matlab_gpu_coder CXX CUDA)

option(BUILD_IMAGE_PROCESSING "Build image processing" OFF)
option(BUILD_BEAMFORM "Build beamforming application" OFF)

add_subdirectory(matlab_utils)
if(BUILD_IMAGE_PROCESSING)
    add_subdirectory(matlab_image_processing)
endif()
if(BUILD_BEAMFORM)
    add_subdirectory(matlab_beamform)
endif()
