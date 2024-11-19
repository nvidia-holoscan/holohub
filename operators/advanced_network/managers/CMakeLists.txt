# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Define targets for each manager type
foreach(MGR IN LISTS ANO_MGR_LIST)
    string(TOUPPER ${MGR} MGR_UPPER)
    string(TOLOWER ${MGR} MGR_LOWER)
    message(STATUS "Adding manager ${MGR_UPPER} to the configuration")
    project(ano_manager_${MGR_LOWER} LANGUAGES CXX CUDA)
    add_library(${PROJECT_NAME} SHARED)
    add_subdirectory(${MGR_LOWER})
    target_compile_definitions(${PROJECT_NAME} PUBLIC "ANO_MGR_${MGR_UPPER}=1")
    target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
    install(TARGETS ${PROJECT_NAME})
endforeach()
