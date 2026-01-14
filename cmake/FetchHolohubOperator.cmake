# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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



# Fetch a single operator from the Holohub repository
#
# This function fetches a specific operator from the Holohub repository using sparse checkout.
# It clones only the required operator directory, making the download process more efficient.
#
# Parameters:
#   OPERATOR_NAME - The name of the operator to fetch
#
# Optional Parameters:
#   PATH - The path to the operator within the Holohub repository (defaults to OPERATOR_NAME)
#   REPO_URL - The URL of the Holohub repository (defaults to git@github.com:nvidia-holoscan/holohub.git)
#   BRANCH - The branch to checkout (defaults to "main")
#   DEPTH - Git clone depth (defaults to 1 for shallow clone, use 0 for full history)
#   DISABLE_PYTHON - Whether to build Python bindings
#   PATCH_COMMAND - Custom command to run after checkout (e.g., to apply patches)
#
# Note: When using PATCH_COMMAND with patches created from older commits, you may need to increase
#       DEPTH or set it to 0 to include sufficient git history for the patch to apply correctly.
#
# Example:
#   fetch_holohub_operator(realsense_camera)
#   fetch_holohub_operator(dds_operator_base PATH dds/base)
#   fetch_holohub_operator(custom_operator REPO_URL "https://github.com/custom/holohub.git")
#   fetch_holohub_operator(custom_operator BRANCH "dev")
#   fetch_holohub_operator(custom_operator DEPTH 0)
#   fetch_holohub_operator(custom_operator DISABLE_PYTHON)
#   fetch_holohub_operator(custom_operator PATCH_COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/fix.patch)
function(fetch_holohub_operator OPERATOR_NAME)

  cmake_parse_arguments(ARGS "DISABLE_PYTHON" "PATH;REPO_URL;BRANCH;DEPTH" "PATCH_COMMAND" ${ARGN})

  if(NOT ARGS_REPO_URL)
    set(ARGS_REPO_URL "https://github.com/nvidia-holoscan/holohub.git")
  endif()

  if(NOT ARGS_PATH)
    set(ARGS_PATH "${OPERATOR_NAME}")
  endif()

  if(NOT ARGS_BRANCH)
    set(ARGS_BRANCH "main")
  endif()

  if(NOT DEFINED ARGS_DEPTH)
    set(ARGS_DEPTH 1)
  endif()

  # Build git depth argument as a list (so it expands as separate arguments)
  if(ARGS_DEPTH EQUAL 0)
    set(DEPTH_ARG "")
  else()
    set(DEPTH_ARG --depth ${ARGS_DEPTH})
  endif()

  if(NOT ARGS_DISABLE_PYTHON)
    set(HOLOHUB_BUILD_PYTHON ON)
  endif()

  # Fetch Holohub repository
  include(FetchContent)
  FetchContent_Declare(
      holohub_${OPERATOR_NAME}
      SOURCE_DIR "${FETCHCONTENT_BASE_DIR}/holohub_${OPERATOR_NAME}-prefix/src"
      DOWNLOAD_COMMAND
        git clone ${DEPTH_ARG} --branch ${ARGS_BRANCH} --no-checkout ${ARGS_REPO_URL} "${FETCHCONTENT_BASE_DIR}/holohub_${OPERATOR_NAME}-prefix/src"
        && cd "${FETCHCONTENT_BASE_DIR}/holohub_${OPERATOR_NAME}-prefix/src"
        && git sparse-checkout set --no-cone
            operators/${ARGS_PATH}
            cmake/pybind11_add_holohub_module.cmake
            cmake/nvidia_video_codec.cmake
            cmake/pybind11/
            cmake/pydoc/
            operators/operator_util.hpp
        && git checkout ${ARGS_BRANCH}
        # Write a CMakeLists.txt in the operators directory to set CMAKE_MODULE_PATH and add the operator subdirectory
        && echo "list(APPEND CMAKE_MODULE_PATH \\$\\{CMAKE_CURRENT_LIST_DIR\\}/cmake)" > "${FETCHCONTENT_BASE_DIR}/holohub_${OPERATOR_NAME}-prefix/src/CMakeLists.txt"
        && echo "if(HOLOHUB_BUILD_PYTHON)" >> "${FETCHCONTENT_BASE_DIR}/holohub_${OPERATOR_NAME}-prefix/src/CMakeLists.txt"
        && echo "  if(NOT CMAKE_INSTALL_LIBDIR)" >> "${FETCHCONTENT_BASE_DIR}/holohub_${OPERATOR_NAME}-prefix/src/CMakeLists.txt"
        && echo "    set(CMAKE_INSTALL_LIBDIR lib)" >> "${FETCHCONTENT_BASE_DIR}/holohub_${OPERATOR_NAME}-prefix/src/CMakeLists.txt"
        && echo "  endif()" >> "${FETCHCONTENT_BASE_DIR}/holohub_${OPERATOR_NAME}-prefix/src/CMakeLists.txt"
        && echo "  set(HOLOHUB_PYTHON_MODULE_OUT_DIR \\$\\{CMAKE_BINARY_DIR\\}/python/\\$\\{CMAKE_INSTALL_LIBDIR\\}/holohub)" >> "${FETCHCONTENT_BASE_DIR}/holohub_${OPERATOR_NAME}-prefix/src/CMakeLists.txt"
        && echo "  file(MAKE_DIRECTORY \\$\\{HOLOHUB_PYTHON_MODULE_OUT_DIR\\})" >> "${FETCHCONTENT_BASE_DIR}/holohub_${OPERATOR_NAME}-prefix/src/CMakeLists.txt"
        && echo "endif()" >> "${FETCHCONTENT_BASE_DIR}/holohub_${OPERATOR_NAME}-prefix/src/CMakeLists.txt"
        && echo "add_subdirectory(operators/${ARGS_PATH})" >> "${FETCHCONTENT_BASE_DIR}/holohub_${OPERATOR_NAME}-prefix/src/CMakeLists.txt"
      PATCH_COMMAND ${ARGS_PATCH_COMMAND}
  )

  FetchContent_MakeAvailable(holohub_${OPERATOR_NAME})

endfunction()
