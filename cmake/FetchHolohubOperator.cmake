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
#   DISABLE_PYTHON - Whether to build Python bindings
#
# Example:
#   fetch_holohub_operator(realsense_camera)
#   fetch_holohub_operator(dds_operator_base PATH dds/base)
#   fetch_holohub_operator(custom_operator REPO_URL "https://github.com/custom/holohub.git")
#   fetch_holohub_operator(custom_operator BRANCH "dev")
#   fetch_holohub_operator(custom_operator DISABLE_PYTHON)
function(fetch_holohub_operator OPERATOR_NAME)

  cmake_parse_arguments(ARGS "DISABLE_PYTHON" "PATH;REPO_URL;BRANCH" "" ${ARGN})

  if(NOT ARGS_REPO_URL)
    set(ARGS_REPO_URL "https://github.com/nvidia-holoscan/holohub.git")
  endif()

  if(NOT ARGS_PATH)
    set(ARGS_PATH "${OPERATOR_NAME}")
  endif()

  if(NOT ARGS_BRANCH)
    set(ARGS_BRANCH "main")
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
        git clone --depth 1 --no-checkout ${ARGS_REPO_URL} "${FETCHCONTENT_BASE_DIR}/holohub_${OPERATOR_NAME}-prefix/src"
        && cd "${FETCHCONTENT_BASE_DIR}/holohub_${OPERATOR_NAME}-prefix/src"
        && git sparse-checkout set --no-cone
            operators/${ARGS_PATH}
            cmake/pybind11_add_holohub_module.cmake
            cmake/pybind11/
            cmake/pydoc/
            operators/operator_util.hpp
        && git checkout ${ARGS_BRANCH}
        # Write a CMakeLists.txt in the operators directory to set CMAKE_MODULE_PATH and add the operator subdirectory
        && echo "list(APPEND CMAKE_MODULE_PATH \\$\\{CMAKE_CURRENT_LIST_DIR\\}/cmake)" > "${FETCHCONTENT_BASE_DIR}/holohub_${OPERATOR_NAME}-prefix/src/CMakeLists.txt"
        # Add code to create python module output directory if building python
        && echo "if(HOLOHUB_BUILD_PYTHON)" >> "${FETCHCONTENT_BASE_DIR}/holohub_${OPERATOR_NAME}-prefix/src/CMakeLists.txt"
        && echo "  if(NOT CMAKE_INSTALL_LIBDIR)" >> "${FETCHCONTENT_BASE_DIR}/holohub_${OPERATOR_NAME}-prefix/src/CMakeLists.txt"
        && echo "    set(CMAKE_INSTALL_LIBDIR lib)" >> "${FETCHCONTENT_BASE_DIR}/holohub_${OPERATOR_NAME}-prefix/src/CMakeLists.txt"
        && echo "  endif()" >> "${FETCHCONTENT_BASE_DIR}/holohub_${OPERATOR_NAME}-prefix/src/CMakeLists.txt"
        && echo "  set(HOLOHUB_PYTHON_MODULE_OUT_DIR \\$\\{CMAKE_BINARY_DIR\\}/python/\\$\\{CMAKE_INSTALL_LIBDIR\\}/holohub)" >> "${FETCHCONTENT_BASE_DIR}/holohub_${OPERATOR_NAME}-prefix/src/CMakeLists.txt"
        && echo "  file(MAKE_DIRECTORY \\$\\{HOLOHUB_PYTHON_MODULE_OUT_DIR\\})" >> "${FETCHCONTENT_BASE_DIR}/holohub_${OPERATOR_NAME}-prefix/src/CMakeLists.txt"
        && echo "endif()" >> "${FETCHCONTENT_BASE_DIR}/holohub_${OPERATOR_NAME}-prefix/src/CMakeLists.txt"
        && echo "add_subdirectory(operators/${ARGS_PATH})" >> "${FETCHCONTENT_BASE_DIR}/holohub_${OPERATOR_NAME}-prefix/src/CMakeLists.txt"

  )

  FetchContent_MakeAvailable(holohub_${OPERATOR_NAME})

endfunction()
