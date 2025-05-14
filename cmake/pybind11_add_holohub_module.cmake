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

# Find pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# We fetch pybind11 since we need the same version as the Holoscan SDK
# and it's not necessarily available on all the platforms
include(FetchContent)
FetchContent_Declare(pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG v2.13.6
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(pybind11)

# Helper function to generate pybind11 operator modules
function(pybind11_add_holohub_module)
    cmake_parse_arguments(MODULE        # PREFIX
        ""                              # OPTIONS
        "CPP_CMAKE_TARGET;CLASS_NAME"   # ONEVAL
        "SOURCES"                       # MULTIVAL
        ${ARGN}
    )

    set(MODULE_NAME ${MODULE_CPP_CMAKE_TARGET})
    set(target_name ${MODULE_NAME}_python)
    pybind11_add_module(${target_name} MODULE ${MODULE_SOURCES})

    target_include_directories(${target_name}
        PUBLIC ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/pydoc
    )

    target_link_libraries(${target_name}
        PRIVATE
            holoscan::core
            ${MODULE_CPP_CMAKE_TARGET}
    )

    # Conditionally link to the ABI config target if it exists (for HSDK >= 3.3.0)
    set(pybind11_abi_details_msg "See https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_create_operator_python_bindings.html#pybind11-abi-compatibility for details")
    if(TARGET holoscan::pybind11)
        message(STATUS "${target_name}: Linking against holoscan::pybind11 to disable strict ABI protection in pybind11. ${pybind11_abi_details_msg}")
        target_link_libraries(${target_name} PRIVATE holoscan::pybind11)
    else()
        message(STATUS "${target_name}: holoscan::pybind11 target not found, using pybind11's default ABI protection. ${pybind11_abi_details_msg}")
    endif()

    # Sets the rpath of the module
    file(RELATIVE_PATH install_lib_relative_path
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR}
    )
    list(APPEND _rpath
        "\$ORIGIN/${install_lib_relative_path}" # in our install tree (same layout as src)
        "\$ORIGIN/../lib" # in our python wheel"
    )
    list(JOIN _rpath ":" _rpath)
    set_property(TARGET ${target_name}
        APPEND PROPERTY BUILD_RPATH ${_rpath}
    )
    unset(_rpath)

    # make submodule folder
    file(MAKE_DIRECTORY ${HOLOHUB_PYTHON_MODULE_OUT_DIR}/${MODULE_NAME})

    # custom target to ensure the module's __init__.py file is copied
    set(CMAKE_SUBMODULE_OUT_DIR ${HOLOHUB_PYTHON_MODULE_OUT_DIR}/${MODULE_NAME})
    configure_file(
        ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/pybind11/__init__.py
        ${HOLOHUB_PYTHON_MODULE_OUT_DIR}/${MODULE_NAME}/__init__.py
    )

    # Note: OUTPUT_NAME filename (_${MODULE_NAME}) must match the module name in the PYBIND11_MODULE macro
    set_target_properties(${target_name} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SUBMODULE_OUT_DIR}
        OUTPUT_NAME _${MODULE_NAME}
    )

endfunction()
