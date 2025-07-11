# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

include(CMakeParseArguments)

# Generate a C++ header file from a text or binary file.
#
# The header file will contain a uint8_t array and will be generated in
#
#  ${CMAKE_CURRENT_BINARY_DIR}\DIR\ARRAY_NAME.hpp
#
# The array name is build from FILE_PATH by using the file name and replacing `-`, and `.` by `_`.
# Example usage in CMakeLists.txt:
#
#   add_library(fonts_target SHARED)
#   gen_header_from_file(TARGET fonts_target FILE_PATH `fonts\Roboto-Bold.ttf` TYPE binary)
#
# This will generate the `fonts\roboto_bold_ttf.hpp` in the current binary dir. The file contains an
# array named `roboto_bold_ttf`. The file added to the project and the current binary dir is added
# to the include paths.
# To use the generated header in the code:
#
#   #include <fonts\roboto_bold_ttf.hpp>
#   void func() {
#       cout << roboto_bold_ttf[0] << std::endl;
#   }
#
# Usage:
#
#     gen_header_from_file (FILE_PATH <PATH> [TARGET <TGT>] [TYPE <TYPE>])
#
#   ``FILE_PATH``
#     file to convert to a header relative to CMAKE_CURRENT_SOURCE_DIR
#   ``TYPE``
#     type of the file to convert to a header, either "binary" or "text"
#   ``TARGET``
#     if specified the generated header file and the include path is added to the target

function(gen_header_from_file)
    set(_options)
    set(_singleargs FILE_PATH TARGET TYPE)
    set(_multiargs)

    cmake_parse_arguments(gen_header "${_options}" "${_singleargs}" "${_multiargs}" "${ARGN}")

    # read the file
    set(SOURCE_FILE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${gen_header_FILE_PATH}")

    if(gen_header_TYPE STREQUAL "binary")
        file(READ "${SOURCE_FILE_PATH}" FILE_CONTENT HEX)

        # separate into individual bytes
        string(REGEX MATCHALL "([A-Fa-f0-9][A-Fa-f0-9])" FILE_CONTENT ${FILE_CONTENT})

        # add 0x and commas
        list(JOIN FILE_CONTENT ", 0x" FILE_CONTENT)
        string(PREPEND FILE_CONTENT "0x")
    elseif(gen_header_TYPE STREQUAL "text")
        file(READ "${SOURCE_FILE_PATH}" FILE_CONTENT)
    else()
        message(FATAL_ERROR "Invalid type: ${gen_header_TYPE}")
    endif()

    # build the array name
    get_filename_component(FILE_ARRAY_NAME ${gen_header_FILE_PATH} NAME)
    string(TOLOWER ${FILE_ARRAY_NAME} FILE_ARRAY_NAME)
    string(REPLACE "-" "_" FILE_ARRAY_NAME ${FILE_ARRAY_NAME})
    string(REPLACE "." "_" FILE_ARRAY_NAME ${FILE_ARRAY_NAME})

    # add a dependency to re-run configure when the file changes
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${SOURCE_FILE_PATH}")

    # create the header file
    get_filename_component(FILE_DIRECTORY ${gen_header_FILE_PATH} DIRECTORY)
    set(HEADER_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/${FILE_DIRECTORY}/${FILE_ARRAY_NAME}.hpp")
    configure_file(
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/gen_header_from_file_template/header_${gen_header_TYPE}.hpp.in"
        ${HEADER_FILE_NAME}
    )

    if(gen_header_TARGET)
        # add the created file to the target
        target_sources(${gen_header_TARGET}
            PRIVATE
                ${HEADER_FILE_NAME}
            )
        # also add the binary dir to include directories so the header can be found
        target_include_directories(${gen_header_TARGET}
            PRIVATE
                $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
            )
    endif()

    message(STATUS "Created header ${HEADER_FILE_NAME} from file '${SOURCE_FILE_PATH}'")

endfunction()
