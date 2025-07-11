# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set(RTI_CONNEXT_DDS_DIR "$ENV{NDDSHOME}" CACHE PATH "RTI Connext DDS Path")

list(APPEND CMAKE_MODULE_PATH ${RTI_CONNEXT_DDS_DIR}/resource/cmake)
find_package(RTIConnextDDS REQUIRED)

# Uses the rtiddsgen tool to generate source code from an DDS IDL type specification.
function(rticodegen)
  set(options UNBOUNDED)
  set(single_value_args IDL_FILE OUTPUT_DIRECTORY)
  set(multi_value_args)
  cmake_parse_arguments(
    _RTICODEGEN
    "${options}"
    "${single_value_args}"
    "${multi_value_args}"
    "${ARGN}"
  )

  get_filename_component(idl_basename "${_RTICODEGEN_IDL_FILE}" NAME_WE)
  set(sources
    "${_RTICODEGEN_OUTPUT_DIRECTORY}/${idl_basename}.cxx"
    "${_RTICODEGEN_OUTPUT_DIRECTORY}/${idl_basename}Plugin.cxx"
  )
  set(headers
    "${_RTICODEGEN_OUTPUT_DIRECTORY}/${idl_basename}.hpp"
    "${_RTICODEGEN_OUTPUT_DIRECTORY}/${idl_basename}Plugin.hpp"
  )
  set(${idl_basename}_SOURCES ${sources} PARENT_SCOPE)
  set(${idl_basename}_HEADERS ${headers} PARENT_SCOPE)
  set(_RTICODEGEN_OUTPUTS ${sources} ${headers} PARENT_SCOPE)

  set(extra_flags)
  if(_RTICODEGEN_UNBOUNDED)
    list(APPEND extra_flags "-unboundedSupport")
  endif()

  add_custom_command(
    OUTPUT
      ${sources} ${headers}
    COMMAND
      ${CMAKE_COMMAND} -E rm -rf ${_RTICODEGEN_OUTPUT_DIRECTORY}
    COMMAND
      ${CMAKE_COMMAND} -E make_directory ${_RTICODEGEN_OUTPUT_DIRECTORY}
    COMMAND
      ${RTICODEGEN} -language c++11 -d ${_RTICODEGEN_OUTPUT_DIRECTORY} ${extra_flags} ${_RTICODEGEN_IDL_FILE}
    DEPENDS
      ${_RTICODEGEN_IDL_FILE}
  )
endfunction()

# Adds a shared library for a DDS IDL type specification.
function(add_rti_type_library target_name idl_file)
  rticodegen(
    IDL_FILE ${idl_file}
    OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rti_gen
    ${ARGN}
  )
  add_library(${target_name} SHARED ${_RTICODEGEN_OUTPUTS})
  target_link_libraries(${target_name} PUBLIC RTIConnextDDS::cpp2_api)
  target_include_directories(${target_name} PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/rti_gen)
endfunction()
