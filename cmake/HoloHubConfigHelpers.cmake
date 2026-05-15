# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# HoloHub Configuration Helpers
# =============================
#
# This file provides CMake helper functions for building HoloHub packages, applications,
# operators, and extensions. These functions simplify the build configuration process
# and handle dependency management automatically.
#
# Available Functions:
# - add_holohub_package(): Build packages with dependencies
# - add_holohub_application(): Build applications with operator/extension dependencies
# - add_holohub_operator(): Build operators with extension dependencies
# - add_holohub_extension(): Build extensions
# - holohub_declare_external_module(): Declare an external Holoscan Module and register
#   its operators with HoloHub's lazy-fetch post-step
#
# Global Variables:
# - BUILD_ALL: Global flag to enable/disable all components (default: OFF)
# - HOLOHUB_BUILD_OPERATORS: List of operators to build when optional dependencies are specified
#
# Usage Examples:
#   add_holohub_package(my_package EXTENSIONS gxf_core OPERATORS my_op APPLICATIONS my_app)
#   add_holohub_application(my_app DEPENDS EXTENSIONS gxf_core OPERATORS my_op)
#   add_holohub_operator(my_op DEPENDS EXTENSIONS gxf_core)
#   add_holohub_extension(my_ext)

# =====================================================
# Helper function to build packages
# =====================================================
# Builds a package and automatically enables its dependencies.
#
# Parameters:
#   NAME: The name of the package to build
#
# Keyword Arguments:
#   EXTENSIONS: List of GXF extensions that this package depends on
#   OPERATORS: List of Holoscan operators that this package depends on
#   APPLICATIONS: List of applications that this package depends on
#
# Creates:
#   PKG_${NAME}: CMake option to enable/disable this package
#
# Example:
#   add_holohub_package(my_package
#     EXTENSIONS gxf_core gxf_serialization
#     OPERATORS my_operator
#     APPLICATIONS my_application
#   )
function(add_holohub_package NAME)
  set(pkgname "PKG_${NAME}")
  option(${pkgname} "Build the ${NAME} package" ${BUILD_ALL})

  message(DEBUG "${pkgname} = ${${pkgname}}")

  # Configure the package if enabled
  if(NOT ${pkgname})
    return()
  endif()
  add_subdirectory(${NAME})

  # If we have dependencies make sure they are built
  cmake_parse_arguments(DEPS "" "" "EXTENSIONS;OPERATORS;APPLICATIONS" ${ARGN})
  message(DEBUG "${pkgname} exts = ${DEPS_EXTENSIONS}")
  message(DEBUG "${pkgname} ops = ${DEPS_OPERATORS}")
  message(DEBUG "${pkgname} apps = ${DEPS_APPLICATIONS}")
  foreach(dep IN LISTS DEPS_EXTENSIONS)
    set("EXT_${dep}" ON CACHE BOOL "Build the ${dep} GXF extension" FORCE)
  endforeach()
  foreach(dep IN LISTS DEPS_OPERATORS)
    set("OP_${dep}" ON CACHE BOOL "Build the ${dep} holoscan operator" FORCE)
  endforeach()
  foreach(dep IN LISTS DEPS_APPLICATIONS)
    set("APP_${dep}" ON CACHE BOOL "Build the ${dep} application" FORCE)
  endforeach()
endfunction()

# =====================================================
# Helper function to build application and dependencies
# =====================================================
# Builds an application and automatically enables its required dependencies.
# Supports optional operator dependencies based on HOLOHUB_BUILD_OPERATORS.
#
# Parameters:
#   NAME: The name of the application to build
#
# Keyword Arguments:
#   DEPENDS: Dependency specification with sub-arguments:
#     EXTENSIONS: List of GXF extensions that this application depends on
#     OPERATORS: List of Holoscan operators that this application depends on
#               Use "OPTIONAL" keyword to make subsequent operators optional
#
# Creates:
#   APP_${NAME}: CMake option to enable/disable this application
#
# Example:
#   add_holohub_application(my_app
#     DEPENDS
#       EXTENSIONS gxf_core gxf_serialization
#       OPERATORS required_op OPTIONAL optional_op1 optional_op2
#   )
function(add_holohub_application NAME)

  cmake_parse_arguments(APP "" "" "DEPENDS" ${ARGN})

  set(appname "APP_${NAME}")
  option(${appname} "Build the ${NAME} application" ${BUILD_ALL})

  if(${appname})
    add_subdirectory(${NAME})

    # If we have dependencies make sure they are built
    if(APP_DEPENDS)
      cmake_parse_arguments(DEPS "" "" "EXTENSIONS;OPERATORS" ${APP_DEPENDS})

      foreach(dependency IN LISTS DEPS_EXTENSIONS)
        set("EXT_${dependency}" ON CACHE BOOL "Build the ${dependency}" FORCE)
      endforeach()

      unset(op_optional)
      foreach(dependency IN LISTS DEPS_OPERATORS)

        # Handle optional operator dependencies
        if(dependency STREQUAL "OPTIONAL")
          set(op_optional 1)
          continue()
        endif()

        if(op_optional)
          string(REPLACE "\"" "" holohub_build_operators "${HOLOHUB_BUILD_OPERATORS}")
          if(${dependency} IN_LIST holohub_build_operators)
            set("OP_${dependency}" ON CACHE BOOL "Build the ${dependency}" FORCE)
          endif()
        else()
          set("OP_${dependency}" ON CACHE BOOL "Build the ${dependency}" FORCE)
        endif()
      endforeach()
    endif()

  endif()

endfunction()

# =====================================================
# Helper function to build operators
# =====================================================
# Builds a Holoscan operator and automatically enables its extension and operator dependencies.
#
# Parameters:
#   NAME: The name of the operator to build
#
# Keyword Arguments:
#   DEPENDS: Dependency specification with sub-arguments:
#     EXTENSIONS: List of GXF extensions that this operator depends on
#     OPERATORS: List of Holoscan operators that this operator depends on
#
# Creates:
#   OP_${NAME}: CMake option to enable/disable this operator
#
# Example:
#   add_holohub_operator(my_op
#     DEPENDS EXTENSIONS gxf_core gxf_serialization
#   )
function(add_holohub_operator NAME)

  cmake_parse_arguments(OP "" "" "DEPENDS" ${ARGN})

  set(opname "OP_${NAME}")
  option(${opname} "Build the ${NAME} operator" ${BUILD_ALL})

  if(${opname})
    add_subdirectory(${NAME})

    # If we have dependencies make sure they are built
    if(OP_DEPENDS)
      cmake_parse_arguments(DEPS "" "" "EXTENSIONS;OPERATORS" ${OP_DEPENDS})

      foreach(dependency IN LISTS DEPS_EXTENSIONS)
        set("EXT_${dependency}" ON CACHE BOOL "Build the ${dependency}" FORCE)
      endforeach()

      foreach(dependency IN LISTS DEPS_OPERATORS)
        set("OP_${dependency}" ON CACHE BOOL "Build the ${dependency} operator" FORCE)
      endforeach()

    endif()

  endif()
endfunction()

# =====================================================
# Helper function to build extensions
# =====================================================
# Builds a GXF extension. This is the simplest helper function with no dependencies.
#
# Parameters:
#   NAME: The name of the extension to build
#
# Creates:
#   EXT_${NAME}: CMake option to enable/disable this extension
#
# Example:
#   add_holohub_extension(my_extension)
function(add_holohub_extension NAME)
  set(extname "EXT_${NAME}")
  option(${extname} "Build the ${NAME} extension" ${BUILD_ALL})

  if(${extname})
    add_subdirectory(${NAME})
  endif()
endfunction()

# =====================================================
# Helper function to declare external Holoscan Modules
# =====================================================
# Declares an external Holoscan Module dependency and registers its operators with
# HoloHub's lazy-fetch post-step. Equivalent to calling FetchContent_Declare followed
# by setting HOLOHUB_EXT_OP_<op>_PROVIDER for each advertised operator.
#
# The HoloHub CLI generates calls to this function automatically from a consumer's
# metadata.json into ${CMAKE_BINARY_DIR}/external_operators_manifest.cmake. Use this
# function directly when bypassing the CLI.
#
# Parameters:
#   PROVIDER: CMake-safe identifier for the module (used as the FetchContent name and
#             in ${PROVIDER}_SOURCE_DIR etc.). Prefer underscores over hyphens.
#
# Keyword Arguments:
#   PROVIDES_OPERATORS: Operators this module supplies. The root CMakeLists.txt
#                       post-step calls FetchContent_MakeAvailable for this module
#                       only when at least one of these operators is OP_<op>=ON.
#   <FetchContent_Declare args>: All remaining arguments are forwarded verbatim to
#                       FetchContent_Declare(PROVIDER ...). Any option accepted by
#                       FetchContent_Declare (GIT_REPOSITORY, GIT_TAG, SOURCE_DIR,
#                       GIT_SHALLOW, etc.) is valid here.
#
# HOLOHUB_EXT_OP_<op>_PROVIDER variables are set as NORMAL (non-cache) variables.
# They must be set fresh each configure run; a cached entry whose FetchContent_Declare
# was not registered in the current run would cause FetchContent_MakeAvailable to fail
# with "No content details recorded for <provider>".
#
# Example:
#   holohub_declare_external_module(holoscan_deltacast
#       GIT_REPOSITORY  https://github.com/nvidia/holoscan-deltacast
#       GIT_TAG         2dac97236a8b3689ab08b5bc0b5a319e0558c807
#       PROVIDES_OPERATORS deltacast_videomaster
#   )
#
# For local development, set FETCHCONTENT_SOURCE_DIR_<UPPER_PROVIDER> before calling
# this function to redirect FetchContent at a local working copy:
#   set(FETCHCONTENT_SOURCE_DIR_HOLOSCAN_DELTACAST "/path/to/local" CACHE PATH "" FORCE)
#   holohub_declare_external_module(holoscan_deltacast
#       SOURCE_DIR  "/path/to/local"
#       PROVIDES_OPERATORS deltacast_videomaster
#   )
function(holohub_declare_external_module PROVIDER)
  cmake_parse_arguments(ARG "" "" "PROVIDES_OPERATORS" ${ARGN})
  include(FetchContent)
  FetchContent_Declare(${PROVIDER} ${ARG_UNPARSED_ARGUMENTS})
  foreach(_op IN LISTS ARG_PROVIDES_OPERATORS)
    set("HOLOHUB_EXT_OP_${_op}_PROVIDER" "${PROVIDER}" PARENT_SCOPE)
  endforeach()
endfunction()
