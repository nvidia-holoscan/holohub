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
