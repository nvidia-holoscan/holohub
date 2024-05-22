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

# Helper function to build application and dependencies
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
          if(${dependency} IN_LIST HOLOHUB_BUILD_OPERATORS)
            set("OP_${dependency}" ON CACHE BOOL "Build the ${dependency}" FORCE)
          endif()
        else()
          set("OP_${dependency}" ON CACHE BOOL "Build the ${dependency}" FORCE)
        endif()
      endforeach()
    endif()

  endif()

endfunction()

# Helper function to build operators
function(add_holohub_operator NAME)

  cmake_parse_arguments(OP "" "" "DEPENDS" ${ARGN})

  set(opname "OP_${NAME}")
  option(${opname} "Build the ${NAME} operator" ${BUILD_ALL})

  if(${opname})
    add_subdirectory(${NAME})

    # If we have dependencies make sure they are built
    if(OP_DEPENDS)
      cmake_parse_arguments(DEPS "" "" "EXTENSIONS" ${OP_DEPENDS})

      foreach(dependency IN LISTS DEPS_EXTENSIONS)
        set("EXT_${dependency}" ON CACHE BOOL "Build the ${dependency}" FORCE)
      endforeach()

    endif()

  endif()
endfunction()

# Helper function to build extensions
function(add_holohub_extension NAME)
  set(extname "EXT_${NAME}")
  option(${extname} "Build the ${NAME} extension" ${BUILD_ALL})

  if(${extname})
    add_subdirectory(${NAME})
  endif()
endfunction()
