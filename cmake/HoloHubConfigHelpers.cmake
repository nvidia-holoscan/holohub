# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

#[=======================================================================[.rst:
add_holohub_application
------------------------

Declare ``APP_<name>`` and add an application when it or ``BUILD_ALL`` is
enabled. Required operators are force-enabled for the later operator
discovery pass.

.. code-block:: cmake

  add_holohub_application(my_app
      DEPENDS OPERATORS required_operator)
#]=======================================================================]
function(add_holohub_application NAME)
    cmake_parse_arguments(APP "" "" "DEPENDS" ${ARGN})

    set(app_option "APP_${NAME}")
    option(${app_option} "Build the ${NAME} application" ${BUILD_ALL})
    if(NOT ${app_option})
        return()
    endif()

    add_subdirectory(${NAME})

    if(APP_DEPENDS)
        cmake_parse_arguments(DEPS "" "" "EXTENSIONS;OPERATORS" ${APP_DEPENDS})
        foreach(dependency IN LISTS DEPS_EXTENSIONS)
            set("EXT_${dependency}" ON CACHE BOOL "Build the ${dependency} extension" FORCE)
        endforeach()
        foreach(dependency IN LISTS DEPS_OPERATORS)
            set("OP_${dependency}" ON CACHE BOOL "Build the ${dependency} operator" FORCE)
        endforeach()
    endif()
endfunction()

#[=======================================================================[.rst:
add_holohub_operator
--------------------

Declare ``OP_<name>`` and add an operator when it or ``BUILD_ALL`` is
enabled. Operator and extension dependencies are force-enabled.

.. code-block:: cmake

  add_holohub_operator(my_operator)
#]=======================================================================]
function(add_holohub_operator NAME)
    cmake_parse_arguments(OP "" "" "DEPENDS" ${ARGN})

    set(operator_option "OP_${NAME}")
    option(${operator_option} "Build the ${NAME} operator" ${BUILD_ALL})
    if(NOT ${operator_option})
        return()
    endif()

    add_subdirectory(${NAME})

    if(OP_DEPENDS)
        cmake_parse_arguments(DEPS "" "" "EXTENSIONS;OPERATORS" ${OP_DEPENDS})
        foreach(dependency IN LISTS DEPS_EXTENSIONS)
            set("EXT_${dependency}" ON CACHE BOOL "Build the ${dependency} extension" FORCE)
        endforeach()
        foreach(dependency IN LISTS DEPS_OPERATORS)
            set("OP_${dependency}" ON CACHE BOOL "Build the ${dependency} operator" FORCE)
        endforeach()
    endif()
endfunction()
