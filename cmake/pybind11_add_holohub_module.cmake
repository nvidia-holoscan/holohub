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

# Find pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# We fetch pybind11 since we need the same version as the Holoscan SDK
# and it's not necessarily available on all the platforms
include(FetchContent)
set(_pybind11_fetch_options)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 4.4 AND
   (NOT DEFINED HOLOHUB_SUPPRESS_DEPENDENCY_DEPRECATION_WARNINGS OR
    HOLOHUB_SUPPRESS_DEPENDENCY_DEPRECATION_WARNINGS))
  # Populate through MakeAvailable(), then add the dependency in our own
  # diagnostic scope below.
  set(_pybind11_fetch_options SOURCE_SUBDIR holohub-fetch-only)
endif()
FetchContent_Declare(pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG v2.13.6
  GIT_SHALLOW TRUE
  ${_pybind11_fetch_options}
)
unset(_pybind11_fetch_options)

# pybind11 2.13.6 uses deprecated CMake compatibility and CMP0148 OLD behavior.
# Function scope protects the caller's normal variable when CMP0126 is OLD.
function(_holohub_fetch_pybind11)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 4.4)
    FetchContent_MakeAvailable(pybind11)
    if(TARGET pybind11::module)
      return()
    endif()

    FetchContent_GetProperties(pybind11)
    if(NOT pybind11_POPULATED OR
       NOT pybind11_SOURCE_DIR OR NOT pybind11_BINARY_DIR)
      message(FATAL_ERROR "FetchContent did not provide the pybind11 directories")
    endif()

    cmake_policy(PUSH)
    cmake_policy(SET CMP0218 NEW)
    cmake_diagnostic(PUSH)
    cmake_diagnostic(SET CMD_DEPRECATED IGNORE)
    set(CMAKE_POLICY_DEFAULT_CMP0218 NEW)
    set(CMAKE_EXPORT_FIND_PACKAGE_NAME pybind11)
    set(CMAKE_VERIFY_INTERFACE_HEADER_SETS FALSE)
    set(CMAKE_VERIFY_PRIVATE_HEADER_SETS FALSE)
    add_subdirectory("${pybind11_SOURCE_DIR}" "${pybind11_BINARY_DIR}")
    cmake_diagnostic(POP)
    cmake_policy(POP)
    return()
  endif()

  set(_cache_was_defined FALSE)
  if(DEFINED CACHE{CMAKE_WARN_DEPRECATED})
    get_property(_saved_value CACHE CMAKE_WARN_DEPRECATED PROPERTY VALUE)
    get_property(_saved_type CACHE CMAKE_WARN_DEPRECATED PROPERTY TYPE)
    get_property(_saved_help CACHE CMAKE_WARN_DEPRECATED PROPERTY HELPSTRING)
    set(_cache_was_defined TRUE)
  endif()

  set(CMAKE_WARN_DEPRECATED OFF)
  set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "Suppress dependency warnings" FORCE)
  FetchContent_MakeAvailable(pybind11)

  if(_cache_was_defined)
    if(_saved_type STREQUAL "UNINITIALIZED")
      set_property(CACHE CMAKE_WARN_DEPRECATED PROPERTY VALUE "${_saved_value}")
      set_property(CACHE CMAKE_WARN_DEPRECATED PROPERTY TYPE "${_saved_type}")
      set_property(CACHE CMAKE_WARN_DEPRECATED PROPERTY HELPSTRING "${_saved_help}")
    else()
      set(CMAKE_WARN_DEPRECATED "${_saved_value}"
        CACHE "${_saved_type}" "${_saved_help}" FORCE)
    endif()
  else()
    unset(CMAKE_WARN_DEPRECATED CACHE)
  endif()
endfunction()

if(NOT DEFINED HOLOHUB_SUPPRESS_DEPENDENCY_DEPRECATION_WARNINGS OR
   HOLOHUB_SUPPRESS_DEPENDENCY_DEPRECATION_WARNINGS)
  _holohub_fetch_pybind11()
else()
  FetchContent_MakeAvailable(pybind11)
endif()

# Helper function to generate pybind11 operator modules
function(pybind11_add_holohub_module)
    cmake_parse_arguments(MODULE                                            # PREFIX
        ""                                                                  # OPTIONS
        "CPP_CMAKE_TARGET;CLASS_NAME;PYTHON_MODULE_NAME;PYTHON_NAMESPACE"   # ONEVAL
        "SOURCES"                                                           # MULTIVAL
        ${ARGN}
    )

    # PYTHON_MODULE_NAME overrides CPP_CMAKE_TARGET as the Python subpackage
    # name (directory under the package root, OUTPUT_NAME prefix, and
    # @MODULE_NAME@ in __init__.py).  Use it when the desired Python import
    # name differs from the C++ library target name.
    if(MODULE_PYTHON_MODULE_NAME)
        set(MODULE_NAME ${MODULE_PYTHON_MODULE_NAME})
    else()
        set(MODULE_NAME ${MODULE_CPP_CMAKE_TARGET})
    endif()

    # PYTHON_NAMESPACE selects the top-level Python namespace (e.g. "holoscan"
    # instead of the default "holohub").  When specified the module files are
    # placed under a namespace-specific directory rather than
    # HOLOHUB_PYTHON_MODULE_OUT_DIR, and a dedicated install() rule is added
    # so the namespace root lands on the right Python path in both wheel and
    # in-tree builds.  The top-level CMakeLists.txt install() only covers
    # HOLOHUB_PYTHON_MODULE_OUT_DIR (the holohub/ tree); modules that declare
    # their own namespace are responsible for their own install here.
    if(MODULE_PYTHON_NAMESPACE)
        if(NOT CMAKE_INSTALL_LIBDIR)
            set(CMAKE_INSTALL_LIBDIR lib)
        endif()
        if(DEFINED SKBUILD)
            # Wheel build: flat layout — namespace dir sits directly under the
            # wheel root, which pip installs straight into site-packages.
            set(_module_base_dir ${CMAKE_BINARY_DIR}/${MODULE_PYTHON_NAMESPACE})
            set(_ns_install_dest ".")
        else()
            # In-tree HoloHub build: mirror the standard python/lib/ tree so
            # the module is importable when that tree is on PYTHONPATH.
            set(_module_base_dir
                ${CMAKE_BINARY_DIR}/python/${CMAKE_INSTALL_LIBDIR}/${MODULE_PYTHON_NAMESPACE})
            set(_ns_install_dest "python/lib")
        endif()
        install(
            DIRECTORY "${_module_base_dir}"
            DESTINATION "${_ns_install_dest}"
            FILE_PERMISSIONS
                OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE
            DIRECTORY_PERMISSIONS
                OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE
            PATTERN "__pycache__" EXCLUDE
        )
    else()
        set(_module_base_dir ${HOLOHUB_PYTHON_MODULE_OUT_DIR})
    endif()

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
    set(pybind11_abi_details_msg "See https://docs.nvidia.com/holoscan/sdk-user-guide/using-the-sdk/python-operator-bindings#pybind11-abi-compatibility for details")
    if(TARGET holoscan::pybind11)
        message(STATUS "${target_name}: Linking against holoscan::pybind11 to disable strict ABI protection in pybind11. ${pybind11_abi_details_msg}")
        target_link_libraries(${target_name} PRIVATE holoscan::pybind11)
    else()
        message(STATUS "${target_name}: holoscan::pybind11 target not found, using pybind11's default ABI protection. ${pybind11_abi_details_msg}")
    endif()

    # Sets the rpath of the module. PROJECT_SOURCE_DIR (not CMAKE_SOURCE_DIR)
    # so the path is correct when this helper is invoked from a Holoscan
    # Module that's been add_subdirectory()'d into another project (HoloHub
    # consuming an external module, etc.) — CMAKE_SOURCE_DIR would point at
    # the parent project's root, which is wrong for our rpath calculation.
    file(RELATIVE_PATH install_lib_relative_path
        ${CMAKE_CURRENT_LIST_DIR}
        ${PROJECT_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR}
    )
    list(APPEND _rpath
        "\$ORIGIN/${install_lib_relative_path}" # in our install tree (same layout as src)
        "\$ORIGIN/../../lib" # in our python wheel (module at <ns>/<pkg>/_mod.so → lib/ is two levels up)
        "\$ORIGIN/../lib"    # legacy fallback for one-level-deep layouts
    )
    list(JOIN _rpath ":" _rpath)
    set_property(TARGET ${target_name}
        APPEND PROPERTY BUILD_RPATH ${_rpath}
    )
    unset(_rpath)

    # make submodule folder
    file(MAKE_DIRECTORY ${_module_base_dir}/${MODULE_NAME})

    # custom target to ensure the module's __init__.py file is copied
    set(CMAKE_SUBMODULE_OUT_DIR ${_module_base_dir}/${MODULE_NAME})
    configure_file(
        ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/pybind11/__init__.py
        ${_module_base_dir}/${MODULE_NAME}/__init__.py
    )

    # Note: OUTPUT_NAME filename (_${MODULE_NAME}) must match the module name in the PYBIND11_MODULE macro
    set_target_properties(${target_name} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SUBMODULE_OUT_DIR}
        OUTPUT_NAME _${MODULE_NAME}
    )

endfunction()
