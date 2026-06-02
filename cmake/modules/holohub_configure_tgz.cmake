# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# holohub_configure_tgz(NAME <name> VERSION <version>
#                       [EXPORT_NAME <export>]
#                       [COMPONENTS <comp> ...])
#
# Generates a CPack TGZ configuration file at
#   ${CMAKE_BINARY_DIR}/pkg/CPackConfig-<name>-TGZ.cmake
#
# The resulting tarball follows the KitMaker multi-variant naming convention:
#   <name_underscored>-<os>-<arch>-<version>.tar.gz
# with archive root <name_underscored>/ and library variant subdirectory
# lib/<cuda_major>/ (e.g. lib/13/) per the multi-variant layout.
#
# EXPORT_NAME: when provided, generates and installs cmake config/version files
#   for the named export target, mirroring holohub_configure_deb behaviour.
#
# COMPONENTS: when provided, only those cmake install components (plus the
#   cmake export component, if EXPORT_NAME is set) are included in the archive.
#   When omitted, CMAKE_INSTALL_DEFAULT_COMPONENT_NAME is used instead.
function(holohub_configure_tgz)
  set(requiredArgs NAME VERSION)
  set(oneValueArgs NAME VERSION EXPORT_NAME)
  set(multiValueArgs COMPONENTS)
  cmake_parse_arguments(ARG "" "${oneValueArgs}" "${multiValueArgs}" ${ARGV})

  foreach(arg ${requiredArgs})
    if(NOT ARG_${arg})
      list(APPEND _missing ${arg})
    endif()
  endforeach()
  if(_missing)
    message(FATAL_ERROR "holohub_configure_tgz: missing required arguments: ${_missing}")
  endif()

  if(ARG_EXPORT_NAME)
    set(config_install_dir "lib/cmake/${ARG_NAME}")
    set(export_component ${ARG_NAME}-cmake)
    install(
      EXPORT ${ARG_EXPORT_NAME}
      DESTINATION ${config_install_dir}
      NAMESPACE holoscan::
      COMPONENT ${export_component}
    )
    include(CMakePackageConfigHelpers)
    configure_package_config_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/Config.cmake.in"
      "${CMAKE_CURRENT_BINARY_DIR}/${ARG_NAME}Config.cmake"
      INSTALL_DESTINATION ${config_install_dir}
      NO_SET_AND_CHECK_MACRO
      NO_CHECK_REQUIRED_COMPONENTS_MACRO
    )
    write_basic_package_version_file(
      "${CMAKE_CURRENT_BINARY_DIR}/${ARG_NAME}ConfigVersion.cmake"
      VERSION "${ARG_VERSION}"
      COMPATIBILITY AnyNewerVersion
    )
    install(FILES
      ${CMAKE_CURRENT_BINARY_DIR}/${ARG_NAME}Config.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/${ARG_NAME}ConfigVersion.cmake
      DESTINATION ${config_install_dir}
      COMPONENT ${export_component}
    )
  endif()

  if(ARG_COMPONENTS)
    set(_components "${ARG_COMPONENTS}")
  else()
    # Fall back to the cmake default component name. The caller is responsible
    # for setting CMAKE_INSTALL_DEFAULT_COMPONENT_NAME before any install()
    # rules so that unqualified installs land in the intended component.
    set(_components "${CMAKE_INSTALL_DEFAULT_COMPONENT_NAME}")
  endif()
  if(ARG_EXPORT_NAME)
    list(APPEND _components "${export_component}")
  endif()

  # Filter to only the desired components by enumerating them in
  # CPACK_INSTALL_CMAKE_PROJECTS. CPack.cmake only sets this variable when it
  # is unset, so our value is preserved through include(CPack). This approach
  # avoids CPACK_ARCHIVE_COMPONENT_INSTALL, which strips the root directory
  # from the archive when used with the TGZ generator.
  set(_install_projects "")
  foreach(_comp IN LISTS _components)
    list(APPEND _install_projects
        "${CMAKE_BINARY_DIR}" "${PROJECT_NAME}" "${_comp}" "/")
  endforeach()
  set(CPACK_INSTALL_CMAKE_PROJECTS "${_install_projects}")

  # KitMaker requires underscores in the component name (no hyphens).
  string(REPLACE "-" "_" _name_us "${ARG_NAME}")

  # KitMaker requires lowercase OS token.
  string(TOLOWER "${CMAKE_SYSTEM_NAME}" _os)

  # Arch token comes from CMAKE_SYSTEM_PROCESSOR (x86_64, aarch64, sbsa, etc.).
  set(_arch "${CMAKE_SYSTEM_PROCESSOR}")

  # KitMaker naming: <name>-<os>-<arch>-<version>
  set(CPACK_PACKAGE_FILE_NAME "${_name_us}-${_os}-${_arch}-${ARG_VERSION}")

  # The archive root directory is the component name alone (KitMaker multi-variant
  # convention). Inject it as the install prefix so files land at
  # <component_name>/lib/..., <component_name>/include/..., etc.
  # CPACK_INCLUDE_TOPLEVEL_DIRECTORY is disabled so CPack does not also prepend
  # the full package-file-name stem, which would double-nest the root directory.
  set(CPACK_PACKAGING_INSTALL_PREFIX "/${_name_us}")
  set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)

  set(CPACK_GENERATOR   "TGZ")
  set(CPACK_STRIP_FILES TRUE)

  set(CPACK_OUTPUT_CONFIG_FILE
      "${CMAKE_BINARY_DIR}/pkg/CPackConfig-${ARG_NAME}-TGZ.cmake")
  set(CPACK_SOURCE_OUTPUT_CONFIG_FILE
      "${CMAKE_BINARY_DIR}/pkg/CPackSourceConfig-${ARG_NAME}-TGZ.cmake")

  include(CPack)
endfunction()
