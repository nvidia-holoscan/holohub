# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Find TensorRT headers and requested libraries.
#
# This follows Holoscan's FindTensorRT module and narrows discovery to requested
# components so TensorRT 10 installations are not required to provide removed
# parsers such as nvcaffe_parser and nvparsers.
#
# Imported targets:
#   TensorRT::<component>
#
# Result variables:
#   TensorRT_FOUND
#   TensorRT_VERSION
#   TensorRT_INCLUDE_DIR

include(FindPackageHandleStandardArgs)

find_path(TensorRT_INCLUDE_DIR
    NAMES NvInferVersion.h
    PATH_SUFFIXES
        aarch64-linux-gnu
        x86_64-linux-gnu
)
mark_as_advanced(TensorRT_INCLUDE_DIR)

if(TensorRT_INCLUDE_DIR)
    file(READ
        "${TensorRT_INCLUDE_DIR}/NvInferVersion.h"
        _tensorrt_version_header
    )
    foreach(_part MAJOR MINOR PATCH)
        string(REGEX MATCH
            "#define TRT_${_part}_ENTERPRISE +([0-9]+)"
            _match
            "${_tensorrt_version_header}"
        )
        if(NOT CMAKE_MATCH_1)
            string(REGEX MATCH
                "#define NV_TENSORRT_${_part} +([0-9]+)"
                _match
                "${_tensorrt_version_header}"
            )
        endif()
        set(_tensorrt_${_part} "${CMAKE_MATCH_1}")
    endforeach()
    set(TensorRT_VERSION
        "${_tensorrt_MAJOR}.${_tensorrt_MINOR}.${_tensorrt_PATCH}"
    )
endif()

if(NOT TensorRT_FIND_COMPONENTS)
    set(TensorRT_FIND_COMPONENTS nvinfer)
endif()

set(_tensorrt_required_vars TensorRT_INCLUDE_DIR)
foreach(_component IN LISTS TensorRT_FIND_COMPONENTS)
    find_library(TensorRT_${_component}_LIBRARY NAMES "${_component}")
    mark_as_advanced(TensorRT_${_component}_LIBRARY)
    if(TensorRT_${_component}_LIBRARY)
        set(TensorRT_${_component}_FOUND TRUE)
        if(NOT TARGET TensorRT::${_component})
            add_library(TensorRT::${_component} UNKNOWN IMPORTED)
            set_target_properties(TensorRT::${_component} PROPERTIES
                IMPORTED_LOCATION "${TensorRT_${_component}_LIBRARY}"
                INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
            )
        endif()
    else()
        set(TensorRT_${_component}_FOUND FALSE)
    endif()
    list(APPEND _tensorrt_required_vars TensorRT_${_component}_LIBRARY)
endforeach()

find_package_handle_standard_args(TensorRT
    REQUIRED_VARS ${_tensorrt_required_vars}
    VERSION_VAR TensorRT_VERSION
    HANDLE_COMPONENTS
)

unset(_match)
unset(_part)
unset(_component)
unset(_tensorrt_required_vars)
unset(_tensorrt_version_header)
unset(_tensorrt_MAJOR)
unset(_tensorrt_MINOR)
unset(_tensorrt_PATCH)
