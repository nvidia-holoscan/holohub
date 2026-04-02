# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_path(S3DK_INCLUDE_DIR
  NAMES s3dk_gpu.hpp
  HINTS
    ${S3DK_ROOT}
    $ENV{S3DK_ROOT}
    /opt/s3dk
  PATH_SUFFIXES include
)

find_library(S3DK_LIBRARY
  NAMES s3dk_gpu libs3dk_gpu
  HINTS
    ${S3DK_ROOT}
    $ENV{S3DK_ROOT}
    /opt/s3dk
  PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(S3DK REQUIRED_VARS S3DK_INCLUDE_DIR S3DK_LIBRARY)

if(S3DK_FOUND AND NOT TARGET S3DK::S3DK)
  add_library(S3DK::S3DK UNKNOWN IMPORTED)
  set_target_properties(S3DK::S3DK PROPERTIES
    IMPORTED_LOCATION "${S3DK_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${S3DK_INCLUDE_DIR}"
  )
endif()
