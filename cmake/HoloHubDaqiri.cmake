# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

function(holohub_link_daqiri target_name)
  if(CMAKE_VERSION VERSION_LESS "3.24")
    message(FATAL_ERROR "holohub_link_daqiri requires CMake 3.24 or newer")
  endif()

  if(NOT TARGET "${target_name}")
    message(FATAL_ERROR "holohub_link_daqiri target does not exist: ${target_name}")
  endif()

  set(_daqiri_targets daqiri::daqiri)
  foreach(_daqiri_engine IN ITEMS dpdk socket rdma ibverbs)
    if(TARGET "daqiri::${_daqiri_engine}")
      list(APPEND _daqiri_targets "daqiri::${_daqiri_engine}")
    endif()
  endforeach()

  # DAQIRI static packages can contain cross-references between core and engine archives.
  string(JOIN "," _daqiri_link_group ${_daqiri_targets})
  target_link_libraries("${target_name}" PRIVATE "$<LINK_GROUP:RESCAN,${_daqiri_link_group}>")
endfunction()
