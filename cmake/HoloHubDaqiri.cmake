# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

function(holohub_link_daqiri target_name)
  if(NOT TARGET "${target_name}")
    message(FATAL_ERROR "holohub_link_daqiri target does not exist: ${target_name}")
  endif()

  set(_daqiri_targets daqiri::daqiri)
  foreach(_daqiri_manager IN ITEMS dpdk socket rdma)
    if(TARGET "daqiri::${_daqiri_manager}")
      list(APPEND _daqiri_targets "daqiri::${_daqiri_manager}")
    endif()
  endforeach()

  # DAQIRI static packages can contain cross-references between core and manager archives.
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    string(JOIN "," _daqiri_link_group ${_daqiri_targets})
    target_link_libraries("${target_name}" PRIVATE "$<LINK_GROUP:RESCAN,${_daqiri_link_group}>")
  else()
    target_link_libraries("${target_name}" PRIVATE
      "-Wl,--start-group"
      ${_daqiri_targets}
      "-Wl,--end-group"
    )
  endif()
endfunction()
