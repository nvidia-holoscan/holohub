# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

if(NOT EXISTS "${PVA_UNSHARP_MASK_LIB_DEST}")
  # Download libpva_unsharp_mask.a using curl
  message(STATUS "libpva_unsharp_mask.a not found in source directory. Downloading from ${PVA_UNSHARP_MASK_URL} using curl")
  execute_process(COMMAND curl -L ${PVA_UNSHARP_MASK_URL} -o ${PVA_UNSHARP_MASK_LIB_DEST}
                  RESULT_VARIABLE result
                  OUTPUT_QUIET)
  if(NOT result EQUAL "0")
    message(FATAL_ERROR "Error downloading libpva_unsharp_mask.a using curl")
  endif()
  # Check if the downloaded file contains a "File not found" error message
  file(READ ${PVA_UNSHARP_MASK_LIB_DEST} contents)
  if(contents MATCHES "\"status\" : 404")
    message(FATAL_ERROR "Downloaded file contains a 'File not found' error. Please check the URL and try again.")
  endif()
  # Download cupva_allowlist_pva_unsharp_mask using curl
  message(STATUS "Downloading cupva_allowlist_pva_unsharp_mask from ${CUPVA_ALLOWLIST_URL} using curl")
  execute_process(COMMAND curl -L ${CUPVA_ALLOWLIST_URL} -o ${CUPVA_ALLOWLIST_DEST}
                  RESULT_VARIABLE result_allowlist
                  OUTPUT_QUIET)
  if(NOT result_allowlist EQUAL "0")
    message(FATAL_ERROR "Error downloading cupva_allowlist_pva_unsharp_mask using curl")
  endif()
  # Check if the downloaded file contains a "File not found" error message
  file(READ ${CUPVA_ALLOWLIST_DEST} contents_allowlist)
  if(contents_allowlist MATCHES "\"status\" : 404")
    message(FATAL_ERROR "Downloaded cupva_allowlist_pva_unsharp_mask contains a 'File not found' error. Please check the URL and try again.")
  endif()
endif()
