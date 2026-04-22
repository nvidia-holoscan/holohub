# SPDX-FileCopyrightText: Copyright (c) 2026 Wayland Technologies. All rights reserved.
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

if(NOT DEFINED DATASET_DIR OR DATASET_DIR STREQUAL "")
  message(FATAL_ERROR "DATASET_DIR must be set")
endif()

if(NOT DEFINED STAMP_FILE OR STAMP_FILE STREQUAL "")
  message(FATAL_ERROR "STAMP_FILE must be set")
endif()

set(EXPECTED_FILE "${DATASET_DIR}/ir_base.gxf_entities")
set(NESTED_DATASET_DIR "${DATASET_DIR}/atracsys_visualizer")

if(EXISTS "${EXPECTED_FILE}")
  file(TOUCH "${STAMP_FILE}")
  return()
endif()

if(EXISTS "${NESTED_DATASET_DIR}/ir_base.gxf_entities")
  file(GLOB NESTED_DATASET_FILES "${NESTED_DATASET_DIR}/*")

  foreach(DATA_FILE IN LISTS NESTED_DATASET_FILES)
    get_filename_component(DATA_FILE_NAME "${DATA_FILE}" NAME)
    file(RENAME "${DATA_FILE}" "${DATASET_DIR}/${DATA_FILE_NAME}")
  endforeach()

  file(REMOVE_RECURSE "${NESTED_DATASET_DIR}")
endif()

if(NOT EXISTS "${EXPECTED_FILE}")
  message(FATAL_ERROR
    "Atracsys Visualizer dataset download completed, but the expected replay files were not "
    "found under ${DATASET_DIR}.")
endif()

file(TOUCH "${STAMP_FILE}")
