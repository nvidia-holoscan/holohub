# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.20)
project(gsplat_scene_recon LANGUAGES NONE)

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------

find_package(holoscan 3.7 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
# MedSAM3 checkpoint: obtain from https://huggingface.co/ChongCong/Medical-SAM3
# (or custom fine-tuned .pt) and place at
# <HOLOHUB_DATA_DIR>/gsplat_scene_recon/medsam3/checkpoint_8_new_best.pt
# Optional: set MEDSAM3_GDOWN_FILE_ID to a Google Drive file ID to download via
# holoscan_download_data (e.g. from .../file/d/FILE_ID/view). Leave empty to skip.
set(MEDSAM3_GDOWN_FILE_ID "" CACHE STRING "Google Drive file ID for MedSAM3 checkpoint. Empty = skip download.")
if(HOLOHUB_DOWNLOAD_DATASETS AND MEDSAM3_GDOWN_FILE_ID)
  include(holoscan_download_data)
  holoscan_download_data(medsam3
    URL "https://drive.google.com/file/d/${MEDSAM3_GDOWN_FILE_ID}/view"
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}/gsplat_scene_recon
    DOWNLOAD_NAME checkpoint_8_new_best.pt
    ALL
  )
endif()

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------

if(BUILD_TESTING)
  # Test that all application modules can be imported (no GPU/data required).
  add_test(NAME gsplat_scene_recon_test_all_imports
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_all_imports.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  set_tests_properties(gsplat_scene_recon_test_all_imports PROPERTIES
    TIMEOUT 60
    PASS_REGULAR_EXPRESSION "SUCCESS|imports OK|SKIP"
  )

  # Test EndoNeRF-style data loading (minimal fixture, no GPU).
  add_test(NAME gsplat_scene_recon_test_data_loading
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_data_loading.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  set_tests_properties(gsplat_scene_recon_test_data_loading PROPERTIES
    TIMEOUT 60
    PASS_REGULAR_EXPRESSION "SUCCESS|data loading OK|SKIP"
  )

  # Test gsplat rendering path (GPU required; may SKIP if no GPU).
  add_test(NAME gsplat_scene_recon_test_dynamic_rendering
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dynamic_rendering.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  set_tests_properties(gsplat_scene_recon_test_dynamic_rendering PROPERTIES
    TIMEOUT 120
    PASS_REGULAR_EXPRESSION "SUCCESS|dynamic rendering OK|SKIP"
  )

  # Test minimal Holoviz app headless (Holoscan + HolovizOp).
  add_test(NAME gsplat_scene_recon_test_holoviz_minimal
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_holoviz_minimal.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  set_tests_properties(gsplat_scene_recon_test_holoviz_minimal PROPERTIES
    TIMEOUT 60
    PASS_REGULAR_EXPRESSION "SUCCESS|Holoviz minimal OK|SKIP"
  )
endif()
