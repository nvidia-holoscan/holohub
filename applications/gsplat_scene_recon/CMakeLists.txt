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
# Data (checkpoints in ${HOLOHUB_DATA_DIR}/gsplat_scene_recon/da2 and .../medsam3)
# -----------------------------------------------------------------------------
# When HOLOHUB_DOWNLOAD_DATASETS is ON, automatically download Depth Anything V2
# (Apache-2.0) and MedSAM3 checkpoints to the application data dir.
if(HOLOHUB_DOWNLOAD_DATASETS)
  set(GSPLAT_DATA_DIR "${HOLOHUB_DATA_DIR}/gsplat_scene_recon")
  set(DA2_DIR "${GSPLAT_DATA_DIR}/da2")
  set(MEDSAM3_DIR "${GSPLAT_DATA_DIR}/medsam3")
  set(DA2_URL "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth")
  set(DA2_FILE "${DA2_DIR}/depth_anything_v2_vits.pth")
  set(MEDSAM3_URL "https://huggingface.co/ChongCong/Medical-SAM3/resolve/main/checkpoint_8_new_best.pt")
  set(MEDSAM3_FILE "${MEDSAM3_DIR}/checkpoint_8_new_best.pt")

  # Depth Anything V2 (ViT-S) - Apache-2.0, single file
  set(DA2_STAMP "${CMAKE_CURRENT_BINARY_DIR}/da2_checkpoint.stamp")
  add_custom_command(OUTPUT "${DA2_STAMP}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${DA2_DIR}"
    COMMAND bash -c "curl -L -f -o '${DA2_FILE}' '${DA2_URL}' && touch '${DA2_STAMP}'"
    COMMENT "Download DA2 checkpoint to ${DA2_FILE}"
    VERBATIM
  )
  add_custom_target(da2_checkpoint_data ALL DEPENDS "${DA2_STAMP}")

  # MedSAM3 checkpoint - single file (Hugging Face; may require login for gated repos)
  set(MEDSAM3_STAMP "${CMAKE_CURRENT_BINARY_DIR}/medsam3_checkpoint.stamp")
  add_custom_command(OUTPUT "${MEDSAM3_STAMP}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${MEDSAM3_DIR}"
    COMMAND bash -c "curl -L -f -o '${MEDSAM3_FILE}' '${MEDSAM3_URL}' && touch '${MEDSAM3_STAMP}'"
    COMMENT "Download MedSAM3 checkpoint to ${MEDSAM3_FILE}"
    VERBATIM
  )
  add_custom_target(medsam3_checkpoint_data ALL DEPENDS "${MEDSAM3_STAMP}")
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
