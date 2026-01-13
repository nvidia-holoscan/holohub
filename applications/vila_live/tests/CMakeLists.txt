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

# Add the VLM unit test
add_test(NAME vila_live_vlm_test
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test_vlm.py)

# Add the TinyChat unit test
add_test(NAME vila_live_tinychat_test
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test_tinychat.py)

# Set test properties
set_tests_properties(vila_live_vlm_test
  PROPERTIES
  ENVIRONMENT "HOLOHUB_DATA_DIR=${HOLOHUB_DATA_DIR};VILA_LIVE_TEST_CONFIG=${CMAKE_CURRENT_BINARY_DIR}/../vila_live_testing.yaml"
  PASS_REGULAR_EXPRESSION "Scheduler finished"
  FAIL_REGULAR_EXPRESSION "FAILED")

# Set test properties for TinyChat test
set_tests_properties(vila_live_tinychat_test
  PROPERTIES
  ENVIRONMENT "VILA_LIVE_TEST_CONFIG=${CMAKE_CURRENT_BINARY_DIR}/../vila_live_testing.yaml"
  PASS_REGULAR_EXPRESSION "Llama"
  FAIL_REGULAR_EXPRESSION "Traceback")
