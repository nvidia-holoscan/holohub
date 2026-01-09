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

add_test(NAME holochat_llm_test
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test_llm.py)
add_test(NAME holochat_chatbot_test
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test_chatbot.py)
add_test(NAME holochat_build_db_test
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test_build_holoscan_db.py)

set_tests_properties(holochat_llm_test
  PROPERTIES
  PASS_REGULAR_EXPRESSION "llm - INFO - Retrieving Holoscan embeddings vector DB.*Ran 4 tests.*OK"
  FAIL_REGULAR_EXPRESSION "Traceback|FAILED|Exception|ERROR")
set_tests_properties(holochat_chatbot_test
  PROPERTIES
  PASS_REGULAR_EXPRESSION ".*Ran 3 tests.*OK"
  FAIL_REGULAR_EXPRESSION "Traceback|FAILED|Exception|ERROR")
set_tests_properties(holochat_build_db_test
  PROPERTIES
  PASS_REGULAR_EXPRESSION "Ran 2 tests.*OK.*Building Holoscan Embeddings Chroma DB"
  FAIL_REGULAR_EXPRESSION "Traceback|FAILED|Exception|ERROR")
