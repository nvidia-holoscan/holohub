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

# CLI-side CTest entries. Two layers:
#
#   1. Direct ``add_test(COMMAND ${CMAKE_SOURCE_DIR}/holohub <subcmd>)`` checks
#      that exercise the user-facing path end-to-end (shell wrapper ->
#      holoscan_cli.__main__ -> argparse -> dispatch -> stdout) and assert on
#      the output via ``PASS_REGULAR_EXPRESSION``. These need no pytest, no
#      Python imports — just a working ``./holohub`` wrapper and the
#      installed consolidated ``holoscan-cli`` package. They cover the
#      assertions that depend on a real HoloHub project tree (real
#      ``metadata.json``, real ``applications/``/``operators/`` layout)
#      and therefore cannot live in ``holoscan-cli/tests/unit/``.
#
#   2. Pytest-based smoke tests (``holoscan_cli_consolidation``,
#      ``holohub_create_module``) that ride on the same ctest invocation but
#      need ``pytest`` importable.

include(CTest)

# ----------------------------------------------------------------------------
# Layer 1 — black-box CLI assertions against the real HoloHub tree.
# ----------------------------------------------------------------------------

# Project listing: ``./holohub list`` must enumerate the standard category
# headers the wrapper-driven UI shows. Catches regressions where a category is
# silently dropped from project discovery.
add_test(
  NAME test_holohub_list
  COMMAND ${CMAKE_SOURCE_DIR}/holohub list
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(test_holohub_list PROPERTIES
  PASS_REGULAR_EXPRESSION "APPLICATIONS.*BENCHMARKS.*PACKAGES.*OPERATORS.*WORKFLOWS"
)

# Autocompletion list: shell completion depends on the wrapper printing one
# project name + command per line. Pre-consolidation
# ``test_holohub_autocompletion_list``.
add_test(
  NAME test_holohub_autocompletion_list
  COMMAND ${CMAKE_SOURCE_DIR}/holohub autocompletion_list
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(test_holohub_autocompletion_list PROPERTIES
  PASS_REGULAR_EXPRESSION "ultrasound_segmentation.*build.*run.*list"
)

# ``HOLOSCAN_CLI_PATH_PREFIX`` env var propagation through ``run-container``.
# (The consolidated CLI renamed ``HOLOHUB_PATH_PREFIX`` to
# ``HOLOSCAN_CLI_PATH_PREFIX``; CTest's regex engine is line-oriented and
# can't bridge ``docker run`` and the ``-e HOLOSCAN_CLI_PATH_PREFIX=…``
# arg across the dryrun pretty-printer's newlines, so the assertion targets
# the env-passing line specifically.) Pre-consolidation
# ``test_holohub_path_prefix_env``.
add_test(
  NAME test_holohub_path_prefix_env
  COMMAND ${CMAKE_SOURCE_DIR}/holohub run-container --dryrun
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(test_holohub_path_prefix_env PROPERTIES
  ENVIRONMENT "HOLOSCAN_CLI_PATH_PREFIX=custom_prefix_"
  PASS_REGULAR_EXPRESSION "HOLOSCAN_CLI_PATH_PREFIX=custom_prefix_"
)

# Multi-language project detection: ``endoscopy_tool_tracking`` ships both
# C++ and Python variants — the CLI must guide the user to disambiguate.
# Pre-consolidation ``test_holohub_run_multi_language_project``.
add_test(
  NAME test_holohub_run_multi_language_project
  COMMAND ${CMAKE_SOURCE_DIR}/holohub run endoscopy_tool_tracking --dryrun
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(test_holohub_run_multi_language_project PROPERTIES
  PASS_REGULAR_EXPRESSION "has multiple languages.*language"
)

# ``holochat`` explicit mode + CLI override → must surface the "will be
# appended to mode command" warning. Pre-consolidation
# ``test_holohub_explicit_mode_cli_override_run_args``.
add_test(
  NAME test_holohub_explicit_mode_cli_override_run_args
  COMMAND ${CMAKE_SOURCE_DIR}/holohub run holochat mcp
          --run-args=conflicting_operator --dryrun
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(test_holohub_explicit_mode_cli_override_run_args PROPERTIES
  PASS_REGULAR_EXPRESSION "CLI.*--run-args.*will be appended to mode command"
)

# ``body_pose_estimation`` implicit-default mode resolution: with no mode
# arg, the CLI must pick the metadata's default (``v4l2``).
# Pre-consolidation ``test_holohub_bpe_run_implicit_default``.
add_test(
  NAME test_holohub_bpe_run_implicit_default
  COMMAND ${CMAKE_SOURCE_DIR}/holohub run body_pose_estimation --dryrun
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(test_holohub_bpe_run_implicit_default PROPERTIES
  PASS_REGULAR_EXPRESSION "Running body_pose_estimation in 'v4l2' mode"
)

# ``body_pose_estimation`` explicit mode: ``replayer`` must override the
# metadata default. Pre-consolidation ``test_holohub_bpe_run_replayer``.
add_test(
  NAME test_holohub_bpe_run_replayer
  COMMAND ${CMAKE_SOURCE_DIR}/holohub run body_pose_estimation replayer --dryrun
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(test_holohub_bpe_run_replayer PROPERTIES
  PASS_REGULAR_EXPRESSION "Running body_pose_estimation in 'replayer' mode"
)

# ``./holohub env-info`` must print the three standard env sections users
# rely on for bug reports. Pre-consolidation ``test_cli_env_info``.
add_test(
  NAME test_cli_env_info
  COMMAND ${CMAKE_SOURCE_DIR}/holohub env-info
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(test_cli_env_info PROPERTIES
  PASS_REGULAR_EXPRESSION "Holoscan CLI Environment Variables"
)

# ----------------------------------------------------------------------------
# Layer 2 — pytest-based smoke tests.
# ----------------------------------------------------------------------------
#
# ``--noconftest`` skips the repo-root ``conftest.py`` (which imports cupy and
# holoscan); these smoke tests do not use any of those fixtures.

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pytest"
    RESULT_VARIABLE _pytest_probe
    OUTPUT_QUIET ERROR_QUIET
  )
  if(_pytest_probe EQUAL 0)
    add_test(
      NAME holoscan_cli_consolidation
      COMMAND ${Python3_EXECUTABLE} -m pytest -q --noconftest
              ${CMAKE_CURRENT_SOURCE_DIR}/test_holoscan_cli_consolidation.py
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    add_test(
      NAME holohub_create_module
      COMMAND ${Python3_EXECUTABLE} -m pytest -q --noconftest
              ${CMAKE_CURRENT_SOURCE_DIR}/test_create_module.py
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
  else()
    message(STATUS
      "BUILD_HOLOHUB_TESTING is ON but Python pytest is not importable; "
      "skipping pytest-based smoke tests. Install pytest into the Python "
      "interpreter at ${Python3_EXECUTABLE} to enable them.")
  endif()
endif()
