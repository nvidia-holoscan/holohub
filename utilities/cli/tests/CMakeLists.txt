# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

include(CTest)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Add functional tests for holohub script
add_test(
    NAME test_holohub_list
    COMMAND ${CMAKE_SOURCE_DIR}/holohub list
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_list PROPERTY
    PASS_REGULAR_EXPRESSION "APPLICATIONS"
    PASS_REGULAR_EXPRESSION "BENCHMARKS"
    PASS_REGULAR_EXPRESSION "PACKAGES"
    PASS_REGULAR_EXPRESSION "OPERATORS"
    PASS_REGULAR_EXPRESSION "WORKFLOWS"
)

add_test(
    NAME test_holohub_lint
    COMMAND ${CMAKE_SOURCE_DIR}/holohub lint --dryrun
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
add_test(
    NAME test_holohub_lint_fix
    COMMAND ${CMAKE_SOURCE_DIR}/holohub lint --fix --dryrun
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
foreach(lint_test "test_holohub_lint;test_holohub_lint_fix")
    set_property(TEST test_holohub_lint PROPERTY
        PASS_REGULAR_EXPRESSION "ruff"
        PASS_REGULAR_EXPRESSION "isort"
        PASS_REGULAR_EXPRESSION "cpplint"
        PASS_REGULAR_EXPRESSION "codespell"
        PASS_REGULAR_EXPRESSION "cmakelint"
    )
endforeach()

add_test(
    NAME test_holohub_setup
    COMMAND ${CMAKE_SOURCE_DIR}/holohub setup --dryrun
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_setup PROPERTY
    PASS_REGULAR_EXPRESSION "libcudnn"
    PASS_REGULAR_EXPRESSION "git"
)

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    set_property(TEST test_holohub_setup APPEND PROPERTY
        FAIL_REGULAR_EXPRESSION "Error checking available versions"
    )
endif()

add_test(
    NAME test_holohub_clear_cache
    COMMAND ${CMAKE_SOURCE_DIR}/holohub clear-cache --dryrun
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_clear_cache PROPERTY
    PASS_REGULAR_EXPRESSION "Would remove:.*${CMAKE_SOURCE_DIR}/build"
)

# Add test for container module
add_test(
    NAME test_container
    COMMAND ${Python3_EXECUTABLE} -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR} -p "test_container.py"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# Add test for CLI module
add_test(
    NAME test_cli
    COMMAND ${Python3_EXECUTABLE} -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR} -p "test_cli.py"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
