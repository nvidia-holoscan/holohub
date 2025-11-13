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
set_tests_properties(test_holohub_list PROPERTIES
    PASS_REGULAR_EXPRESSION "APPLICATIONS.*BENCHMARKS.*PACKAGES.*OPERATORS.*WORKFLOWS"
)

add_test(
    NAME test_holohub_autocompletion_list
    COMMAND ${CMAKE_SOURCE_DIR}/holohub autocompletion_list
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(test_holohub_autocompletion_list PROPERTIES
    PASS_REGULAR_EXPRESSION "ultrasound_segmentation.*build.*run.*list"
)


add_test(
    NAME test_holohub_lint
    COMMAND ${CMAKE_SOURCE_DIR}/holohub lint --dryrun
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(test_holohub_lint PROPERTIES
    PASS_REGULAR_EXPRESSION "ruff check --ignore.*isort -c.*black --check.*cpplint.*codespell.*cmakelint"
)
add_test(
    NAME test_holohub_lint_fix
    COMMAND ${CMAKE_SOURCE_DIR}/holohub lint --fix --dryrun
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(test_holohub_lint_fix PROPERTIES
    PASS_REGULAR_EXPRESSION "ruff check --fix.*isort.*black.*codespell.*cpplint"  # TODO: cmakelint missing
)

add_test(
    NAME test_holohub_setup
    COMMAND ${CMAKE_SOURCE_DIR}/holohub setup --dryrun
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(test_holohub_setup PROPERTIES
    PASS_REGULAR_EXPRESSION "(already installed|apt install)"
    FAIL_REGULAR_EXPRESSION "returned non-zero exit status;failed;Traceback;Error"
)

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    get_test_property(test_holohub_setup PASS_REGULAR_EXPRESSION pass_expression)
    set_tests_properties(test_holohub_setup PROPERTIES
        PASS_REGULAR_EXPRESSION "${pass_expression}.*(cudnn|nvinfer|nvonnxparsers)"
        FAIL_REGULAR_EXPRESSION "returned non-zero exit status;failed;Traceback;Error checking available versions"
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

add_test(
    NAME test_holohub_build_container_dryrun
    COMMAND ${CMAKE_SOURCE_DIR}/holohub build-container --dryrun --no-cache
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_build_container_dryrun PROPERTY
    PASS_REGULAR_EXPRESSION "docker build"
)

add_test(
    NAME test_holohub_build_container_build_args
    COMMAND ${CMAKE_SOURCE_DIR}/holohub build-container --dryrun --build-args "--build-arg TEST=value"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_build_container_build_args PROPERTY
    PASS_REGULAR_EXPRESSION "docker build"
    PASS_REGULAR_EXPRESSION "--build-arg TEST=value"
)

# default docker build args via environment variable
add_test(
    NAME test_holohub_default_docker_build_args_env
    COMMAND ${CMAKE_SOURCE_DIR}/holohub build-container --dryrun
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_default_docker_build_args_env PROPERTY
    ENVIRONMENT "HOLOHUB_DEFAULT_DOCKER_BUILD_ARGS=--build-arg DEFAULT_FLAG=abc"
)
set_property(TEST test_holohub_default_docker_build_args_env PROPERTY
    PASS_REGULAR_EXPRESSION "docker build"
    PASS_REGULAR_EXPRESSION "DEFAULT_FLAG=abc"
)

add_test(
    NAME test_holohub_build_container_cuda_version
    COMMAND ${CMAKE_SOURCE_DIR}/holohub build-container --dryrun --cuda 13
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_build_container_cuda_version PROPERTY
    PASS_REGULAR_EXPRESSION "docker build"
    PASS_REGULAR_EXPRESSION "--build-arg CUDA_MAJOR=13"
)

add_test(
    NAME test_holohub_run_container_docker_opts
    COMMAND ${CMAKE_SOURCE_DIR}/holohub run-container --dryrun --docker-opts "--memory 4g --entrypoint=bash"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_run_container_docker_opts PROPERTY
    PASS_REGULAR_EXPRESSION "docker run"
    PASS_REGULAR_EXPRESSION "--memory 4g"
)

add_test(
    NAME test_holohub_run_container_add_volume
    COMMAND ${CMAKE_SOURCE_DIR}/holohub run-container --dryrun --add-volume "/tmp:/tmp"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_run_container_add_volume PROPERTY
    PASS_REGULAR_EXPRESSION "docker run"
    PASS_REGULAR_EXPRESSION "-v /tmp:/tmp"
)

# default docker run args via environment variable
add_test(
    NAME test_holohub_default_docker_run_args_env
    COMMAND ${CMAKE_SOURCE_DIR}/holohub run-container --dryrun
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_default_docker_run_args_env PROPERTY
    ENVIRONMENT "HOLOHUB_DEFAULT_DOCKER_RUN_ARGS=-e TEST_ENV=123"
)
set_property(TEST test_holohub_default_docker_run_args_env PROPERTY
    PASS_REGULAR_EXPRESSION "docker run"
    PASS_REGULAR_EXPRESSION "TEST_ENV=123"
)

add_test(
    NAME test_holohub_build_dryrun
    COMMAND ${CMAKE_SOURCE_DIR}/holohub build endoscopy_tool_tracking --dryrun
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_build_dryrun PROPERTY
    PASS_REGULAR_EXPRESSION "build endoscopy_tool_tracking.*--local"
)

add_test(
    NAME test_holohub_run_dryrun
    COMMAND ${CMAKE_SOURCE_DIR}/holohub run endoscopy_tool_tracking --dryrun
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_run_dryrun PROPERTY
    PASS_REGULAR_EXPRESSION "run endoscopy_tool_tracking.*--local"
)

add_test(
    NAME test_holohub_run_forward_args
    COMMAND ${CMAKE_SOURCE_DIR}/holohub run endoscopy_tool_tracking --dryrun --run-args=--flag
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_run_forward_args PROPERTY
    PASS_REGULAR_EXPRESSION "run endoscopy_tool_tracking.*--run-args=--flag"
)

add_test(
    NAME test_holohub_run_pythonpath
    COMMAND ${CMAKE_SOURCE_DIR}/holohub run-container --dryrun
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_run_pythonpath PROPERTY
    PASS_REGULAR_EXPRESSION "Inspect docker image PYTHONPATH: docker inspect --format"
)

add_test(
    NAME test_holohub_run_container_dryrun
    COMMAND ${CMAKE_SOURCE_DIR}/holohub run-container --dryrun -- echo hello
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_run_container_dryrun PROPERTY
    PASS_REGULAR_EXPRESSION "docker run"
    PASS_REGULAR_EXPRESSION "holohub.*echo hello"
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

add_test(
    NAME test_cli_env_info
    COMMAND ${CMAKE_SOURCE_DIR}/holohub env-info
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_cli_env_info PROPERTY
    PASS_REGULAR_EXPRESSION "HoloHub Environment Variables"
    PASS_REGULAR_EXPRESSION "Holoscan Environment Variables"
    PASS_REGULAR_EXPRESSION "Other Relevant Environment Variables"
)

# test ambiguous dash-prefixed arguments
add_test(
    NAME test_cli_ambiguous_dash_prefixed_arguments
    COMMAND ${CMAKE_SOURCE_DIR}/holohub run-container --dryrun --run-args --local
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_cli_ambiguous_dash_prefixed_arguments PROPERTY
    PASS_REGULAR_EXPRESSION "ambiguous dash-prefixed arguments"
)

# test validate_mode conflict detection
add_test(
    NAME test_holohub_validate_mode_build_with_conflicts
    COMMAND ${CMAKE_SOURCE_DIR}/holohub run holochat mcp --run-args "conflicting_operator" --dryrun
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_validate_mode_build_with_conflicts PROPERTY
    PASS_REGULAR_EXPRESSION "Cannot specify CLI parameters.*--run-args.*when using explicit mode.*mcp"
    WILL_FAIL TRUE
)

# test validate_mode multiple conflicts detection
add_test(
    NAME test_holohub_validate_mode_build_with_and_run_args_conflicts
    COMMAND ${CMAKE_SOURCE_DIR}/holohub run holochat mcp --build-with "operator1" --run-args="--verbose" --dryrun
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_validate_mode_build_with_and_run_args_conflicts PROPERTY
    PASS_REGULAR_EXPRESSION "Cannot specify CLI parameters.*--run-args.*when using explicit mode.*mcp"
    WILL_FAIL TRUE
)

# test implicit default mode allows CLI parameter overrides
add_test(
    NAME test_holohub_implicit_default_mode_run_args_allowed
    COMMAND ${CMAKE_SOURCE_DIR}/holohub run holochat --run-args="--debug" --dryrun
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_implicit_default_mode_run_args_allowed PROPERTY
    PASS_REGULAR_EXPRESSION "Running holochat in.*mode.*--run-args=.*--debug"
)

# test implicit default mode with multiple CLI parameter overrides
add_test(
    NAME test_holohub_implicit_default_mode_multiple_overrides
    COMMAND ${CMAKE_SOURCE_DIR}/holohub run holochat --run-args="--verbose" --build-with="custom_ops" --dryrun
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_implicit_default_mode_multiple_overrides PROPERTY
    PASS_REGULAR_EXPRESSION "Running holochat in.*mode.*--build-with.*custom_ops.*--run-args=.*--verbose"
)

# test explicit mode still blocks CLI parameter overrides
add_test(
    NAME test_holohub_explicit_mode_blocks_overrides
    COMMAND ${CMAKE_SOURCE_DIR}/holohub run holochat cloud --run-args="--debug" --dryrun
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_explicit_mode_blocks_overrides PROPERTY
    PASS_REGULAR_EXPRESSION "Cannot specify CLI parameters.*--run-args.*when using explicit mode.*cloud"
    WILL_FAIL TRUE
)

# test build command with implicit default mode overrides
add_test(
    NAME test_holohub_build_implicit_default_overrides
    COMMAND ${CMAKE_SOURCE_DIR}/holohub build holochat --build-with="debug_ops" --dryrun
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_build_implicit_default_overrides PROPERTY
    PASS_REGULAR_EXPRESSION "Building holochat in.*mode.*--build-with.*debug_ops"
)

# implicit default mode: run-args append warning is printed
add_test(
    NAME test_holohub_implicit_default_warn_run_args_appended
    COMMAND ${CMAKE_SOURCE_DIR}/holohub run holochat --run-args="--debug" --dryrun
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_implicit_default_warn_run_args_appended PROPERTY
    PASS_REGULAR_EXPRESSION "CLI args --run-args=.*--debug.* will be appended to mode command"
)

# implicit default mode: docker-opts override warning when mode sets docker_run_args
add_test(
    NAME test_holohub_implicit_default_warn_docker_opts_override
    COMMAND ${CMAKE_SOURCE_DIR}/holohub run isaac_sim_holoscan_bridge --dryrun --docker-opts "--memory 2g"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_implicit_default_warn_docker_opts_override PROPERTY
    PASS_REGULAR_EXPRESSION "CLI args --docker-opts='--memory 2g' overrides mode --docker-opts"
)

# body_pose_estimation: implicit default mode run (v4l2)
add_test(
    NAME test_holohub_bpe_run_implicit_default
    COMMAND ${CMAKE_SOURCE_DIR}/holohub run body_pose_estimation --dryrun
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_bpe_run_implicit_default PROPERTY
    PASS_REGULAR_EXPRESSION "Running body_pose_estimation in 'v4l2' mode"
)

# body_pose_estimation: explicit mode run replayer
add_test(
    NAME test_holohub_bpe_run_replayer
    COMMAND ${CMAKE_SOURCE_DIR}/holohub run body_pose_estimation replayer --dryrun
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_bpe_run_replayer PROPERTY
    PASS_REGULAR_EXPRESSION "Running body_pose_estimation in 'replayer' mode"
)

# body_pose_estimation: implicit default allows run-args append warning
add_test(
    NAME test_holohub_bpe_run_implicit_warn_run_args
    COMMAND ${CMAKE_SOURCE_DIR}/holohub run body_pose_estimation --run-args="--source replayer" --dryrun
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_bpe_run_implicit_warn_run_args PROPERTY
    PASS_REGULAR_EXPRESSION "CLI args --run-args=.*--source replayer.* will be appended to mode command"
)

# body_pose_estimation: implicit default build (v4l2)
add_test(
    NAME test_holohub_bpe_build_implicit_default
    COMMAND ${CMAKE_SOURCE_DIR}/holohub build body_pose_estimation --dryrun
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_bpe_build_implicit_default PROPERTY
    PASS_REGULAR_EXPRESSION "Building body_pose_estimation in 'v4l2' mode"
)

# body_pose_estimation: explicit mode blocks conflicting CLI parameters (dds has build.depends)
add_test(
    NAME test_holohub_bpe_explicit_dds_blocks_build_with
    COMMAND ${CMAKE_SOURCE_DIR}/holohub run body_pose_estimation dds --build-with="extra_ops" --dryrun
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_bpe_explicit_dds_blocks_build_with PROPERTY
    PASS_REGULAR_EXPRESSION "Cannot specify CLI parameters.*--build-with.*when using explicit mode.*dds"
    WILL_FAIL TRUE
)

# test multi language project
add_test(
    NAME test_holohub_run_multi_language_project
    COMMAND ${CMAKE_SOURCE_DIR}/holohub run endoscopy_tool_tracking --dryrun
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_run_multi_language_project PROPERTY
    PASS_REGULAR_EXPRESSION "has multiple languages.*language"
)

add_test(
    NAME test_holohub_test_build_name_suffix
    COMMAND ${CMAKE_SOURCE_DIR}/holohub test --dryrun --build-name-suffix=test-branch
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_test_build_name_suffix PROPERTY
    PASS_REGULAR_EXPRESSION "DTAG=test-branch"
)

add_test(
    NAME test_holohub_test_cmake_options
    COMMAND ${CMAKE_SOURCE_DIR}/holohub test --dryrun --cmake-options="-DMAKE=1" --cmake-options="-DMAKE2=2" --ctest-options="-DTEST=2"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_test_cmake_options PROPERTY
    PASS_REGULAR_EXPRESSION "DCONFIGURE_OPTIONS.*DMAKE=1.*DMAKE2=2.*DTEST=2"
)

add_test(
    NAME test_holohub_test_ctest_options
    COMMAND ${CMAKE_SOURCE_DIR}/holohub test --dryrun --ctest-options="-DMAKE2=2" --ctest-options="-DTEST=2"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_test_ctest_options PROPERTY
    PASS_REGULAR_EXPRESSION "DMAKE2=2.*DTEST=2"
)

# Test that HOLOHUB_PATH_PREFIX is passed from host to container
add_test(
    NAME test_holohub_path_prefix_env
    COMMAND ${CMAKE_SOURCE_DIR}/holohub run-container --dryrun
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_path_prefix_env PROPERTY
    ENVIRONMENT "HOLOHUB_PATH_PREFIX=custom_prefix_"
)
set_property(TEST test_holohub_path_prefix_env PROPERTY
    PASS_REGULAR_EXPRESSION "docker run"
    PASS_REGULAR_EXPRESSION "-e HOLOHUB_PATH_PREFIX=custom_prefix_"
)

# Test that HOLOHUB_PATH_PREFIX is passed when running an application locally
add_test(
    NAME test_holohub_path_prefix_app_local
    COMMAND ${CMAKE_SOURCE_DIR}/holohub run endoscopy_tool_tracking --local --dryrun
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_path_prefix_app_local PROPERTY
    ENVIRONMENT "HOLOHUB_PATH_PREFIX=custom_prefix_"
)
set_property(TEST test_holohub_path_prefix_app_local PROPERTY
    PASS_REGULAR_EXPRESSION "custom_prefix_root"
)

add_test(  # Test that Python projects automatically get -DHOLOHUB_BUILD_PYTHON=ON flag
    NAME test_holohub_build_python_auto_flag
    COMMAND ${CMAKE_SOURCE_DIR}/holohub build volume_rendering --language python --local --dryrun
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_build_python_auto_flag PROPERTY
    PASS_REGULAR_EXPRESSION "-DHOLOHUB_BUILD_PYTHON=ON"
)

add_test(
    NAME test_holohub_run_container_extra_scripts_xvfb
    COMMAND ${CMAKE_SOURCE_DIR}/holohub run-container --extra-scripts xvfb --dryrun
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST test_holohub_run_container_extra_scripts_xvfb PROPERTY
    PASS_REGULAR_EXPRESSION "SCRIPT=utilities/setup/xvfb.sh"
)
