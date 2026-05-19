# SPDX-FileCopyrightText: Copyright (c) {% now 'utc', '%Y' %} {{ cookiecutter.full_name }}{% if cookiecutter.affiliation %} / {{ cookiecutter.affiliation }}{% endif %}. All rights reserved.
# SPDX-License-Identifier: {{ cookiecutter._license }}

{% if cookiecutter.language == 'cpp' %}add_subdirectory(cpp)

{% endif %}find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_test(
    NAME pytest
    COMMAND ${Python3_EXECUTABLE} -m pytest ${PROJECT_SOURCE_DIR}/tests/python/ -v --tb=short
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

# PYTHONPATH is *prepended* (note the `:$ENV{PYTHONPATH}`) so an ambient
# holoscan SDK install on the runner's PYTHONPATH stays visible. A bare
# assignment would silently drop the SDK, the module-level
# `pytest.importorskip("holoscan")` would fire, pytest would collect zero
# items and exit 5, and CTest would mark the run "Skipped" with no obvious
# cause. SKIP_RETURN_CODE 5 below is the safety net for the env-broken case.
set_tests_properties(pytest PROPERTIES
    ENVIRONMENT "PYTHONPATH=${{'{'}}{{ cookiecutter.module_slug | upper }}_PYTHON_ROOT}:$ENV{PYTHONPATH};{{ cookiecutter.module_slug | upper }}_BUILD_DIR=${CMAKE_BINARY_DIR}"
    SKIP_RETURN_CODE 5
    LABELS "python")
