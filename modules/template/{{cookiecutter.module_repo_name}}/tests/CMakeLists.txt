# SPDX-FileCopyrightText: Copyright (c) {% now 'utc', '%Y' %} {{ cookiecutter.full_name }}{% if cookiecutter.affiliation %} / {{ cookiecutter.affiliation }}{% endif %}. All rights reserved.
# SPDX-License-Identifier: {{ cookiecutter._license }}

{% if cookiecutter.language == 'cpp' %}add_subdirectory(cpp)

{% endif %}find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_test(
    NAME pytest
    COMMAND ${Python3_EXECUTABLE} -m pytest ${PROJECT_SOURCE_DIR}/tests/python/ -v --tb=short
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

set_tests_properties(pytest PROPERTIES
    ENVIRONMENT "PYTHONPATH=${{'{'}}{{ cookiecutter.module_slug | upper }}_PYTHON_ROOT};{{ cookiecutter.module_slug | upper }}_BUILD_DIR=${CMAKE_BINARY_DIR}"
    SKIP_RETURN_CODE 5
    LABELS "python")
