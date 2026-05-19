# SPDX-FileCopyrightText: Copyright (c) {% now 'utc', '%Y' %} {{ cookiecutter.full_name }}{% if cookiecutter.affiliation %} / {{ cookiecutter.affiliation }}{% endif %}. All rights reserved.
# SPDX-License-Identifier: {{ cookiecutter._license }}

{% if cookiecutter.language == 'cpp' %}add_subdirectory(cpp)

{% endif %}find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_test(
    NAME pytest
    COMMAND ${Python3_EXECUTABLE} -m pytest ${PROJECT_SOURCE_DIR}/tests/python/ -v --tb=short
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

# PYTHONPATH must be *prepended* so an ambient holoscan SDK install on
# PYTHONPATH stays visible. Use ENVIRONMENT_MODIFICATION + path_list_prepend
# (CMake 3.22+) rather than splicing `:$ENV{PYTHONPATH}` into ENVIRONMENT:
#   - Splicing evaluates $ENV{PYTHONPATH} at *configure* time, so a value
#     exported between cmake and ctest invocations would be lost.
#   - When the env var is empty/unset the spliced form yields a trailing
#     colon (`PYTHONPATH=…ROOT:`), and Python treats an empty path entry as
#     `.` — silently adding the test CWD to sys.path and letting any local
#     file shadow installed packages.
# path_list_prepend handles unset/empty/set correctly and runs at test time.
# SKIP_RETURN_CODE 5 is the safety net for the env-broken case where the
# holoscan SDK still isn't importable (pytest collects zero items, exits 5).
set_tests_properties(pytest PROPERTIES
    ENVIRONMENT "{{ cookiecutter.module_slug | upper }}_BUILD_DIR=${CMAKE_BINARY_DIR}"
    ENVIRONMENT_MODIFICATION "PYTHONPATH=path_list_prepend:${{'{'}}{{ cookiecutter.module_slug | upper }}_PYTHON_ROOT}"
    SKIP_RETURN_CODE 5
    LABELS "python")
