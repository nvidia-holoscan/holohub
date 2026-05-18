# SPDX-FileCopyrightText: Copyright (c) {% now 'utc', '%Y' %} {{ cookiecutter.full_name }}{% if cookiecutter.affiliation %} / {{ cookiecutter.affiliation }}{% endif %}. All rights reserved.
# SPDX-License-Identifier: {{ cookiecutter._license }}

# Python3 + pybind11 are pulled in (find_package + FetchContent) by
# pybind11_add_holohub_module, which the operator's own python/CMakeLists.txt
# includes when {{ cookiecutter.language }} == 'cpp'. No find_package needed
# here at the aggregate operators/ level.

# Each operator is gated on its own OP_<name> option, defaulting to ${BUILD_ALL}.
# Add one add_holohub_operator() call per operator.
add_holohub_operator({{ cookiecutter.operator_slug }})

# Install the aggregate __init__.py alongside each operator's built module.
configure_file(
    ${PROJECT_SOURCE_DIR}/python/holoscan/{{ cookiecutter.module_slug }}/__init__.py
    ${{'{'}}{{ cookiecutter.module_slug | upper }}_PYTHON_PKG_DIR}/__init__.py COPYONLY)

# Relative DESTINATION — interpreted as <CMAKE_INSTALL_PREFIX>/holoscan/<mod>
# in plain `cmake --install` (set --prefix to ${Python3_SITEARCH}), and as
# <wheel-root>/holoscan/<mod> when scikit-build-core builds the wheel.
install(DIRECTORY ${{'{'}}{{ cookiecutter.module_slug | upper }}_PYTHON_PKG_DIR}
    DESTINATION holoscan)
