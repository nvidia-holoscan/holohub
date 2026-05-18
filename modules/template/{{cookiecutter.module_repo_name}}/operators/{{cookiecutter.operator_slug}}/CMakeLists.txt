# SPDX-FileCopyrightText: Copyright (c) {% now 'utc', '%Y' %} {{ cookiecutter.full_name }}{% if cookiecutter.affiliation %} / {{ cookiecutter.affiliation }}{% endif %}. All rights reserved.
# SPDX-License-Identifier: {{ cookiecutter._license }}

{% if cookiecutter.language == 'cpp' %}add_library({{ cookiecutter.operator_slug }} {{ cookiecutter.operator_slug }}.cpp)
add_library(holoscan::{{ cookiecutter.operator_slug }} ALIAS {{ cookiecutter.operator_slug }})

target_include_directories({{ cookiecutter.operator_slug }} PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)

target_link_libraries({{ cookiecutter.operator_slug }} PRIVATE holoscan::core)

install(FILES {{ cookiecutter.operator_slug }}.hpp DESTINATION include/{{ cookiecutter.operator_slug }})

add_subdirectory(python)
{% else %}# Pure Python operator — copy source into the Python package tree.
configure_file(
    {{ cookiecutter.operator_slug }}.py
    ${{'{'}}{{ cookiecutter.module_slug | upper }}_PYTHON_PKG_DIR}/{{ cookiecutter.operator_slug }}.py COPYONLY)
{% endif %}
