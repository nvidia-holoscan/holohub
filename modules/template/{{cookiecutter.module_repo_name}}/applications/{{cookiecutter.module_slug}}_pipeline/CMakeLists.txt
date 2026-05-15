# SPDX-FileCopyrightText: Copyright (c) {% now 'utc', '%Y' %} {{ cookiecutter.full_name }}{% if cookiecutter.affiliation %} / {{ cookiecutter.affiliation }}{% endif %}. All rights reserved.
# SPDX-License-Identifier: {{ cookiecutter._license }}

{% if cookiecutter.language == 'cpp' %}add_executable({{ cookiecutter.module_slug }}_pipeline {{ cookiecutter.module_slug }}_pipeline.cpp)

target_link_libraries({{ cookiecutter.module_slug }}_pipeline PRIVATE
    {{ cookiecutter.operator_slug }}
    holoscan::core)

target_include_directories({{ cookiecutter.module_slug }}_pipeline PRIVATE
    ${PROJECT_SOURCE_DIR}/operators)
{% else %}# Python-only application — no compilation needed.
# Install the script so 'cmake --install' makes it available on PATH.
install(PROGRAMS {{ cookiecutter.module_slug }}_pipeline.py
    DESTINATION bin
    RENAME {{ cookiecutter.module_slug }}_pipeline)
{% endif %}
