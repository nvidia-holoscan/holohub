# SPDX-FileCopyrightText: Copyright (c) {% now 'utc', '%Y' %} {{ cookiecutter.full_name }}{% if cookiecutter.affiliation %} / {{ cookiecutter.affiliation }}{% endif %}. All rights reserved.
# SPDX-License-Identifier: {{ cookiecutter._license }}

add_executable({{ cookiecutter.module_slug }}_pipeline {{ cookiecutter.module_slug }}_pipeline.cpp)

target_link_libraries({{ cookiecutter.module_slug }}_pipeline PRIVATE
    holoscan::{{ cookiecutter.operator_slug }}
    holoscan::core)

target_include_directories({{ cookiecutter.module_slug }}_pipeline PRIVATE
    ${PROJECT_SOURCE_DIR}/operators)
