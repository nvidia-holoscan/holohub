# SPDX-FileCopyrightText: Copyright (c) {% now 'utc', '%Y' %} {{ cookiecutter.full_name }}{% if cookiecutter.affiliation %} / {{ cookiecutter.affiliation }}{% endif %}. All rights reserved.
# SPDX-License-Identifier: {{ cookiecutter._license }}

# This file is entered via add_holohub_package({{ cookiecutter.module_repo_name }} ...) in
# pkg/CMakeLists.txt, which only fires when PKG_{{ cookiecutter.module_repo_name.replace('-', '_') }} is ON.
include(holohub_configure_deb)

holohub_configure_deb(
    NAME        "{{ cookiecutter.module_repo_name }}"
    DESCRIPTION "${PROJECT_DESCRIPTION}"
    VERSION     "${PROJECT_VERSION}"
    VENDOR      "{% if cookiecutter.affiliation %}{{ cookiecutter.affiliation }}{% else %}NVIDIA{% endif %}"
    CONTACT     "{{ cookiecutter.contact_email }}"
    DEPENDS     "holoscan (>= {{ cookiecutter.holoscan_version }})"
)
