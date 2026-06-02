# SPDX-FileCopyrightText: Copyright (c) {% now 'utc', '%Y' %} {{ cookiecutter.full_name }}{% if cookiecutter.affiliation %} / {{ cookiecutter.affiliation }}{% endif %}. All rights reserved.
# SPDX-License-Identifier: {{ cookiecutter._license }}

# This file is entered via add_holohub_package({{ cookiecutter.module_repo_name }} ...) in
# pkg/CMakeLists.txt, which only fires when PKG_{{ cookiecutter.module_repo_name.replace('-', '_') }} is ON.
include(holohub_configure_deb)

# metadata.json alongside this file registers the package with the holohub CLI:
#   - `./holohub list` surfaces it as a project_type=package entry (discovered by
#     the recursive HOLOHUB_SEARCH_PATH scan from the module root).
#   - package.dockerfile points the CLI to the container image used for
#     container-based package builds (`./holohub run --build`).
holohub_configure_deb(
    NAME        "{{ cookiecutter.module_repo_name }}"
    DESCRIPTION "${PROJECT_DESCRIPTION}"
    VERSION     "${PROJECT_VERSION}"
    VENDOR      "{% if cookiecutter.affiliation %}{{ cookiecutter.affiliation }}{% else %}NVIDIA{% endif %}"
    CONTACT     "{{ cookiecutter.contact_email }}"
    DEPENDS     "holoscan (>= {{ cookiecutter.holoscan_version }})"
)
