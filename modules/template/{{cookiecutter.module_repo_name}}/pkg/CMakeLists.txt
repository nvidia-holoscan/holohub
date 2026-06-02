# SPDX-FileCopyrightText: Copyright (c) {% now 'utc', '%Y' %} {{ cookiecutter.full_name }}{% if cookiecutter.affiliation %} / {{ cookiecutter.affiliation }}{% endif %}. All rights reserved.
# SPDX-License-Identifier: {{ cookiecutter._license }}

# add_holohub_package() handles the full cascade for both standalone
# packaging (`./holohub package {{ cookiecutter.module_repo_name }} --pkg-generator DEB`, which
# sets -DPKG_{{ cookiecutter.module_repo_name.replace('-', '_') }}=ON via the HoloHub CLI) and in-tree
# HoloHub-monorepo packaging once this module is fetched as an external
# dependency:
#
#   1. Defines option(PKG_{{ cookiecutter.module_repo_name.replace('-', '_') }} ... ${BUILD_ALL}) — ON
#      when BUILD_ALL=ON (default standalone) or when the CLI passes the
#      option explicitly.
#   2. add_subdirectory({{ cookiecutter.module_repo_name }}) — enters
#      pkg/{{ cookiecutter.module_repo_name }}/ where holohub_configure_deb() emits
#      CPackConfig-{{ cookiecutter.module_repo_name }}.cmake.
#   3. set("OP_<x>" ON CACHE BOOL ... FORCE) and the equivalent for
#      APPLICATIONS — cascades to the operator and application subprojects
#      so the .deb gets actual content. pkg/ is add_subdirectory()'d
#      *before* operators/ and applications/, so those see the forced ON
#      value when their own helpers run option().
add_holohub_package({{ cookiecutter.module_repo_name }}
                    OPERATORS    {{ cookiecutter.operator_slug }}
                    APPLICATIONS {{ cookiecutter.module_slug }}_pipeline)
