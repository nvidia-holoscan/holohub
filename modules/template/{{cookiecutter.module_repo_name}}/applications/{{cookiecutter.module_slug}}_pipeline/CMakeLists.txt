# SPDX-FileCopyrightText: Copyright (c) {% now 'utc', '%Y' %} {{ cookiecutter.full_name }}{% if cookiecutter.affiliation %} / {{ cookiecutter.affiliation }}{% endif %}. All rights reserved.
# SPDX-License-Identifier: {{ cookiecutter._license }}

# The example pipeline ships a Python implementation for every module, plus a
# C++ implementation for cpp-language modules. Each lives in its own
# subdirectory with its own metadata.json so the Holoscan CLI can discover and
# run them per language (e.g. `./holohub run {{ cookiecutter.module_slug }}_pipeline --language python`).
add_subdirectory(python)
{% if cookiecutter.language == 'cpp' %}add_subdirectory(cpp)
{% endif %}
