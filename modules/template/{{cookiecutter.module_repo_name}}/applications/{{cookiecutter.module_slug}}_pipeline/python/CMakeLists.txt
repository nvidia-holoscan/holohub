# SPDX-FileCopyrightText: Copyright (c) {% now 'utc', '%Y' %} {{ cookiecutter.full_name }}{% if cookiecutter.affiliation %} / {{ cookiecutter.affiliation }}{% endif %}. All rights reserved.
# SPDX-License-Identifier: {{ cookiecutter._license }}

# Python application — no compilation needed.
# Install the script so 'cmake --install' makes it available on PATH.
install(PROGRAMS {{ cookiecutter.module_slug }}_pipeline.py
    DESTINATION bin
    RENAME {{ cookiecutter.module_slug }}_pipeline)
