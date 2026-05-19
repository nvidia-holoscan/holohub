# SPDX-FileCopyrightText: Copyright (c) {% now 'utc', '%Y' %} {{ cookiecutter.full_name }}{% if cookiecutter.affiliation %} / {{ cookiecutter.affiliation }}{% endif %}. All rights reserved.
# SPDX-License-Identifier: {{ cookiecutter._license }}
#
# Python pipeline example for {{ cookiecutter.project_name }}.
# TODO: replace the stub with your actual pipeline topology.
#
# Run with:
#   {{ cookiecutter.module_slug | upper }}_BUILD_DIR=build \
#   PYTHONPATH=build/python:$PYTHONPATH \
#   python applications/{{ cookiecutter.module_slug }}_pipeline/{{ cookiecutter.module_slug }}_pipeline.py
{%- set op_class = cookiecutter.operator_slug.split('_')|map('capitalize')|join('') %}

import logging

from holoscan.core import Application
from holoscan.{{cookiecutter.module_slug}} import {{ op_class }}

logging.basicConfig(level=logging.INFO)


class {{ cookiecutter.module_slug.split('_')|map('capitalize')|join('') }}PipelineApp(Application):
    def compose(self) -> None:
        # TODO: instantiate and connect your operators, e.g.:
        op = {{ op_class }}(self, name="{{ cookiecutter.operator_slug }}")  # noqa: F841 (wire up via add_flow below)
        # self.add_flow(source, op, {("out", "in")})


if __name__ == "__main__":
    app = {{ cookiecutter.module_slug.split('_')|map('capitalize')|join('') }}PipelineApp()
    app.run()
