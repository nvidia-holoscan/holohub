# SPDX-FileCopyrightText: Copyright (c) {% now 'utc', '%Y' %} {{ cookiecutter.full_name }}{% if cookiecutter.affiliation %} / {{ cookiecutter.affiliation }}{% endif %}. All rights reserved.
# SPDX-License-Identifier: {{ cookiecutter._license }}
#
# Functional tests for {{ cookiecutter.operator_slug.split('_')|map('capitalize')|join('') }}.
# TODO: replace the placeholder with real pipeline coverage.
# See holoscan-example-module/tests/python/ for worked examples.

import pytest

pytest.importorskip("holoscan", reason="holoscan SDK not installed", exc_type=ImportError)

# TODO: import your operator:
# from holoscan.{{ cookiecutter.module_slug }} import {{ cookiecutter.operator_slug.split('_')|map('capitalize')|join('') }}


def test_placeholder():
    # TODO: build a minimal Application, call app.run(), assert on results.
    pass
