# SPDX-FileCopyrightText: Copyright (c) {% now 'utc', '%Y' %} {{ cookiecutter.full_name }}{% if cookiecutter.affiliation %} / {{ cookiecutter.affiliation }}{% endif %}. All rights reserved.
# SPDX-License-Identifier: {{ cookiecutter._license }}
#
# Functional tests for {{ cookiecutter.operator_slug.split('_')|map('capitalize')|join('') }}.
# TODO: extend test_placeholder() with real pipeline coverage.
# See holoscan-example-module/tests/python/ for worked examples.
{%- set op_class = cookiecutter.operator_slug.split('_')|map('capitalize')|join('') %}

import importlib
import inspect

import pytest

# Skip the whole module only when the Holoscan SDK itself is unavailable
# (e.g. CUDA missing outside the dev container). We deliberately do NOT
# importorskip("holoscan.{{ cookiecutter.module_slug }}") at module level — that
# would mask an actual build/import failure of *our* module as a "Skipped"
# result, and CTest would silently report success-with-zero-tests (exit code 5).
pytest.importorskip("holoscan", reason="holoscan SDK not installed", exc_type=ImportError)


EXPECTED_OPERATORS = (
    "{{ op_class }}",
)


@pytest.mark.parametrize("operator_name", EXPECTED_OPERATORS)
def test_operator_is_importable(operator_name):
    module = importlib.import_module("holoscan.{{ cookiecutter.module_slug }}")

    assert hasattr(module, operator_name), (
        f"holoscan.{{ cookiecutter.module_slug }} does not expose {operator_name}; "
        f"available names: {sorted(n for n in dir(module) if not n.startswith('_'))}"
    )
    operator_cls = getattr(module, operator_name)
    assert inspect.isclass(operator_cls), (
        f"holoscan.{{ cookiecutter.module_slug }}.{operator_name} is not a class "
        f"(got {type(operator_cls)!r})"
    )


def test_placeholder():
    # TODO: build a minimal Application, call app.run(), assert on results.
    pass
