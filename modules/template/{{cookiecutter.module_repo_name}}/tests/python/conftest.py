# SPDX-FileCopyrightText: Copyright (c) {% now 'utc', '%Y' %} {{ cookiecutter.full_name }}{% if cookiecutter.affiliation %} / {{ cookiecutter.affiliation }}{% endif %}. All rights reserved.
# SPDX-License-Identifier: {{ cookiecutter._license }}
#
# conftest.py — makes the build-tree Python package visible to pytest.
#
# Holoscan SDK is installed as a regular package (has __init__.py), so Python
# discards namespace-package directories alongside it. We extend holoscan.__path__
# directly after import to insert our build-tree holoscan/ directory, making
# holoscan.{{ cookiecutter.module_slug }} resolvable to our compiled modules.

import os

build_dir = os.environ.get(
    "{{ cookiecutter.module_slug | upper }}_BUILD_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "build"),
)

try:
    import holoscan  # noqa: E402

    build_holoscan_path = os.path.join(build_dir, "python", "holoscan")
    if build_holoscan_path not in holoscan.__path__:
        holoscan.__path__.insert(0, build_holoscan_path)
except ImportError:
    # holoscan not importable (e.g. CUDA not available outside the development container).
    # Individual tests use pytest.importorskip("holoscan") to skip gracefully.
    pass
