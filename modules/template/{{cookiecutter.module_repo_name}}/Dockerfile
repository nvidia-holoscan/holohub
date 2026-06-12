# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) {% now 'utc', '%Y' %} {{ cookiecutter.full_name }}{% if cookiecutter.affiliation %} / {{ cookiecutter.affiliation }}{% endif %}. All rights reserved.
# SPDX-License-Identifier: {{ cookiecutter._license }}
#
# Development container for {{ cookiecutter.project_name }}.
#
# Extends the Holoscan SDK image with the build tools needed to configure,
# compile, and test the module from a live source mount. No source is copied
# into the image; the project tree is bind-mounted by the holoscan-cli at
# /workspace/{{ cookiecutter.module_slug }} when the container is launched.
#
# Build & run via the holoscan-cli wrapper (recommended):
#   ./holohub run-container
#
# Manual build (without the CLI):
#   docker build -t holoscan-{{ cookiecutter.module_slug.replace('_', '-') }} .

ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v{{ cookiecutter.holoscan_version }}-cuda13-dgpu
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

# Install holoscan-cli developer tools
ARG HOLOSCAN_CLI_INSTALL_ARGS="--pre --extra-index-url https://pypi.nvidia.com holoscan-cli>4.2.0"
RUN if ! python3 -m holoscan_cli --help 2>/dev/null | grep -qw build; then \
        python3 -m pip install ${HOLOSCAN_CLI_INSTALL_ARGS} \
    ; fi

# TODO: add module-specific runtime / build dependencies below.
{% if cookiecutter.language == "cpp" %}RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libgtest-dev \
        clang-format \
        pybind11-dev \
    && rm -rf /var/lib/apt/lists/*
{% endif %}
# Python tooling: pytest-timeout for tests, build + scikit-build-core for
# `./holohub package --pkg-generator WHEEL` (drives `python -m build`).
RUN pip install --no-cache-dir pytest-timeout build scikit-build-core
