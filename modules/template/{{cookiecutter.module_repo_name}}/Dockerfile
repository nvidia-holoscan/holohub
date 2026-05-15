# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) {% now 'utc', '%Y' %} {{ cookiecutter.full_name }}{% if cookiecutter.affiliation %} / {{ cookiecutter.affiliation }}{% endif %}. All rights reserved.
# SPDX-License-Identifier: {{ cookiecutter._license }}
#
# Dev container for {{ cookiecutter.project_name }}.
#
# Extends the Holoscan SDK image with the build tools needed to configure,
# compile, and test the module from a live source mount. No source is copied
# into the image; the project tree is bind-mounted by the HoloHub CLI at
# /workspace/{{ cookiecutter.module_slug }} when the container is launched.
#
# Build & run via the HoloHub CLI (recommended):
#   ./{{ cookiecutter.module_slug }} run-container
#
# Manual build (without the CLI):
#   docker build -t holoscan-{{ cookiecutter.module_slug.replace('_', '-') }} .

ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v{{ cookiecutter.holoscan_version }}-cuda12-dgpu
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

# TODO: add module-specific runtime / build dependencies below.
{% if cookiecutter.language == "cpp" %}RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libgtest-dev \
        clang-format \
        pybind11-dev \
    && rm -rf /var/lib/apt/lists/*
{% endif %}
# Python tooling: pytest-timeout for tests, build + scikit-build-core for
# `./{{ cookiecutter.module_slug }} package --pkg-generator WHEEL` (drives `python -m build`).
RUN pip install --no-cache-dir pytest-timeout build scikit-build-core
