# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) {{ cookiecutter.year }} {{ cookiecutter.author }}
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

############################################################
# Base image
############################################################

ARG BASE_IMAGE
ARG GPU_TYPE

FROM ${BASE_IMAGE} as base

ARG DEBIAN_FRONTEND=noninteractive

# --------------------------------------------------------------------------
#
# Holohub dev setup
#
RUN mkdir -p /tmp/scripts
COPY holohub /tmp/scripts/
RUN mkdir -p /tmp/scripts/utilities
COPY utilities /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/holohub
RUN /tmp/scripts/holohub setup && rm -rf /var/lib/apt/lists/*

# Enable autocomplete
RUN echo ". /etc/bash_completion.d/holohub_autocomplete" >> /etc/bash.bashrc

# - This variable is consumed by all dependencies below as an environment variable (CMake 3.22+)
# - We use ARG to only set it at docker build time, so it does not affect cmake builds
#   performed at docker run time in case users want to use a different BUILD_TYPE
ARG CMAKE_BUILD_TYPE=Release

# --------------------------------------------------------------------------
#
# {{ cookiecutter.project_name }} specific dependencies
#
{% if cookiecutter.language == "python" %}
# Copy over app-specific requirements
COPY applications/{{ cookiecutter.project_slug }}/requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt
{% endif %}

# TODO: Add any additional system dependencies required for {{ cookiecutter.project_name }}
# Example:
# RUN apt update && apt install -y \
#     your-package-here \
#     another-package \
#     && rm -rf /var/lib/apt/lists/*

# TODO: Add any additional setup steps required for {{ cookiecutter.project_name }}
# Example:
# RUN your-setup-command-here

# --------------------------------------------------------------------------
#
# Application setup
#
WORKDIR /workspace/holohub

# Set default environment variables for the application
ENV HOLOSCAN_INPUT_PATH=/workspace/holohub/data
# TODO: Add any application-specific environment variables here
# ENV YOUR_APP_ENV_VAR=value

# TODO: Uncomment and modify the following if your application needs specific expose ports
# EXPOSE 8080

# TODO: Uncomment and modify the following if your application needs a specific entrypoint
# ENTRYPOINT ["./holohub", "run", "{{ cookiecutter.project_slug }}"]
