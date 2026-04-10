# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

ARG BASE_IMAGE
ARG GPU_TYPE
ARG APP_USER=appuser
ARG APP_UID=1000
ARG APP_GID=1000

FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive
ARG APP_USER
ARG APP_UID
ARG APP_GID

# --------------------------------------------------------------------------
#
# Install required apt packages and python modules
#
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    ultralytics \
    opencv-python-headless \
    pytak \
    aiohttp \
    cryptography \
    lap

# --------------------------------------------------------------------------
#
# OpenTAKServer infrastructure (permissively-licensed system packages only).
# OTS itself (GPL-3.0) is installed at runtime by setup_ots.sh to avoid
# distributing GPL code in the container image.
#
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       rabbitmq-server python3.12-venv \
       postgresql postgresql-postgis libpq-dev \
       nginx \
    && rm -rf /var/lib/apt/lists/*

# Create the dedicated non-root runtime user before preparing service state.
RUN set -eux; \
    if ! getent group "${APP_GID}" >/dev/null; then \
        groupadd --gid "${APP_GID}" "${APP_USER}"; \
    fi; \
    if ! getent passwd "${APP_USER}" >/dev/null; then \
        if getent passwd "${APP_UID}" >/dev/null; then \
            useradd --non-unique --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /bin/bash "${APP_USER}"; \
        else \
            useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /bin/bash "${APP_USER}"; \
        fi; \
    fi

# Re-create PostgreSQL cluster owned by the runtime user's uid/gid so it can start without root.
RUN pg_dropcluster 16 main \
    && pg_createcluster -u "${APP_UID}" -g "${APP_GID}" 16 main \
    && sed -i 's/peer$/trust/' /etc/postgresql/16/main/pg_hba.conf \
    && pg_ctlcluster 16 main start \
    && sleep 2 \
    && PG_SUPER=$(getent passwd "${APP_UID}" | head -n1 | cut -d: -f1) \
    && psql -U "${PG_SUPER}" -d postgres -c "CREATE USER ots WITH PASSWORD 'ots';" \
    && psql -U "${PG_SUPER}" -d postgres -c "CREATE DATABASE ots OWNER ots;" \
    && psql -U "${PG_SUPER}" -d ots -c "CREATE EXTENSION IF NOT EXISTS postgis;" \
    && pg_ctlcluster 16 main stop \
    && chown -R "${APP_UID}:${APP_GID}" /var/run/postgresql /etc/postgresql /var/log/postgresql

# Generate Erlang cookie at build time (required by RabbitMQ)
RUN openssl rand -hex 16 > /var/lib/rabbitmq/.erlang.cookie \
    && chown "${APP_UID}:${APP_GID}" /var/lib/rabbitmq/.erlang.cookie \
    && chmod 400 /var/lib/rabbitmq/.erlang.cookie

# Enable RabbitMQ plugins required by OpenTAKServer
RUN rabbitmq-plugins enable rabbitmq_mqtt rabbitmq_auth_backend_http

# Replace RabbitMQ /usr/sbin wrappers (symlinks to rabbitmq-script-wrapper
# that reject non-root) with pass-through scripts to the real binaries.
RUN rm -f /usr/sbin/rabbitmq-server /usr/sbin/rabbitmqctl \
          /usr/sbin/rabbitmq-plugins /usr/sbin/rabbitmq-diagnostics \
    && echo '#!/bin/sh'  > /usr/sbin/rabbitmq-server \
    && echo 'exec /usr/lib/rabbitmq/bin/rabbitmq-server "$@"' >> /usr/sbin/rabbitmq-server \
    && echo '#!/bin/sh'  > /usr/sbin/rabbitmqctl \
    && echo 'exec /usr/lib/rabbitmq/bin/rabbitmqctl "$@"' >> /usr/sbin/rabbitmqctl \
    && echo '#!/bin/sh'  > /usr/sbin/rabbitmq-plugins \
    && echo 'exec /usr/lib/rabbitmq/bin/rabbitmq-plugins "$@"' >> /usr/sbin/rabbitmq-plugins \
    && echo '#!/bin/sh'  > /usr/sbin/rabbitmq-diagnostics \
    && echo 'exec /usr/lib/rabbitmq/bin/rabbitmq-diagnostics "$@"' >> /usr/sbin/rabbitmq-diagnostics \
    && chmod +x /usr/sbin/rabbitmq-server /usr/sbin/rabbitmqctl \
                /usr/sbin/rabbitmq-plugins /usr/sbin/rabbitmq-diagnostics

# Prepare empty venv and directories (OTS is installed into these at runtime)
ENV OTS_DATA_FOLDER=/opt/ots/data
RUN python3 -m venv /opt/ots/venv \
    && /opt/ots/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
    && mkdir -p /opt/ots/data /var/www/html/opentakserver /var/lib/nginx /var/log/nginx \
    && touch /run/nginx.pid

# Nginx config: serve static UI and reverse-proxy /api + /socket.io to OTS
COPY applications/tak/ots_nginx.conf /etc/nginx/sites-available/ots
RUN rm -f /etc/nginx/sites-enabled/default \
    && ln -sf /etc/nginx/sites-available/ots /etc/nginx/sites-enabled/ots \
    && chown -R "${APP_UID}:${APP_GID}" /var/www/html/opentakserver /var/log/nginx /var/lib/nginx \
    && chown "${APP_UID}:${APP_GID}" /run/nginx.pid \
    && chown -R "${APP_UID}:${APP_GID}" /etc/nginx/sites-available /etc/nginx/sites-enabled

# Copy startup and setup scripts; make everything accessible to non-root users
COPY applications/tak/setup_ots.sh /opt/ots/setup_ots.sh
COPY applications/tak/start_ots.sh /opt/ots/start_ots.sh
RUN chmod +x /opt/ots/setup_ots.sh /opt/ots/start_ots.sh \
    && chown -R "${APP_UID}:${APP_GID}" /opt/ots /var/lib/rabbitmq /var/log/rabbitmq /etc/rabbitmq

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

ENV HOME=/home/${APP_USER}
USER ${APP_USER}
