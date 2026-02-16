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

FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive

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
# OpenTAKServer (CoT server for receiving detection markers)
#
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       rabbitmq-server python3.12-venv \
       postgresql postgresql-postgis libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Re-create PostgreSQL cluster owned by uid 1000 so non-root can start it
RUN pg_dropcluster 16 main \
    && pg_createcluster -u 1000 -g 1000 16 main \
    && sed -i 's/peer$/trust/' /etc/postgresql/16/main/pg_hba.conf \
    && pg_ctlcluster 16 main start \
    && sleep 2 \
    && PG_SUPER=$(getent passwd 1000 | cut -d: -f1) \
    && psql -U "$PG_SUPER" -d postgres -c "CREATE USER ots WITH PASSWORD 'ots';" \
    && psql -U "$PG_SUPER" -d postgres -c "CREATE DATABASE ots OWNER ots;" \
    && psql -U "$PG_SUPER" -d ots -c "CREATE EXTENSION IF NOT EXISTS postgis;" \
    && pg_ctlcluster 16 main stop \
    && chmod -R 777 /var/run/postgresql /etc/postgresql /var/log/postgresql

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

# Install OpenTAKServer in an isolated venv to avoid dependency conflicts
RUN python3 -m venv /opt/ots/venv \
    && /opt/ots/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
    && /opt/ots/venv/bin/pip install --no-cache-dir opentakserver psycopg2-binary

# Patch OTS to use psycopg2 (psycopg v3 returns bytes that break Flask-Security)
RUN sed -i 's|postgresql+psycopg://|postgresql+psycopg2://|' \
        /opt/ots/venv/lib/python3.12/site-packages/opentakserver/defaultconfig.py \
    && SA_BASE=$(find /opt/ots/venv -path '*/dialects/postgresql/base.py' | head -1) \
    && sed -i '/def _get_server_version_info/,/\.scalar()/{s/\.scalar()/\.scalar(); v = v.decode() if isinstance(v, bytes) else v/}' "$SA_BASE" \
    && find /opt/ots/venv -name '__pycache__' -path '*/sqlalchemy/*' -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/ots/venv -name '__pycache__' -path '*/opentakserver/*' -exec rm -rf {} + 2>/dev/null || true

# Patch Flask-SocketIO init: increase ping_timeout (1s is too aggressive through
# nginx proxy) and allow all CORS origins for the Web UI reverse-proxy setup.
RUN OTS_APP_PY=$(find /opt/ots/venv -path '*/opentakserver/app.py' | head -1) \
    && sed -i 's|ping_timeout=1,|ping_timeout=30, cors_allowed_origins="*",|' "$OTS_APP_PY" \
    && find /opt/ots/venv -name '__pycache__' -path '*/opentakserver/*' -exec rm -rf {} + 2>/dev/null || true

# Patch cot_parser: when the eud_handler sends body["uid"]=None (which happens
# when the TCP connection's sender uid isn't set), don't fall back to the event's
# uid—that uid won't exist as an EUD and the FK constraint on cot.sender_uid fails.
# Instead, let sender_uid be NULL (the column is nullable).
# Also set marker callsign from <remarks> text (since markers can't have <contact>).
RUN COT_PARSER=$(find /opt/ots/venv -path '*/cot_parser/cot_parser.py' | head -1) \
    && sed -i 's|uid = body\["uid"\] or event.attrs\["uid"\]|uid = body.get("uid") or None|' "$COT_PARSER" \
    && sed -i 's|marker.uid = event.attrs\["uid"\]|marker.uid = event.attrs["uid"]; _r = event.find("remarks"); marker.callsign = _r.text if _r and _r.text else event.attrs.get("uid", "")|' "$COT_PARSER" \
    && find /opt/ots/venv -name '__pycache__' -path '*/cot_parser/*' -exec rm -rf {} + 2>/dev/null || true

# Patch EUD.to_json() to include the latest point so the Web UI can place the
# EUD on the map (OTS hardcodes last_point=None for performance, but we need it).
RUN EUD_MODEL=$(find /opt/ots/venv -path '*/models/EUD.py' | head -1) \
    && sed -i 's|"last_point": None,.*# Setting to.*|"last_point": self.points[-1].to_json() if self.points else None,|' "$EUD_MODEL" \
    && find /opt/ots/venv -name '__pycache__' -path '*/models/*' -exec rm -rf {} + 2>/dev/null || true

# Initialize the OTS certificate authority and data directory
ENV OTS_DATA_FOLDER=/opt/ots/data
RUN mkdir -p /opt/ots/data \
    && OTS_APP=$(find /opt/ots/venv -path '*/opentakserver/app.py' | head -1) \
    && /opt/ots/venv/bin/flask --app "$OTS_APP" ots create-ca

# --------------------------------------------------------------------------
#
# OpenTAKServer Web UI (static SPA served by nginx)
#
RUN apt-get update \
    && apt-get install -y --no-install-recommends nginx \
    && rm -rf /var/lib/apt/lists/*

# Download pre-built OTS Web UI from GitHub releases
RUN /opt/ots/venv/bin/pip install --no-cache-dir lastversion \
    && mkdir -p /var/www/html/opentakserver \
    && cd /var/www/html/opentakserver \
    && /opt/ots/venv/bin/lastversion --assets extract brian7704/OpenTAKServer-UI

# Nginx config: serve static UI and reverse-proxy /api + /socket.io to OTS
COPY applications/tak/ots_nginx.conf /etc/nginx/sites-available/ots
RUN rm -f /etc/nginx/sites-enabled/default \
    && ln -sf /etc/nginx/sites-available/ots /etc/nginx/sites-enabled/ots \
    && chmod -R 777 /var/www/html/opentakserver /var/log/nginx /var/lib/nginx /run

# Copy startup script and make everything accessible to non-root users
COPY applications/tak/start_ots.sh /opt/ots/start_ots.sh
RUN chmod +x /opt/ots/start_ots.sh \
    && chmod -R 777 /opt/ots /var/lib/rabbitmq /var/log/rabbitmq /etc/rabbitmq

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
