#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Start OpenTAKServer services inside the container.
# Usage: source /opt/ots/start_ots.sh   (background, returns control)
#        /opt/ots/start_ots.sh           (background, returns control)

set -e

OTS_VENV=/opt/ots/venv
export OTS_DATA_FOLDER=/opt/ots/data

# --- First-run setup (downloads OTS from PyPI if not already installed) ---
if [ ! -f /opt/ots/.setup_complete ]; then
    /opt/ots/setup_ots.sh
fi

# PostgreSQL cluster port (assigned by pg_createcluster)
PG_PORT=5433

# OTS listen ports (offset from defaults to avoid host conflicts with --net host)
OTS_API_PORT=${OTS_API_PORT:-18081}
OTS_COT_PORT=${OTS_COT_PORT:-18088}

# --- PostgreSQL ---
echo "[OTS] Starting PostgreSQL..."
# PostgreSQL requires 0700 on its data directory
chmod 700 /var/lib/postgresql/16/main 2>/dev/null || true
pg_ctlcluster 16 main start 2>/dev/null || true
for i in $(seq 1 15); do
    if pg_isready -p "$PG_PORT" -q 2>/dev/null; then
        echo "[OTS] PostgreSQL ready (port $PG_PORT)"
        break
    fi
    sleep 1
done

# --- RabbitMQ ---
echo "[OTS] Starting RabbitMQ..."
export RABBITMQ_NODENAME=rabbit@localhost
export RABBITMQ_CONF_ENV_FILE=/dev/null
export RABBITMQ_MNESIA_BASE=/tmp/rabbitmq/mnesia
export RABBITMQ_LOG_BASE=/tmp/rabbitmq/log
export RABBITMQ_PID_FILE=/tmp/rabbitmq/rabbit.pid
export ERL_EPMD_ADDRESS=127.0.0.1
export HOME=${HOME:-/tmp}
mkdir -p "$RABBITMQ_MNESIA_BASE" "$RABBITMQ_LOG_BASE"

epmd -daemon 2>/dev/null || true
rabbitmq-server -detached 2>/dev/null || true

echo "[OTS] Waiting for RabbitMQ..."
for i in $(seq 1 30); do
    if rabbitmqctl -n "$RABBITMQ_NODENAME" status >/dev/null 2>&1; then
        echo "[OTS] RabbitMQ ready"
        break
    fi
    sleep 1
done

# --- Patch OTS config ---
OTS_DB_URI="postgresql+psycopg2://ots:ots@127.0.0.1:${PG_PORT}/ots"
if [ -f "$OTS_DATA_FOLDER/config.yml" ]; then
    sed -i "s|SQLALCHEMY_DATABASE_URI:.*|SQLALCHEMY_DATABASE_URI: ${OTS_DB_URI}|" "$OTS_DATA_FOLDER/config.yml"
    sed -i "s|OTS_LISTENER_PORT:.*|OTS_LISTENER_PORT: ${OTS_API_PORT}|" "$OTS_DATA_FOLDER/config.yml"
    sed -i "s|OTS_TCP_STREAMING_PORT:.*|OTS_TCP_STREAMING_PORT: ${OTS_COT_PORT}|" "$OTS_DATA_FOLDER/config.yml"
else
    # Generate config by starting opentakserver briefly, then patch it
    echo "[OTS] Generating initial config..."
    timeout 5 "$OTS_VENV/bin/opentakserver" >/dev/null 2>&1 || true
    sleep 1
    if [ -f "$OTS_DATA_FOLDER/config.yml" ]; then
        sed -i "s|SQLALCHEMY_DATABASE_URI:.*|SQLALCHEMY_DATABASE_URI: ${OTS_DB_URI}|" "$OTS_DATA_FOLDER/config.yml"
        sed -i "s|OTS_LISTENER_PORT:.*|OTS_LISTENER_PORT: ${OTS_API_PORT}|" "$OTS_DATA_FOLDER/config.yml"
        sed -i "s|OTS_TCP_STREAMING_PORT:.*|OTS_TCP_STREAMING_PORT: ${OTS_COT_PORT}|" "$OTS_DATA_FOLDER/config.yml"
    fi
fi

# --- OpenTAKServer processes ---
echo "[OTS] Starting opentakserver (API on :${OTS_API_PORT})..."
"$OTS_VENV/bin/opentakserver" > /tmp/ots_api.log 2>&1 &
sleep 3

echo "[OTS] Starting cot_parser..."
"$OTS_VENV/bin/cot_parser" > /tmp/ots_cot_parser.log 2>&1 &

echo "[OTS] Starting eud_handler (TCP CoT on :${OTS_COT_PORT})..."
"$OTS_VENV/bin/eud_handler" > /tmp/ots_eud_handler.log 2>&1 &

# --- Nginx (Web UI) ---
OTS_WEB_PORT=${OTS_WEB_PORT:-18080}
echo "[OTS] Starting nginx (Web UI on :${OTS_WEB_PORT})..."
nginx 2>/dev/null || true

echo "[OTS] OpenTAKServer running"
echo "[OTS]   Web UI:   http://localhost:${OTS_WEB_PORT}"
echo "[OTS]   TCP CoT:  port ${OTS_COT_PORT}"
echo "[OTS]   HTTP API: port ${OTS_API_PORT}"
echo "[OTS]   Logs:     /tmp/ots_*.log"
