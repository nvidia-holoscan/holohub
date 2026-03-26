#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Start OpenTAKServer services inside the container.
# PostgreSQL, RabbitMQ, and first-run setup run in parallel for fast startup.

set -e

OTS_VENV=/opt/ots/venv
export OTS_DATA_FOLDER=/opt/ots/data
OTS_PATCH_DIR=/opt/ots/patches
OTS_PYTHONPATH="${OTS_PATCH_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PG_CONFIG=/etc/postgresql/16/main/postgresql.conf
PG_PORT=$(awk '$1 == "port" {print $3; exit}' "$PG_CONFIG" 2>/dev/null)
PG_PORT=${PG_PORT:-5432}
OTS_API_PORT=${OTS_API_PORT:-18081}
OTS_COT_PORT=${OTS_COT_PORT:-18088}
OTS_WEB_PORT=${OTS_WEB_PORT:-18080}
export OTS_SOCKETIO_CORS_ALLOWED_ORIGINS=${OTS_SOCKETIO_CORS_ALLOWED_ORIGINS:-http://localhost:${OTS_WEB_PORT},http://127.0.0.1:${OTS_WEB_PORT}}

FIRST_RUN=false
if [ ! -f /opt/ots/.setup_complete ]; then
    FIRST_RUN=true
fi

echo ""
echo "============================================================"
echo "[OTS] Starting OpenTAKServer"
if [ "$FIRST_RUN" = true ]; then
    echo "[OTS]"
    echo "[OTS] FIRST RUN DETECTED"
    echo "[OTS] OpenTAKServer will be downloaded and installed from"
    echo "[OTS] PyPI. This may take 1-2 minutes depending on your"
    echo "[OTS] network speed. Subsequent launches will be faster."
fi
echo "============================================================"
echo ""
echo "[OTS] PostgreSQL configured port: ${PG_PORT}"

# --- Launch PostgreSQL, RabbitMQ, and first-run setup in parallel ----------

# PostgreSQL (background)
(
    echo "[OTS] Starting PostgreSQL..."
    chmod 700 /var/lib/postgresql/16/main 2>/dev/null || true
    pg_ctlcluster 16 main start 2>/dev/null || true
    pg_ready=false
    for i in $(seq 1 15); do
        if pg_isready -h 127.0.0.1 -p "$PG_PORT" -U ots -d ots -q 2>/dev/null; then
            pg_ready=true
            break
        fi
        sleep 1
    done
    if [ "$pg_ready" = true ]; then
        echo "[OTS] PostgreSQL ready (port $PG_PORT)"
    else
        echo "[OTS] ERROR: PostgreSQL failed to start on port $PG_PORT" >&2
        exit 1
    fi
) &
PG_PID=$!

# RabbitMQ (background)
(
    echo "[OTS] Starting RabbitMQ..."
    export RABBITMQ_NODENAME=rabbit@localhost
    export RABBITMQ_HOME=/tmp/rabbitmq/home
    export HOME="$RABBITMQ_HOME"
    export RABBITMQ_CONF_ENV_FILE=/tmp/rabbitmq/rabbitmq-env.conf
    export RABBITMQ_MNESIA_BASE=/tmp/rabbitmq/mnesia
    export RABBITMQ_LOG_BASE=/tmp/rabbitmq/log
    export RABBITMQ_PID_FILE=/tmp/rabbitmq/rabbit.pid
    export ERL_EPMD_ADDRESS=127.0.0.1
    mkdir -p "$RABBITMQ_HOME" "$RABBITMQ_MNESIA_BASE" "$RABBITMQ_LOG_BASE"
    chmod 700 "$RABBITMQ_HOME"
    : > "$RABBITMQ_CONF_ENV_FILE"

    epmd -daemon 2>/dev/null || true
    rabbitmq-server -detached 2>/dev/null || true

    rmq_ready=false
    for i in $(seq 1 30); do
        if rabbitmqctl -n "$RABBITMQ_NODENAME" status >/dev/null 2>&1; then
            rmq_ready=true
            break
        fi
        sleep 1
    done
    if [ "$rmq_ready" = true ]; then
        echo "[OTS] RabbitMQ ready"
    else
        echo "[OTS] ERROR: RabbitMQ failed to start" >&2
        exit 1
    fi
) &
RMQ_PID=$!

# First-run setup (background -- pip install, patches, CA, Web UI)
if [ "$FIRST_RUN" = true ]; then
    /opt/ots/setup_ots.sh &
    SETUP_PID=$!
else
    SETUP_PID=""
fi

# Wait for all parallel tasks to finish
if [ "$FIRST_RUN" = true ]; then
    echo "[OTS] Waiting for PostgreSQL + RabbitMQ + first-run install (in parallel)..."
else
    echo "[OTS] Waiting for PostgreSQL + RabbitMQ..."
fi
wait $PG_PID
wait $RMQ_PID
[ -n "$SETUP_PID" ] && wait $SETUP_PID
echo "[OTS] All services initialized"

# Re-export RabbitMQ env (subshell exports don't propagate)
export RABBITMQ_NODENAME=rabbit@localhost
export RABBITMQ_HOME=/tmp/rabbitmq/home
export HOME="$RABBITMQ_HOME"

# --- Patch OTS config -------------------------------------------------
OTS_DB_URI="postgresql+psycopg2://ots:ots@127.0.0.1:${PG_PORT}/ots"
if [ -f "$OTS_DATA_FOLDER/config.yml" ]; then
    echo "[OTS] Configuring database and port settings..."
    sed -i "s|SQLALCHEMY_DATABASE_URI:.*|SQLALCHEMY_DATABASE_URI: ${OTS_DB_URI}|" "$OTS_DATA_FOLDER/config.yml"
    sed -i "s|OTS_LISTENER_PORT:.*|OTS_LISTENER_PORT: ${OTS_API_PORT}|" "$OTS_DATA_FOLDER/config.yml"
    sed -i "s|OTS_TCP_STREAMING_PORT:.*|OTS_TCP_STREAMING_PORT: ${OTS_COT_PORT}|" "$OTS_DATA_FOLDER/config.yml"
else
    echo "[OTS] Generating initial config..."
    PYTHONPATH="$OTS_PYTHONPATH" timeout 5 "$OTS_VENV/bin/opentakserver" >/dev/null 2>&1 || true
    sleep 1
    if [ -f "$OTS_DATA_FOLDER/config.yml" ]; then
        sed -i "s|SQLALCHEMY_DATABASE_URI:.*|SQLALCHEMY_DATABASE_URI: ${OTS_DB_URI}|" "$OTS_DATA_FOLDER/config.yml"
        sed -i "s|OTS_LISTENER_PORT:.*|OTS_LISTENER_PORT: ${OTS_API_PORT}|" "$OTS_DATA_FOLDER/config.yml"
        sed -i "s|OTS_TCP_STREAMING_PORT:.*|OTS_TCP_STREAMING_PORT: ${OTS_COT_PORT}|" "$OTS_DATA_FOLDER/config.yml"
    fi
fi

# --- Start OTS processes ----------------------------------------------
echo "[OTS] Starting opentakserver API (port ${OTS_API_PORT})..."
PYTHONPATH="$OTS_PYTHONPATH" "$OTS_VENV/bin/opentakserver" > /tmp/ots_api.log 2>&1 &

# Wait for API port instead of hardcoded sleep
echo "[OTS] Waiting for API to accept connections..."
api_ready=false
for i in $(seq 1 15); do
    if python3 -c "import socket; s=socket.socket(); s.settimeout(0.5); s.connect(('127.0.0.1',$OTS_API_PORT)); s.close()" 2>/dev/null; then
        api_ready=true
        echo "[OTS] API ready (port ${OTS_API_PORT})"
        break
    fi
    sleep 0.5
done
if [ "$api_ready" != true ]; then
    echo "[OTS] WARNING: API not responding on port ${OTS_API_PORT}, continuing anyway" >&2
fi

echo "[OTS] Starting cot_parser..."
PYTHONPATH="$OTS_PYTHONPATH" "$OTS_VENV/bin/cot_parser" > /tmp/ots_cot_parser.log 2>&1 &

echo "[OTS] Starting eud_handler (TCP CoT on port ${OTS_COT_PORT})..."
PYTHONPATH="$OTS_PYTHONPATH" "$OTS_VENV/bin/eud_handler" > /tmp/ots_eud_handler.log 2>&1 &

# --- Nginx (Web UI) ---
echo "[OTS] Starting nginx (Web UI on port ${OTS_WEB_PORT})..."
sed -i "s/\${OTS_WEB_PORT}/${OTS_WEB_PORT}/g; s/\${OTS_API_PORT}/${OTS_API_PORT}/g" \
    /etc/nginx/sites-available/ots
nginx 2>/dev/null || true

echo ""
echo "============================================================"
if [ "$api_ready" = true ]; then
    echo "[OTS] OpenTAKServer ready"
else
    echo "[OTS] OpenTAKServer started (API may not be fully ready)"
fi
echo "[OTS]   Web UI:   http://localhost:${OTS_WEB_PORT}"
echo "[OTS]   TCP CoT:  port ${OTS_COT_PORT}"
echo "[OTS]   HTTP API: port ${OTS_API_PORT}"
echo "[OTS]   Logs:     /tmp/ots_*.log"
echo "============================================================"
echo ""
