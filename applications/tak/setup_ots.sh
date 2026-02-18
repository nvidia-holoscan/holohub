#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# First-run setup for OpenTAKServer (GPL-3.0).
#
# This script downloads and installs OpenTAKServer and its Web UI at runtime
# so that no GPL-licensed code is distributed in the container image itself.
# By running this script, the user is obtaining GPL-3.0-licensed software
# directly from its upstream sources (PyPI and GitHub).
#
# OpenTAKServer: https://github.com/brian7704/OpenTAKServer (GPL-3.0)
# OpenTAKServer-UI: https://github.com/brian7704/OpenTAKServer-UI

set -e

MARKER_FILE=/opt/ots/.setup_complete
if [ -f "$MARKER_FILE" ]; then
    echo "[OTS Setup] Already complete, skipping."
    exit 0
fi

OTS_VENV=/opt/ots/venv
export OTS_DATA_FOLDER=/opt/ots/data

echo "============================================================"
echo "[OTS Setup] Installing OpenTAKServer (GPL-3.0) from PyPI..."
echo "[OTS Setup] Source: https://github.com/brian7704/OpenTAKServer"
echo "============================================================"

# --------------------------------------------------------------------------
# Install OTS and database adapter
# --------------------------------------------------------------------------
"$OTS_VENV/bin/pip" install --no-cache-dir opentakserver psycopg2-binary

# --------------------------------------------------------------------------
# Patch OTS to use psycopg2 (psycopg v3 returns bytes that break Flask-Security)
# --------------------------------------------------------------------------
echo "[OTS Setup] Applying psycopg2 compatibility patch..."
sed -i 's|postgresql+psycopg://|postgresql+psycopg2://|' \
    "$OTS_VENV"/lib/python3.*/site-packages/opentakserver/defaultconfig.py

SA_BASE=$(find "$OTS_VENV" -path '*/dialects/postgresql/base.py' | head -1)
sed -i '/def _get_server_version_info/,/\.scalar()/{s/\.scalar()/\.scalar(); v = v.decode() if isinstance(v, bytes) else v/}' "$SA_BASE"
find "$OTS_VENV" -name '__pycache__' -path '*/sqlalchemy/*' -exec rm -rf {} + 2>/dev/null || true
find "$OTS_VENV" -name '__pycache__' -path '*/opentakserver/*' -exec rm -rf {} + 2>/dev/null || true

# --------------------------------------------------------------------------
# Patch Flask-SocketIO init: increase ping_timeout and allow CORS origins
# --------------------------------------------------------------------------
echo "[OTS Setup] Applying Flask-SocketIO patch..."
OTS_APP_PY=$(find "$OTS_VENV" -path '*/opentakserver/app.py' | head -1)
sed -i 's|ping_timeout=1,|ping_timeout=30, cors_allowed_origins="*",|' "$OTS_APP_PY"
find "$OTS_VENV" -name '__pycache__' -path '*/opentakserver/*' -exec rm -rf {} + 2>/dev/null || true

# --------------------------------------------------------------------------
# Patch cot_parser: allow nullable sender_uid and set marker callsign from
# <remarks> text (markers cannot have a <contact> element).
# --------------------------------------------------------------------------
echo "[OTS Setup] Applying cot_parser patches..."
COT_PARSER=$(find "$OTS_VENV" -path '*/cot_parser/cot_parser.py' | head -1)
sed -i 's|uid = body\["uid"\] or event.attrs\["uid"\]|uid = body.get("uid") or None|' "$COT_PARSER"
sed -i 's|marker.uid = event.attrs\["uid"\]|marker.uid = event.attrs["uid"]; _r = event.find("remarks"); marker.callsign = _r.text if _r and _r.text else event.attrs.get("uid", "")|' "$COT_PARSER"
find "$OTS_VENV" -name '__pycache__' -path '*/cot_parser/*' -exec rm -rf {} + 2>/dev/null || true

# --------------------------------------------------------------------------
# Patch EUD.to_json() to include the latest point for map display
# --------------------------------------------------------------------------
echo "[OTS Setup] Applying EUD model patch..."
EUD_MODEL=$(find "$OTS_VENV" -path '*/models/EUD.py' | head -1)
sed -i 's|"last_point": None,.*# Setting to.*|"last_point": self.points[-1].to_json() if self.points else None,|' "$EUD_MODEL"
find "$OTS_VENV" -name '__pycache__' -path '*/models/*' -exec rm -rf {} + 2>/dev/null || true

# --------------------------------------------------------------------------
# Initialize OTS certificate authority
# --------------------------------------------------------------------------
echo "[OTS Setup] Creating certificate authority..."
mkdir -p "$OTS_DATA_FOLDER"
OTS_APP=$(find "$OTS_VENV" -path '*/opentakserver/app.py' | head -1)
"$OTS_VENV/bin/flask" --app "$OTS_APP" ots create-ca

# --------------------------------------------------------------------------
# Download the OpenTAKServer Web UI
# --------------------------------------------------------------------------
echo "[OTS Setup] Downloading OpenTAKServer Web UI..."
"$OTS_VENV/bin/pip" install --no-cache-dir lastversion
mkdir -p /var/www/html/opentakserver
cd /var/www/html/opentakserver
"$OTS_VENV/bin/lastversion" --assets extract brian7704/OpenTAKServer-UI

# --------------------------------------------------------------------------
# Mark setup as complete
# --------------------------------------------------------------------------
touch "$MARKER_FILE"
echo "============================================================"
echo "[OTS Setup] Complete."
echo "============================================================"
