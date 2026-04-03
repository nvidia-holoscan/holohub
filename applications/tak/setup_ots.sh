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

# Verify a sed patch applied by checking the replacement text exists in the file.
# Usage: verify_patch <file> <grep_pattern> <description>
verify_patch() {
    if ! grep -q "$2" "$1"; then
        echo "[OTS Setup] ERROR: Patch failed — $3" >&2
        echo "[OTS Setup] File: $1" >&2
        echo "[OTS Setup] Expected pattern: $2" >&2
        exit 1
    fi
}

MARKER_FILE=/opt/ots/.setup_complete
if [ -f "$MARKER_FILE" ]; then
    echo "[OTS Setup] Already complete, skipping."
    exit 0
fi

OTS_VENV=/opt/ots/venv
export OTS_DATA_FOLDER=/opt/ots/data
OTS_PATCH_DIR=/opt/ots/patches
OTS_PYTHONPATH="${OTS_PATCH_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo ""
echo "[OTS Setup] =========================================="
echo "[OTS Setup] First-run installation of OpenTAKServer"
echo "[OTS Setup] License: GPL-3.0"
echo "[OTS Setup] Source:  https://github.com/brian7704/OpenTAKServer"
echo "[OTS Setup] =========================================="
echo ""

# --------------------------------------------------------------------------
# Step 1: Install OTS and database adapter from PyPI
# --------------------------------------------------------------------------
echo "[OTS Setup] [1/6] Installing OpenTAKServer from PyPI (this is the slowest step)..."
"$OTS_VENV/bin/pip" install --no-cache-dir opentakserver==1.7.9 sqlalchemy==2.0.44 psycopg2-binary==2.9.11
echo "[OTS Setup] [1/6] Done"

# --------------------------------------------------------------------------
# Step 2: Patch database compatibility
# --------------------------------------------------------------------------
echo "[OTS Setup] [2/6] Patching database compatibility..."
OTS_DEFAULT_CFG=$(find "$OTS_VENV" -path '*/opentakserver/defaultconfig.py' | head -1)
sed -i 's|postgresql+psycopg://|postgresql+psycopg2://|' "$OTS_DEFAULT_CFG"
verify_patch "$OTS_DEFAULT_CFG" "psycopg2" "psycopg2 driver substitution in defaultconfig.py"

mkdir -p "$OTS_PATCH_DIR"
SQLA_PATCH_FILE="$OTS_PATCH_DIR/sitecustomize.py"
cat > "$SQLA_PATCH_FILE" <<'PY'
"""Compatibility shim for OpenTAKServer with SQLAlchemy 2.0.44 and psycopg2."""

import re

import sqlalchemy
from sqlalchemy.dialects.postgresql.base import PGDialect

EXPECTED_SQLALCHEMY_VERSION = "2.0.44"
if sqlalchemy.__version__ != EXPECTED_SQLALCHEMY_VERSION:
    raise RuntimeError(
        "Unexpected SQLAlchemy version for OpenTAKServer compatibility shim: "
        f"{sqlalchemy.__version__} != {EXPECTED_SQLALCHEMY_VERSION}"
    )


def _ots_get_server_version_info(self, connection):
    v = connection.exec_driver_sql("select pg_catalog.version()").scalar()
    if isinstance(v, (bytes, bytearray)):
        v = v.decode()
    m = re.match(
        r".*(?:PostgreSQL|EnterpriseDB) "
        r"(\d+)\.?(\d+)?(?:\.(\d+))?(?:\.\d+)?(?:devel|beta)?",
        v,
    )
    if not m:
        raise AssertionError(f"Could not determine version from string '{v}'")
    return tuple(int(x) for x in m.group(1, 2, 3) if x is not None)


PGDialect._get_server_version_info = _ots_get_server_version_info
PY
verify_patch "$SQLA_PATCH_FILE" 'EXPECTED_SQLALCHEMY_VERSION = "2.0.44"' "SQLAlchemy compatibility shim version guard"
verify_patch "$SQLA_PATCH_FILE" "v.decode()" "SQLAlchemy bytes decode compatibility shim"
find "$OTS_VENV" -name '__pycache__' -path '*/opentakserver/*' -exec rm -rf {} + 2>/dev/null || true
echo "[OTS Setup] [2/6] Done"

# --------------------------------------------------------------------------
# Step 3: Patch Flask-SocketIO (ping timeout + CORS)
# --------------------------------------------------------------------------
echo "[OTS Setup] [3/6] Patching Flask-SocketIO settings..."
OTS_APP_PY=$(find "$OTS_VENV" -path '*/opentakserver/app.py' | head -1)
sed -i 's|ping_timeout=1,|ping_timeout=30, cors_allowed_origins=[origin.strip() for origin in __import__("os").getenv("OTS_SOCKETIO_CORS_ALLOWED_ORIGINS", "http://localhost:18080,http://127.0.0.1:18080").split(",") if origin.strip()],|' "$OTS_APP_PY"
verify_patch "$OTS_APP_PY" 'OTS_SOCKETIO_CORS_ALLOWED_ORIGINS' "Flask-SocketIO CORS origins in app.py"
verify_patch "$OTS_APP_PY" 'ping_timeout=30' "Flask-SocketIO ping timeout in app.py"
find "$OTS_VENV" -name '__pycache__' -path '*/opentakserver/*' -exec rm -rf {} + 2>/dev/null || true
echo "[OTS Setup] [3/6] Done"

# --------------------------------------------------------------------------
# Step 4: Patch cot_parser (nullable sender_uid + marker callsign)
# --------------------------------------------------------------------------
echo "[OTS Setup] [4/6] Patching CoT parser for marker support..."
COT_PARSER=$(find "$OTS_VENV" -path '*/cot_parser/cot_parser.py' | head -1)
sed -i 's|uid = body\["uid"\] or event.attrs\["uid"\]|uid = body.get("uid") or None|' "$COT_PARSER"
verify_patch "$COT_PARSER" 'body.get("uid")' "nullable uid in cot_parser.py"
sed -i 's|marker.uid = event.attrs\["uid"\]|marker.uid = event.attrs["uid"]; _r = event.find("remarks"); marker.callsign = _r.text if _r and _r.text else event.attrs.get("uid", "")|' "$COT_PARSER"
verify_patch "$COT_PARSER" 'marker.callsign' "marker callsign in cot_parser.py"
find "$OTS_VENV" -name '__pycache__' -path '*/cot_parser/*' -exec rm -rf {} + 2>/dev/null || true

EUD_MODEL=$(find "$OTS_VENV" -path '*/models/EUD.py' | head -1)
sed -i 's|"last_point": None,.*# Setting to.*|"last_point": self.points[-1].to_json() if self.points else None,|' "$EUD_MODEL"
verify_patch "$EUD_MODEL" 'self.points\[-1\].to_json()' "last_point in EUD.py"
find "$OTS_VENV" -name '__pycache__' -path '*/models/*' -exec rm -rf {} + 2>/dev/null || true
echo "[OTS Setup] [4/6] Done"

# --------------------------------------------------------------------------
# Step 5: Initialize certificate authority
# --------------------------------------------------------------------------
echo "[OTS Setup] [5/6] Creating certificate authority..."
mkdir -p "$OTS_DATA_FOLDER"
OTS_APP=$(find "$OTS_VENV" -path '*/opentakserver/app.py' | head -1)
PYTHONPATH="$OTS_PYTHONPATH" "$OTS_VENV/bin/flask" --app "$OTS_APP" ots create-ca
echo "[OTS Setup] [5/6] Done"

# --------------------------------------------------------------------------
# Step 6: Download the Web UI
# --------------------------------------------------------------------------
echo "[OTS Setup] [6/6] Downloading pinned OpenTAKServer Web UI release from GitHub..."
OTS_UI_VERSION="${OTS_UI_VERSION:-v1.7.4}"
OTS_UI_ASSET="${OTS_UI_ASSET:-OpenTAKServer-UI-${OTS_UI_VERSION}.zip}"
OTS_UI_URL="${OTS_UI_URL:-https://github.com/brian7704/OpenTAKServer-UI/releases/download/${OTS_UI_VERSION}/${OTS_UI_ASSET}}"
OTS_UI_ROOT=/var/www/html
OTS_UI_TARGET="${OTS_UI_ROOT}/opentakserver"
OTS_UI_TMP=$(mktemp -d)
OTS_UI_ARCHIVE="${OTS_UI_TMP}/opentakserver-ui.zip"
trap 'rm -rf "$OTS_UI_TMP"' EXIT

mkdir -p "$OTS_UI_TARGET"
echo "[OTS Setup] [6/6] UI release: ${OTS_UI_VERSION} (${OTS_UI_ASSET})"
wget -q -O "$OTS_UI_ARCHIVE" "$OTS_UI_URL"
if [ ! -s "$OTS_UI_ARCHIVE" ]; then
    echo "[OTS Setup] ERROR: OpenTAKServer Web UI download failed: ${OTS_UI_URL}" >&2
    exit 1
fi

export OTS_UI_TARGET OTS_UI_TMP OTS_UI_ARCHIVE
"$OTS_VENV/bin/python" - <<'PY'
import os
import shutil
import zipfile

tmp_dir = os.environ["OTS_UI_TMP"]
ui_target = os.environ["OTS_UI_TARGET"]
archive_path = os.environ["OTS_UI_ARCHIVE"]
extract_root = os.path.join(tmp_dir, "extract")

with zipfile.ZipFile(archive_path) as archive:
    archive.extractall(extract_root)

source_root = os.path.join(extract_root, "opentakserver")
if not os.path.isdir(source_root):
    raise RuntimeError(f"Missing expected UI root in archive: {source_root}")

for entry in os.listdir(ui_target):
    path = os.path.join(ui_target, entry)
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        os.unlink(path)

for entry in os.listdir(source_root):
    src = os.path.join(source_root, entry)
    dst = os.path.join(ui_target, entry)
    if os.path.isdir(src) and not os.path.islink(src):
        shutil.copytree(src, dst, symlinks=True)
    else:
        shutil.copy2(src, dst, follow_symlinks=False)
PY

if [ ! -f "${OTS_UI_TARGET}/index.html" ]; then
    echo "[OTS Setup] ERROR: OpenTAKServer Web UI extraction failed: missing ${OTS_UI_TARGET}/index.html" >&2
    exit 1
fi

rm -rf "$OTS_UI_TMP"
trap - EXIT
echo "[OTS Setup] [6/6] Done"

# --------------------------------------------------------------------------
# Mark setup as complete
# --------------------------------------------------------------------------
touch "$MARKER_FILE"
echo ""
echo "[OTS Setup] =========================================="
echo "[OTS Setup] Installation complete."
echo "[OTS Setup] Subsequent launches will skip this step."
echo "[OTS Setup] =========================================="
echo ""
