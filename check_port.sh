#!/bin/bash
#
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

# Port Checker Script
# Usage: ./check_port.sh [PORT_NUMBER]
# Default port: 49010

PORT=${1:-49010}

echo "========================================"
echo "🔍 PORT $PORT STATUS CHECK"
echo "========================================"

# Function to check command availability
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 1. Check if port is listening with netstat
echo ""
echo "📡 1. Checking if port $PORT is listening (netstat):"
if command_exists netstat; then
    NETSTAT_RESULT=$(netstat -tulpn 2>/dev/null | grep ":$PORT ")
    if [ -n "$NETSTAT_RESULT" ]; then
        echo "✅ Port $PORT is LISTENING:"
        echo "$NETSTAT_RESULT"
    else
        echo "❌ Port $PORT is NOT listening"
    fi
else
    echo "⚠️  netstat not available"
fi

# 2. Check with ss command
echo ""
echo "📡 2. Checking if port $PORT is listening (ss):"
if command_exists ss; then
    SS_RESULT=$(ss -tulpn 2>/dev/null | grep ":$PORT ")
    if [ -n "$SS_RESULT" ]; then
        echo "✅ Port $PORT is LISTENING:"
        echo "$SS_RESULT"
    else
        echo "❌ Port $PORT is NOT listening"
    fi
else
    echo "⚠️  ss not available"
fi

# 3. Check with lsof
echo ""
echo "📡 3. Checking processes using port $PORT (lsof):"
if command_exists lsof; then
    LSOF_RESULT=$(lsof -i :$PORT 2>/dev/null)
    if [ -n "$LSOF_RESULT" ]; then
        echo "✅ Processes using port $PORT:"
        echo "$LSOF_RESULT"
    else
        echo "❌ No processes using port $PORT"
    fi
else
    echo "⚠️  lsof not available"
fi

# 4. Test if port is available for binding
echo ""
echo "🔧 4. Testing if port $PORT is available for binding:"
if command_exists nc; then
    # Try to bind to the port
    timeout 2 nc -l $PORT </dev/null >/dev/null 2>&1 &
    NC_PID=$!
    sleep 0.5

    # Check if nc is still running (successful bind)
    if kill -0 $NC_PID 2>/dev/null; then
        echo "✅ Port $PORT is AVAILABLE (can bind successfully)"
        kill $NC_PID 2>/dev/null
        wait $NC_PID 2>/dev/null
    else
        echo "❌ Port $PORT is NOT AVAILABLE (bind failed - likely in use)"
    fi
else
    echo "⚠️  nc (netcat) not available for bind test"
fi

# 5. Check if it's a well-known port
echo ""
echo "📋 5. Port information:"
if [ $PORT -lt 1024 ]; then
    echo "⚠️  Port $PORT is a PRIVILEGED port (< 1024) - requires root to bind"
elif [ $PORT -ge 1024 ] && [ $PORT -le 49151 ]; then
    echo "ℹ️  Port $PORT is a REGISTERED port (1024-49151)"
else
    echo "ℹ️  Port $PORT is a DYNAMIC/PRIVATE port (49152-65535)"
fi

echo ""
echo "========================================"
echo "✅ Port $PORT check completed!"
echo "========================================"
