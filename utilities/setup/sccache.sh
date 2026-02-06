#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Install RAPIDS sccache for Holoscan SDK projects
# This script downloads and installs the RAPIDS-customized sccache binary.
#
# Prerequisites: wget, tar (must be available on PATH).

set -e

# Configuration
SCCACHE_MIN_VERSION="${SCCACHE_MIN_VERSION:-0.12.0-rapids.20}"
INSTALL_DIR="/opt/sccache"
SYMLINK_PATH="/usr/local/bin/sccache"
BASE_URL="https://github.com/rapidsai/sccache/releases/download"

# Expected format: [v]MAJOR.MINOR.PATCH-rapids.NN (e.g. 0.12.0-rapids.20 or v0.12.0-rapids.20)
SCCACHE_VERSION_REGEX='^v?[0-9]+\.[0-9]+\.[0-9]+-rapids\.[0-9]+$'

# Parse semantic version MAJOR.MINOR.PATCH (ignores suffix like -rapids.20)
parse_version() {
    [[ "$1" =~ ^v?([0-9]+)\.([0-9]+)\.([0-9]+) ]] && echo "${BASH_REMATCH[1]} ${BASH_REMATCH[2]} ${BASH_REMATCH[3]}"
}

# Compare versions: returns 0 if installed >= required
version_gte() {
    local installed=($1) required=($2)
    for i in 0 1 2; do
        [[ ${installed[$i]} -gt ${required[$i]} ]] && return 0
        [[ ${installed[$i]} -lt ${required[$i]} ]] && return 1
    done
    return 0
}

# Cleanup on exit: remove tarball and, on failure, any partially extracted binary.
cleanup_install() {
    local exit_code=$?
    rm -f "$tar_path"
    if [[ $exit_code -ne 0 ]]; then
        rm -f "${INSTALL_DIR}/sccache"
    fi
    exit "$exit_code"
}

# Check if sccache is already installed and meets minimum version
check_existing_sccache() {
    command -v sccache &>/dev/null || return 1
    local ver_output=$(sccache --version 2>/dev/null || true)
    echo "$ver_output" | grep -q "rapids" || return 1

    local installed_ver=$(parse_version "$(echo "$ver_output" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)")
    local required_ver=$(parse_version "$SCCACHE_MIN_VERSION")
    [[ -n "$installed_ver" && -n "$required_ver" ]] || return 1

    if version_gte "$installed_ver" "$required_ver"; then
        echo "sccache already installed and meets minimum version requirement"
        return 0
    fi
    return 1
}

# Main installation
main() {
    # Verify required tools are available
    command -v wget &>/dev/null || { echo "wget is required but not installed." >&2; exit 1; }
    command -v tar &>/dev/null || { echo "tar is required but not installed." >&2; exit 1; }

    if [[ ! "$SCCACHE_MIN_VERSION" =~ $SCCACHE_VERSION_REGEX ]]; then
        echo "ERROR: SCCACHE_MIN_VERSION format [v]MAJOR.MINOR.PATCH-rapids.NN (e.g. 0.12.0-rapids.20)." >&2
        echo "Current value: SCCACHE_MIN_VERSION=${SCCACHE_MIN_VERSION}" >&2
        exit 1
    fi

    # Check if already installed
    if check_existing_sccache; then
        exit 0
    fi

    # Normalize version (ensure 'v' prefix)
    local version="${SCCACHE_MIN_VERSION#v}"
    local INSTALL_VERSION="v${version}"

    # Determine architecture
    local machine arch
    machine=$(uname -m | tr '[:upper:]' '[:lower:]')
    case "$machine" in
        x86_64|amd64)
            arch="x86_64"
            ;;
        aarch64|arm64)
            arch="aarch64"
            ;;
        *)
            arch="$machine"
            ;;
    esac

    local tarball_name="sccache-${INSTALL_VERSION}-${arch}-unknown-linux-musl.tar.gz"
    local url="${BASE_URL}/${INSTALL_VERSION}/${tarball_name}"
    local tar_path="${INSTALL_DIR}/sccache.tar.gz"
    local extracted_rel="sccache-${INSTALL_VERSION}-${arch}-unknown-linux-musl/sccache"

    echo "Installing sccache ${INSTALL_VERSION} for ${arch}..."

    # Ensure cleanup on exit (partial download or failed extraction)
    trap cleanup_install EXIT

    # Prepare install directory
    mkdir -p "$INSTALL_DIR"

    # Download, verify, and extract release tarball
    echo "Downloading from ${url}..."
    if ! wget --quiet --content-disposition "$url" -O "$tar_path" \
        || [[ ! -s "$tar_path" ]] \
        || ! tar -tzf "$tar_path" &>/dev/null \
        || ! tar -xzf "$tar_path" -C "$INSTALL_DIR" --strip-components=1 "$extracted_rel"; then
        echo "ERROR: Failed to download or extract sccache (${INSTALL_VERSION}/${arch})." >&2
        echo "URL: ${url}" >&2
        exit 1
    fi
    chmod u+x "${INSTALL_DIR}/sccache"

    # Create symlink in /usr/local/bin
    ln -sf "${INSTALL_DIR}/sccache" "$SYMLINK_PATH"

    trap - EXIT
    rm -f "$tar_path"

    echo "sccache ${INSTALL_VERSION} installed successfully"
    sccache --version
}

main "$@"
