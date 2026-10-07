#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Extract build inputs without installing the BSP or running its package scripts.
# An optional destination root supports inspecting the extracted files outside a container.
set -euo pipefail

l4t_version=${1:?Usage: install_argus_build_deps.sh L4T_VERSION [ARCH [DESTDIR]]}
architecture=${2:-$(dpkg --print-architecture)}
destination=${3:-/}
if [[ ! ${l4t_version} =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Expected an L4T release such as 39.2.0; got ${l4t_version}" >&2
    exit 1
fi
if [[ ${architecture} != arm64 && ${architecture} != aarch64 ]]; then
    echo "Skipping Argus build dependencies: requires an ARM64 target (got ${architecture})"
    exit 0
fi
l4t_series=${l4t_version%.*}
l4t_major=${l4t_version%%.*}
# Preserve NVIDIA's literal L4T repository name.
bsp_repository=som  # codespell:ignore som
if (( l4t_major < 39 )); then
    bsp_repository=t234
fi

download_dir=$(mktemp -d)
trap 'rm -rf "${download_dir}"' EXIT
manifest=${destination%/}/usr/share/holoscan-camera/argus-build-packages.txt
mkdir -p "$(dirname "${manifest}")"
: > "${manifest}"

fetch_package() {
    local repository=$1 package=$2
    local index=${download_dir}/${repository}-Packages
    if [[ ! -f ${index} ]]; then
        curl -fsSL --retry 3 \
            "https://repo.download.nvidia.com/jetson/${repository}/dists/r${l4t_series}/main/binary-arm64/Packages" \
            -o "${index}"
    fi
    local metadata version filename checksum
    metadata=$(awk -v package="${package}" -v release="${l4t_version}-" '
        BEGIN { RS = ""; FS = "\n" }
        {
            name = version = filename = checksum = ""
            for (i = 1; i <= NF; ++i) {
                if ($i ~ /^Package: /) name = substr($i, 10)
                if ($i ~ /^Version: /) version = substr($i, 10)
                if ($i ~ /^Filename: /) filename = substr($i, 11)
                if ($i ~ /^SHA256: /) checksum = substr($i, 9)
            }
            if (name == package && index(version, release) == 1) {
                print version, filename, checksum
                exit
            }
        }' "${index}")
    read -r version filename checksum <<< "${metadata}"
    if [[ -z ${version} || -z ${filename} || -z ${checksum} ]]; then
        echo "${package} for L4T ${l4t_version} not found in ${repository}" >&2
        exit 1
    fi
    curl -fsSL --retry 3 "https://repo.download.nvidia.com/jetson/${repository}/${filename}" \
        -o "${download_dir}/${package}.deb"
    echo "${checksum}  ${download_dir}/${package}.deb" | sha256sum --check --status
    echo "${package}=${version} sha256=${checksum}" | tee -a "${manifest}"
}

fetch_package common nvidia-l4t-jetson-multimedia-api
dpkg-deb --extract "${download_dir}/nvidia-l4t-jetson-multimedia-api.deb" "${destination}"

# Keep the direct ELF dependencies so the same binary can load the host's matching libraries.
# Their transitive BSP dependencies are deliberately supplied only on the runtime device.
fetch_package "${bsp_repository}" nvidia-l4t-camera
dpkg-deb --fsys-tarfile "${download_dir}/nvidia-l4t-camera.deb" |
    tar -x --wildcards -C "${destination}" \
        './usr/lib/aarch64-linux-gnu/nvidia/libnvargus_socketclient.so*' \
        './usr/share/doc/nvidia-l4t-camera/*'
fetch_package "${bsp_repository}" nvidia-l4t-multimedia-utils
dpkg-deb --fsys-tarfile "${download_dir}/nvidia-l4t-multimedia-utils.deb" |
    tar -x --wildcards -C "${destination}" \
        './usr/lib/aarch64-linux-gnu/nvidia/libnvbufsurface.so*' \
        './usr/share/doc/nvidia-l4t-multimedia-utils/*'

test -f "${destination%/}/usr/src/jetson_multimedia_api/argus/include/Argus/Argus.h"
test -f "${destination%/}/usr/src/jetson_multimedia_api/include/nvbufsurface.h"
