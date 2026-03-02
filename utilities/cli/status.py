#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import glob
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from .util import (
    Color,
    get_default_cuda_version,
    get_gpu_name,
    get_host_arch,
    get_host_gpu,
    run_info_command,
)


@dataclass
class PlatformInfo:
    arch: str
    gpu_type: str
    gpu_name: Optional[str]
    cuda_version: str
    holoscan_version: str


@dataclass
class ContainerInfo:
    name: str
    image: str
    size: str
    created: str
    status: str  # "Running" or "Stopped" or similar


@dataclass
class BuildInfo:
    name: str
    status: str  # "OK" or "FAIL"
    last_modified: str
    path: str


@dataclass
class DeviceInfo:
    category: str
    devices: List[str]


@dataclass
class CacheDir:
    path: str
    size_mb: float


@dataclass
class CacheInfo:
    build: List[CacheDir]
    data: List[CacheDir]
    install: List[CacheDir]
    total_mb: float


def collect_platform_info() -> PlatformInfo:
    """Collect platform information using existing util functions"""
    arch = get_host_arch()
    gpu_type = get_host_gpu()
    gpu_name = get_gpu_name()
    cuda_version = get_default_cuda_version()

    # Try to find Holoscan SDK version
    sdk_dir = os.environ.get("HOLOHUB_DEFAULT_HSDK_DIR", "/opt/nvidia/holoscan")
    holoscan_version = "not found"
    version_file = Path(sdk_dir) / "VERSION"
    if version_file.exists():
        holoscan_version = version_file.read_text().strip()
    else:
        cmake_config = (
            Path(sdk_dir) / "lib" / "cmake" / "holoscan" / "holoscan-config-version.cmake"
        )
        if cmake_config.exists():
            content = cmake_config.read_text()
            match = re.search(r'PACKAGE_VERSION\s+"([^"]+)"', content)
            if match:
                holoscan_version = match.group(1)

    return PlatformInfo(
        arch=arch,
        gpu_type=gpu_type,
        gpu_name=gpu_name.split("\n")[0] if gpu_name else None,
        cuda_version=cuda_version,
        holoscan_version=holoscan_version,
    )


def collect_container_info() -> List[ContainerInfo]:
    """Collect information about holohub-related Docker containers and images"""
    containers = []

    # Get the container prefix for filtering
    container_prefix = os.environ.get("HOLOHUB_REPO_PREFIX", "holohub")
    prefixes = [container_prefix]
    # Also check common external codebase prefixes
    for extra in ["i4h_build", "isaac"]:
        if extra != container_prefix:
            prefixes.append(extra)

    # List holohub-related Docker images
    images_output = run_info_command(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}"]
    )
    if images_output:
        for line in images_output.split("\n"):
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            image_name, size, created = parts[0], parts[1], parts[2]
            if not any(prefix in image_name for prefix in prefixes):
                continue

            # Check if there's a running container for this image
            status = "Stopped"
            ps_output = run_info_command(
                [
                    "docker",
                    "ps",
                    "--format",
                    "{{.Names}}\t{{.Status}}",
                    "--filter",
                    f"ancestor={image_name}",
                ]
            )
            container_name = image_name
            if ps_output and ps_output.strip():
                first_line = ps_output.strip().split("\n")[0]
                ps_parts = first_line.split("\t")
                if len(ps_parts) >= 2:
                    container_name = ps_parts[0]
                    status = f"Running ({ps_parts[1]})"

            containers.append(
                ContainerInfo(
                    name=container_name,
                    image=image_name,
                    size=size,
                    created=f"Built {created}",
                    status=status,
                )
            )

    return containers


def collect_build_info(build_parent_dir: Path) -> List[BuildInfo]:
    """Scan build directory for build subdirectories and their status"""
    builds = []

    if not build_parent_dir.exists():
        return builds

    for subdir in sorted(build_parent_dir.iterdir()):
        if not subdir.is_dir():
            continue
        # Skip hidden directories
        if subdir.name.startswith("."):
            continue

        # Check if it has CMakeCache.txt (indicates a configured build)
        has_cmake_cache = (subdir / "CMakeCache.txt").exists()
        if not has_cmake_cache:
            continue

        # Check for error log
        error_log = subdir / "CMakeFiles" / "CMakeError.log"
        has_errors = error_log.exists() and error_log.stat().st_size > 0

        status = "FAIL" if has_errors else "OK"

        # Get last modification time
        try:
            mtime = subdir.stat().st_mtime
            elapsed = time.time() - mtime
            if elapsed < 60:
                time_str = "just now"
            elif elapsed < 3600:
                time_str = f"{int(elapsed / 60)} min ago"
            elif elapsed < 86400:
                time_str = f"{int(elapsed / 3600)}h ago"
            else:
                time_str = f"{int(elapsed / 86400)}d ago"
        except OSError:
            time_str = "unknown"

        builds.append(
            BuildInfo(
                name=subdir.name,
                status=status,
                last_modified=time_str,
                path=str(subdir),
            )
        )

    return builds


def collect_device_info() -> List[DeviceInfo]:
    """Collect information about detected hardware devices"""
    devices = []

    # V4L2 video devices
    v4l2_devs = sorted(glob.glob("/dev/video[0-9]*"))
    if v4l2_devs:
        devices.append(DeviceInfo(category="V4L2", devices=v4l2_devs))

    # AJA devices
    aja_devs = sorted(glob.glob("/dev/ajantv2[0-9]*"))
    if aja_devs:
        devices.append(DeviceInfo(category="AJA", devices=aja_devs))

    # Deltacast devices
    delta_devs = []
    for i in range(4):
        for prefix in ["/dev/delta-x380", "/dev/delta-x370", "/dev/delta-x350"]:
            dev = f"{prefix}{i}"
            if os.path.exists(dev):
                delta_devs.append(dev)
    if delta_devs:
        devices.append(DeviceInfo(category="Deltacast", devices=delta_devs))

    # ConnectX / InfiniBand devices
    ib_devs = sorted(glob.glob("/dev/infiniband/uverbs[0-9]*"))
    if ib_devs:
        devices.append(DeviceInfo(category="ConnectX", devices=ib_devs))

    return devices


def _dir_size_mb(path: Path) -> float:
    """Get total size of a directory in MB (non-recursive stat, fast approximation)."""
    try:
        total = 0
        for root, dirs, files in os.walk(str(path)):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    continue
        return total / (1024 * 1024)
    except OSError:
        return 0.0


def _collect_dirs(
    holohub_root: Path, patterns: List[str], default_dir: Optional[Path] = None
) -> List[CacheDir]:
    """Collect existing cache directories matching patterns (mirrors _collect_cache_dirs in holohub.py)."""
    seen: set[Path] = set()
    results: List[CacheDir] = []

    if default_dir is not None and default_dir.exists() and default_dir.is_dir():
        seen.add(default_dir)
        results.append(CacheDir(path=str(default_dir), size_mb=_dir_size_mb(default_dir)))

    for pattern in patterns:
        for path in holohub_root.glob(pattern):
            if path.is_dir() and path not in seen:
                seen.add(path)
                results.append(CacheDir(path=str(path), size_mb=_dir_size_mb(path)))

    return results


def collect_cache_info(holohub_root: Path, build_parent_dir: Path, data_dir: Path) -> CacheInfo:
    """Collect info about directories that ./holohub clear-cache would remove."""
    build = _collect_dirs(holohub_root, ["build", "build-*"], build_parent_dir)
    data = _collect_dirs(holohub_root, ["data", "data-*"], data_dir)
    install = _collect_dirs(holohub_root, ["install", "install-*"])
    total = sum(d.size_mb for d in build + data + install)
    return CacheInfo(build=build, data=data, install=install, total_mb=total)


def _format_size(mb: float) -> str:
    """Format MB as human-readable size."""
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB"


def format_status(
    platform: PlatformInfo,
    containers: List[ContainerInfo],
    builds: List[BuildInfo],
    devices: List[DeviceInfo],
    cache: Optional[CacheInfo] = None,
) -> str:
    """Format status information for terminal display"""
    lines = []

    # Platform line
    gpu_str = platform.gpu_name or "unknown"
    lines.append(
        f"Platform: {platform.arch} / {platform.gpu_type} / {gpu_str} "
        f"/ CUDA {platform.cuda_version} / Holoscan {platform.holoscan_version}"
    )

    # Containers
    if containers:
        lines.append(f"\n{Color.white('Containers:', bold=True)}")
        for c in containers:
            status_color = Color.green if "Running" in c.status else Color.yellow
            lines.append(f"  {c.image:<40} {c.created:<20} ({c.size})   {status_color(c.status)}")
    else:
        lines.append(f"\n{Color.white('Containers:', bold=True)} (none found)")

    # Builds
    if builds:
        lines.append(f"\n{Color.white('Builds (local):', bold=True)}")
        for b in builds:
            status_str = Color.green("OK") if b.status == "OK" else Color.red("FAIL")
            lines.append(f"  {b.name:<30} {status_str}    {b.last_modified:<15} {b.path}")
    else:
        lines.append(f"\n{Color.white('Builds (local):', bold=True)} (none found)")

    # Devices
    if devices:
        lines.append(f"\n{Color.white('Devices:', bold=True)}")
        for d in devices:
            lines.append(f"  {d.category:<12} {', '.join(d.devices)}")
    else:
        lines.append(f"\n{Color.white('Devices:', bold=True)} (none detected)")

    # Cache (clear-cache targets)
    if cache:
        all_dirs = cache.build + cache.data + cache.install
        if all_dirs:
            lines.append(
                f"\n{Color.white('Cache (clear-cache):', bold=True)}"
                f"  {_format_size(cache.total_mb)} total"
            )
            for label, dirs in [
                ("build", cache.build),
                ("data", cache.data),
                ("install", cache.install),
            ]:
                for d in dirs:
                    lines.append(f"  {d.path:<55} {_format_size(d.size_mb):>10}")

    return "\n".join(lines)


def format_status_short(
    platform: PlatformInfo,
    containers: List[ContainerInfo],
    builds: List[BuildInfo],
    devices: List[DeviceInfo],
    cache: Optional[CacheInfo] = None,
) -> str:
    """Format a single-line status summary"""
    gpu_str = platform.gpu_name.split("\n")[0] if platform.gpu_name else "no GPU"
    running = sum(1 for c in containers if "Running" in c.status)
    total_containers = len(containers)
    ok_builds = sum(1 for b in builds if b.status == "OK")
    fail_builds = sum(1 for b in builds if b.status == "FAIL")
    total_devices = sum(len(d.devices) for d in devices)

    parts = [
        f"{platform.arch}/{gpu_str}",
        f"CUDA {platform.cuda_version}",
        f"Holoscan {platform.holoscan_version}",
        f"containers: {running}/{total_containers} running",
        f"builds: {ok_builds} ok" + (f"/{fail_builds} fail" if fail_builds else ""),
        f"devices: {total_devices}",
    ]
    if cache:
        parts.append(f"cache: {_format_size(cache.total_mb)}")
    return " | ".join(parts)


def format_status_json(
    platform: PlatformInfo,
    containers: List[ContainerInfo],
    builds: List[BuildInfo],
    devices: List[DeviceInfo],
    cache: Optional[CacheInfo] = None,
) -> str:
    """Format status as JSON"""
    data = {
        "platform": asdict(platform),
        "containers": [asdict(c) for c in containers],
        "builds": [asdict(b) for b in builds],
        "devices": [asdict(d) for d in devices],
    }
    if cache:
        data["cache"] = asdict(cache)
    return json.dumps(data, indent=2)
