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

"""Compact status display for HoloHub development environment."""

import json
import os
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
    get_sdk_version,
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
class ImageInfo:
    image: str
    created: str
    status: str


@dataclass
class GitInfo:
    branch: str
    commit: str
    dirty: bool
    modified_count: int


@dataclass
class BuildInfo:
    name: str
    status: str  # "OK" or "FAIL"
    last_modified: str


@dataclass
class FolderInfo:
    path: str
    size_mb: float


def collect_platform_info() -> PlatformInfo:
    arch = get_host_arch()
    gpu_type = get_host_gpu()
    gpu_name = get_gpu_name()
    cuda_version = get_default_cuda_version()
    sdk_path = Path(os.environ.get("HOLOHUB_DEFAULT_HSDK_DIR", "/opt/nvidia/holoscan"))
    holoscan_version = get_sdk_version(sdk_path)

    return PlatformInfo(
        arch=arch,
        gpu_type=gpu_type,
        gpu_name=gpu_name.split("\n")[0] if gpu_name else None,
        cuda_version=cuda_version,
        holoscan_version=holoscan_version,
    )


def collect_git_info(holohub_root: Path) -> Optional[GitInfo]:
    branch = run_info_command(["git", "-C", str(holohub_root), "branch", "--show-current"])
    commit = run_info_command(["git", "-C", str(holohub_root), "rev-parse", "--short", "HEAD"])
    porcelain = run_info_command(["git", "-C", str(holohub_root), "status", "--porcelain"])
    if branch is None or commit is None:
        return None
    modified = [line for line in (porcelain or "").splitlines() if line.strip()]
    return GitInfo(
        branch=branch or "(detached)",
        commit=commit,
        dirty=len(modified) > 0,
        modified_count=len(modified),
    )


def _dir_size_mb(path: Path) -> float:
    total = 0
    for root, _dirs, files in os.walk(str(path)):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                continue
    return total / (1024 * 1024)


def collect_folder_info(paths: List[Path]) -> List[FolderInfo]:
    seen: set[Path] = set()
    results: List[FolderInfo] = []
    for path in sorted(paths):
        resolved = path.resolve()
        if path.is_dir() and resolved not in seen:
            seen.add(resolved)
            results.append(FolderInfo(path=str(path), size_mb=_dir_size_mb(path)))
    return results


def collect_image_info() -> List[ImageInfo]:
    images = []
    image_prefix = os.environ.get("HOLOHUB_REPO_PREFIX", "holohub")
    prefixes = [image_prefix]
    docker_exe = os.environ.get("HOLOHUB_DOCKER_EXE", "docker")

    # Use image IDs from running containers for reliable comparison
    running_image_ids: set = set()
    ps_output = run_info_command([docker_exe, "ps", "--format", "{{.Image}}\t{{.ID}}"])
    if ps_output:
        for line in ps_output.split("\n"):
            parts = line.strip().split("\t")
            if parts:
                running_image_ids.add(parts[0].strip())

    images_output = run_info_command(
        [docker_exe, "images", "--format", "{{.ID}}\t{{.Repository}}:{{.Tag}}\t{{.CreatedSince}}"]
    )
    if not images_output:
        return images

    for line in images_output.split("\n"):
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        image_id, image_name, created = parts[0], parts[1], parts[2]
        if not any(prefix in image_name for prefix in prefixes):
            continue

        # Check if any running container uses this image (by ID or name)
        is_running = image_id in running_image_ids or image_name in running_image_ids
        status = "Running" if is_running else "Stopped"
        images.append(ImageInfo(image=image_name, created=created, status=status))
    return images


def collect_docker_disk_usage() -> Optional[str]:
    """Get total Docker disk usage summary in one call."""
    docker_exe = os.environ.get("HOLOHUB_DOCKER_EXE", "docker")
    output = run_info_command([docker_exe, "system", "df", "--format", "{{.Type}}\t{{.Size}}"])
    if not output:
        return None
    parts = {}
    for line in output.strip().split("\n"):
        cols = line.split("\t")
        if len(cols) >= 2:
            parts[cols[0].strip()] = cols[1].strip()
    if not parts:
        return None
    return ", ".join(f"{k}: {v}" for k, v in parts.items())


def _relative_time(mtime: float) -> str:
    elapsed = time.time() - mtime
    if elapsed < 60:
        return "just now"
    if elapsed < 3600:
        return f"{int(elapsed / 60)}m ago"
    if elapsed < 86400:
        return f"{int(elapsed / 3600)}h ago"
    return f"{int(elapsed / 86400)}d ago"


def collect_build_info(build_parent_dir: Path) -> List[BuildInfo]:
    builds = []
    if not build_parent_dir.is_dir():
        return builds

    try:
        entries = sorted(build_parent_dir.iterdir())
    except OSError:
        return builds

    for subdir in entries:
        if not subdir.is_dir() or subdir.name.startswith("."):
            continue
        if not (subdir / "CMakeCache.txt").exists():
            continue

        has_build_system = (subdir / "Makefile").exists() or (subdir / "build.ninja").exists()
        try:
            time_str = _relative_time(subdir.stat().st_mtime)
        except OSError:
            time_str = "unknown"

        builds.append(
            BuildInfo(
                name=subdir.name,
                status="OK" if has_build_system else "FAIL",
                last_modified=time_str,
            )
        )
    return builds


def _format_size(mb: float) -> str:
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB"


def format_status(
    platform: PlatformInfo,
    git: Optional[GitInfo],
    images: List[ImageInfo],
    builds: List[BuildInfo],
    build_folders: List[FolderInfo],
    data_folders: List[FolderInfo],
    docker_disk: Optional[str] = None,
) -> str:
    lines = []

    # Platform
    gpu = platform.gpu_name or "unknown"
    lines.append(
        f"Platform: {platform.arch} | {platform.gpu_type} | {gpu}"
        f" | CUDA {platform.cuda_version} | Holoscan {platform.holoscan_version}"
    )

    # Git
    if git:
        dirty = Color.yellow(f" ({git.modified_count} modified)") if git.dirty else ""
        lines.append(f"Git:      {git.branch} @ {git.commit}{dirty}")

    # Images
    if images:
        lines.append(f"\n{Color.white('Images:', bold=True)}")
        for img in images:
            color = Color.green if img.status == "Running" else Color.yellow
            lines.append(f"  {img.image:<50} {img.created:<20} {color(img.status)}")
    else:
        lines.append(f"\n{Color.white('Images:', bold=True)} (none)")

    # Docker disk summary
    if docker_disk:
        lines.append(f"\n{Color.white('Docker disk:', bold=True)} {docker_disk}")

    # Builds
    if builds:
        lines.append(f"\n{Color.white('Builds:', bold=True)}")
        for b in builds:
            s = Color.green("OK") if b.status == "OK" else Color.red("FAIL")
            lines.append(f"  {b.name:<30} {s}  {b.last_modified}")
    else:
        lines.append(f"\n{Color.white('Builds:', bold=True)} (none)")

    # Build folders
    if build_folders:
        total = sum(f.size_mb for f in build_folders)
        lines.append(f"\n{Color.white('Build folders:', bold=True)} {_format_size(total)} total")
        for f in build_folders:
            lines.append(f"  {f.path:<55} {_format_size(f.size_mb):>10}")

    # Data folders
    if data_folders:
        total = sum(f.size_mb for f in data_folders)
        lines.append(f"\n{Color.white('Data folders:', bold=True)} {_format_size(total)} total")
        for f in data_folders:
            lines.append(f"  {f.path:<55} {_format_size(f.size_mb):>10}")

    return "\n".join(lines)


def format_status_json(
    platform: PlatformInfo,
    git: Optional[GitInfo],
    images: List[ImageInfo],
    builds: List[BuildInfo],
    build_folders: List[FolderInfo],
    data_folders: List[FolderInfo],
    docker_disk: Optional[str] = None,
) -> str:
    data = {
        "platform": asdict(platform),
        "git": asdict(git) if git else None,
        "images": [asdict(img) for img in images],
        "builds": [asdict(b) for b in builds],
        "build_folders": [asdict(f) for f in build_folders],
        "data_folders": [asdict(f) for f in data_folders],
    }
    if docker_disk:
        data["docker_disk"] = docker_disk
    return json.dumps(data, indent=2)
