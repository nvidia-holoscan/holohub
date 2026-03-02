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
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from .util import (
    Color,
    get_default_cuda_version,
    get_gpu_name,
    get_holohub_root,
    is_running_in_docker,
    run_info_command,
)


@dataclass
class CheckResult:
    """Result of a single system info check"""

    status: str  # "OK", "WARN", "FAIL", "SKIP"
    name: str
    message: str
    fix_suggestion: Optional[str] = None
    details: Optional[str] = None


def check_gpu() -> CheckResult:
    """Check GPU availability and info"""
    gpu_name = get_gpu_name()
    if gpu_name is None:
        return CheckResult(
            status="FAIL",
            name="GPU",
            message="No NVIDIA GPU detected (nvidia-smi not available)",
            fix_suggestion="Install NVIDIA drivers: https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/",
        )

    # Get driver version
    driver_version = run_info_command(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )

    # Get compute capability
    compute_cap = run_info_command(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
    )

    # Get memory
    mem_total = run_info_command(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader"]
    )

    parts = []
    if gpu_name:
        # Take first GPU if multiple
        parts.append(gpu_name.split("\n")[0])
    if driver_version:
        parts.append(f"driver {driver_version.split(chr(10))[0]}")
    if compute_cap:
        parts.append(f"compute {compute_cap.split(chr(10))[0]}")
    if mem_total:
        parts.append(mem_total.split("\n")[0])

    return CheckResult(
        status="OK",
        name="GPU",
        message=", ".join(parts),
    )


def check_cuda() -> CheckResult:
    """Check CUDA toolkit availability"""
    nvcc = shutil.which("nvcc")
    if nvcc:
        nvcc_version = run_info_command(["nvcc", "--version"])
        if nvcc_version:
            match = re.search(r"release (\d+\.\d+)", nvcc_version)
            version_str = match.group(1) if match else "unknown"
            build_match = re.search(r"V(\d+\.\d+\.\d+)", nvcc_version)
            build_str = build_match.group(1) if build_match else ""
            msg = version_str
            if build_str:
                msg = f"{version_str} (nvcc {build_str})"
            return CheckResult(status="OK", name="CUDA", message=msg)

    # Fallback: check driver-based CUDA version
    cuda_ver = get_default_cuda_version()
    return CheckResult(
        status="WARN",
        name="CUDA",
        message=f"nvcc not found; driver supports CUDA {cuda_ver}",
        fix_suggestion="Install CUDA toolkit or use container mode",
    )


def check_docker() -> CheckResult:
    """Check Docker installation and runtime"""
    docker = shutil.which("docker")
    if not docker:
        return CheckResult(
            status="WARN",
            name="Docker",
            message="Docker not installed (container mode unavailable)",
            fix_suggestion="Install Docker: https://docs.docker.com/engine/install/",
        )

    # Check daemon running
    result = run_info_command(["docker", "info"])
    if result is None:
        return CheckResult(
            status="FAIL",
            name="Docker",
            message="Docker daemon not running",
            fix_suggestion="sudo systemctl start docker",
        )

    # Get Docker version
    docker_version = run_info_command(["docker", "--version"])
    version_str = ""
    if docker_version:
        match = re.search(r"(\d+\.\d+\.\d+)", docker_version)
        version_str = match.group(1) if match else docker_version.strip()

    # Check nvidia-ctk
    ctk = shutil.which("nvidia-ctk")
    ctk_version = ""
    if ctk:
        ctk_out = run_info_command(["nvidia-ctk", "--version"])
        if ctk_out:
            match = re.search(r"(\d+\.\d+\.\d+)", ctk_out)
            ctk_version = match.group(1) if match else ""

    # Check BuildKit
    buildkit = os.environ.get("DOCKER_BUILDKIT", "")
    buildkit_str = "BuildKit" if buildkit == "1" else ""

    parts = [version_str]
    if ctk_version:
        parts.append(f"nvidia-ctk {ctk_version}")
    if buildkit_str:
        parts.append(buildkit_str)

    status = "OK"
    fix = None
    if not ctk:
        status = "WARN"
        fix = "Install NVIDIA Container Toolkit for GPU support in containers"

    return CheckResult(
        status=status,
        name="Docker",
        message=" + ".join(p for p in parts if p),
        fix_suggestion=fix,
    )


def check_python() -> CheckResult:
    """Check Python version"""
    ver = sys.version_info
    version_str = f"{ver.major}.{ver.minor}.{ver.micro}"
    if ver >= (3, 10):
        return CheckResult(
            status="OK",
            name="Python",
            message=f"{version_str} (>= 3.10 required)",
        )
    return CheckResult(
        status="FAIL",
        name="Python",
        message=f"{version_str} (>= 3.10 required)",
        fix_suggestion="Install Python 3.10 or newer",
    )


def check_holoscan() -> CheckResult:
    """Check Holoscan SDK availability"""
    sdk_dir = os.environ.get("HOLOHUB_DEFAULT_HSDK_DIR", "/opt/nvidia/holoscan")
    sdk_path = Path(sdk_dir)

    if not sdk_path.exists():
        return CheckResult(
            status="WARN",
            name="Holoscan",
            message=f"SDK not found at {sdk_dir}",
            fix_suggestion="Install Holoscan SDK or use container mode",
        )

    # Try to get version
    version_file = sdk_path / "VERSION"
    version = "unknown"
    if version_file.exists():
        version = version_file.read_text().strip()
    else:
        # Try cmake config
        cmake_config = sdk_path / "lib" / "cmake" / "holoscan" / "holoscan-config-version.cmake"
        if cmake_config.exists():
            content = cmake_config.read_text()
            match = re.search(r'PACKAGE_VERSION\s+"([^"]+)"', content)
            if match:
                version = match.group(1)

    return CheckResult(
        status="OK",
        name="Holoscan",
        message=f"SDK {version} at {sdk_dir}",
    )


def _find_mount_point(path: Path) -> str:
    """Walk up from path to find its mount point."""
    p = path.resolve()
    while not p.is_mount():
        p = p.parent
    return str(p)


def check_disk() -> CheckResult:
    """Check free disk space on the filesystem containing the build directory"""
    holohub_root = get_holohub_root()
    build_dir = Path(os.environ.get("HOLOHUB_BUILD_PARENT_DIR", holohub_root / "build"))

    # Check the filesystem where builds go
    check_path = build_dir if build_dir.exists() else holohub_root
    try:
        stat = os.statvfs(str(check_path))
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        mount_point = _find_mount_point(check_path)

        if free_gb < 5:
            return CheckResult(
                status="FAIL",
                name="Disk",
                message=f"{free_gb:.0f}GB free on {mount_point} (build dir: {check_path})",
                fix_suggestion="Free disk space: ./holohub clear-cache && docker system prune",
            )
        elif free_gb < 20:
            return CheckResult(
                status="WARN",
                name="Disk",
                message=f"{free_gb:.0f}GB free on {mount_point} (build dir: {check_path}, < 20GB)",
                fix_suggestion="Consider freeing space: ./holohub clear-cache",
            )
        return CheckResult(
            status="OK",
            name="Disk",
            message=f"{free_gb:.0f}GB free on {mount_point}",
        )
    except OSError:
        return CheckResult(
            status="WARN",
            name="Disk",
            message="Could not check disk space",
        )


def check_cli() -> CheckResult:
    """Check CLI version and commit"""
    holohub_root = get_holohub_root()
    commit_hash = run_info_command(["git", "rev-parse", "--short=7", "HEAD"], cwd=str(holohub_root))

    # Check for external codebase commit hash file
    cli_commit_file = holohub_root / ".cli_commit_hash"
    if cli_commit_file.exists():
        cli_hash = cli_commit_file.read_text().strip()
        return CheckResult(
            status="OK",
            name="CLI",
            message=f"holohub (cli commit {cli_hash})",
        )

    msg = "holohub"
    if commit_hash:
        msg += f" (commit {commit_hash})"

    return CheckResult(status="OK", name="CLI", message=msg)


def check_container() -> CheckResult:
    """Check if running inside a container"""
    if is_running_in_docker():
        return CheckResult(
            status="OK",
            name="Container",
            message="Running inside Docker container",
        )
    return CheckResult(
        status="OK",
        name="Container",
        message="Running on host (not in container)",
    )


def check_display() -> CheckResult:
    """Check X11/display availability for visualization apps"""
    display = os.environ.get("DISPLAY")
    x11_socket = os.path.exists("/tmp/.X11-unix")

    if display and x11_socket:
        return CheckResult(
            status="OK",
            name="Display",
            message=f"DISPLAY={display}, X11 socket present",
        )

    if display and not x11_socket:
        return CheckResult(
            status="WARN",
            name="Display",
            message=f"DISPLAY={display} but /tmp/.X11-unix missing",
            fix_suggestion="X11 socket not found — visualization apps may fail. "
            "If using SSH, try: ssh -X or set up X11 forwarding",
        )

    if is_running_in_docker():
        return CheckResult(
            status="WARN",
            name="Display",
            message="DISPLAY not set (in container)",
            fix_suggestion="Pass display to container: "
            "--docker-opts '-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix'",
        )

    return CheckResult(
        status="WARN",
        name="Display",
        message="DISPLAY not set — visualization apps will fail",
        fix_suggestion="export DISPLAY=:0  (or use SSH X11 forwarding: ssh -X)",
    )


def check_devices() -> CheckResult:
    """Scan /dev for known device patterns"""
    devices_found = []

    # V4L2 video devices
    v4l2_devs = sorted(glob.glob("/dev/video[0-9]*"))
    if v4l2_devs:
        devices_found.append(f"V4L2: {', '.join(v4l2_devs)}")

    # AJA devices
    aja_devs = sorted(glob.glob("/dev/ajantv2[0-9]*"))
    if aja_devs:
        devices_found.append(f"AJA: {', '.join(aja_devs)}")

    # Deltacast devices
    delta_devs = []
    for i in range(4):
        for prefix in ["/dev/delta-x380", "/dev/delta-x370", "/dev/delta-x350"]:
            dev = f"{prefix}{i}"
            if os.path.exists(dev):
                delta_devs.append(dev)
    if delta_devs:
        devices_found.append(f"Deltacast: {', '.join(delta_devs)}")

    # ConnectX / InfiniBand devices
    ib_devs = sorted(glob.glob("/dev/infiniband/uverbs[0-9]*"))
    if ib_devs:
        devices_found.append(f"ConnectX: {', '.join(ib_devs)}")

    # Audio devices
    if os.path.isdir("/dev/snd"):
        audio_count = len(glob.glob("/dev/snd/pcm*"))
        if audio_count:
            devices_found.append(f"Audio: {audio_count} PCM device(s)")

    # iGPU specific devices
    if os.path.exists("/dev/nvgpu/igpu0/nvsched"):
        devices_found.append("iGPU: /dev/nvgpu/igpu0/nvsched")

    if not devices_found:
        return CheckResult(
            status="SKIP",
            name="Devices",
            message="No specialized devices detected",
        )

    return CheckResult(
        status="OK",
        name="Devices",
        message="; ".join(devices_found),
    )


def run_all_checks(verbose: bool = False) -> List[CheckResult]:
    """Run all system info checks and return results"""
    checks = [
        check_gpu,
        check_cuda,
        check_docker,
        check_python,
        check_holoscan,
        check_disk,
        check_cli,
        check_container,
        check_display,
        check_devices,
    ]

    results = []
    for check_fn in checks:
        try:
            result = check_fn()
            results.append(result)
        except Exception as e:
            results.append(
                CheckResult(
                    status="FAIL",
                    name=check_fn.__name__.replace("check_", "").title(),
                    message=f"Check failed: {e}",
                )
            )
    return results


def format_results(results: List[CheckResult], elapsed: float, verbose: bool = False) -> str:
    """Format check results for terminal display"""
    status_formats = {
        "OK": Color.green("[OK]  "),
        "WARN": Color.yellow("[WARN]"),
        "FAIL": Color.red("[FAIL]"),
        "SKIP": "[--]  ",
    }

    lines = [f"\nSystem Info ({elapsed:.1f}s)\n"]

    fail_count = 0
    warn_count = 0
    skip_count = 0

    for r in results:
        prefix = status_formats.get(r.status, f"[{r.status}]")
        name_padded = f"{r.name:<12}"
        lines.append(f"  {prefix} {name_padded} {r.message}")

        if verbose and r.details:
            for detail_line in r.details.split("\n"):
                lines.append(f"                     {detail_line}")

        if r.status == "FAIL":
            fail_count += 1
            if r.fix_suggestion:
                lines.append(f"                     Fix: {Color.yellow(r.fix_suggestion)}")
        elif r.status == "WARN":
            warn_count += 1
            if r.fix_suggestion and verbose:
                lines.append(f"                     Hint: {Color.yellow(r.fix_suggestion)}")
        elif r.status == "SKIP":
            skip_count += 1

    lines.append("")

    if fail_count:
        lines.append(Color.red(f"{fail_count} check(s) failed. See suggestions above."))
    elif warn_count:
        lines.append(
            Color.yellow(f"All checks passed with {warn_count} warning(s).")
            + (f" {skip_count} optional check(s) skipped." if skip_count else "")
        )
    else:
        msg = Color.green("All checks passed.")
        if skip_count:
            msg += f" {skip_count} optional device(s) not detected."
        lines.append(msg)

    return "\n".join(lines)


def format_results_json(results: List[CheckResult], elapsed: float) -> str:
    """Format check results as JSON"""
    data = {
        "elapsed_seconds": round(elapsed, 2),
        "checks": [asdict(r) for r in results],
        "summary": {
            "ok": sum(1 for r in results if r.status == "OK"),
            "warn": sum(1 for r in results if r.status == "WARN"),
            "fail": sum(1 for r in results if r.status == "FAIL"),
            "skip": sum(1 for r in results if r.status == "SKIP"),
        },
    }
    return json.dumps(data, indent=2)
