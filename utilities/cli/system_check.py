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
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from .util import (
    Color,
    cuda_major_from_driver,
    find_hsdk_build_rel_dir,
    get_cuda_runtime_version,
    get_git_short_sha,
    get_gpu_name,
    get_holohub_root,
    get_sdk_version,
    is_running_in_docker,
    is_valid_sdk_installation,
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
    """Check GPU availability and info (supports multi-GPU: iGPU + dGPU)"""
    gpu_name = get_gpu_name()
    if gpu_name is None:
        return CheckResult(
            status="FAIL",
            name="GPU",
            message="No NVIDIA GPU detected (nvidia-smi not available)",
            fix_suggestion="Install NVIDIA drivers: https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/",
        )

    # Query per-GPU info (one line per GPU)
    raw = run_info_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,compute_cap,memory.total",
            "--format=csv,noheader",
        ]
    )
    if not raw:
        return CheckResult(status="OK", name="GPU", message=gpu_name.split("\n")[0])

    gpu_lines = [line.strip() for line in raw.strip().split("\n") if line.strip()]
    gpu_count = len(gpu_lines)

    # Format each GPU
    gpu_descs = []
    driver = None
    for line in gpu_lines:
        fields = [f.strip() for f in line.split(",")]
        if len(fields) >= 5:
            idx, name, drv, cc, mem = fields[0], fields[1], fields[2], fields[3], fields[4]
            driver = drv
            gpu_descs.append(f"[{idx}] {name} (compute {cc}, {mem})")
        else:
            gpu_descs.append(line)

    # Note CUDA_VISIBLE_DEVICES if set
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    details_parts = []
    if driver:
        details_parts.append(f"driver {driver}")
    if cuda_vis is not None:
        details_parts.append(f"CUDA_VISIBLE_DEVICES={cuda_vis}")

    if gpu_count == 1:
        msg = gpu_descs[0]
        if details_parts:
            msg += f" ({', '.join(details_parts)})"
        return CheckResult(status="OK", name="GPU", message=msg)

    # Multi-GPU
    header = f"{gpu_count} GPUs detected"
    if details_parts:
        header += f" ({', '.join(details_parts)})"
    detail_lines = "\n".join(f"  {d}" for d in gpu_descs)

    return CheckResult(
        status="OK",
        name="GPU",
        message=header,
        details=detail_lines,
    )


def _get_driver_cuda_version() -> str:
    """Get CUDA version from driver without printing warnings (silent alternative to
    get_default_cuda_version for use in structured output like --json)."""
    driver_version = run_info_command(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    if not driver_version:
        return "unknown"
    first_line = driver_version.splitlines()[0].strip()
    return cuda_major_from_driver(first_line) or "unknown"


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
            msg = f"{version_str} (nvcc {build_str})" if build_str else version_str
            return CheckResult(status="OK", name="CUDA", message=msg)
        return CheckResult(
            status="WARN",
            name="CUDA",
            message=f"nvcc found at {nvcc} but could not read version",
            fix_suggestion="Check that nvcc is not corrupted: nvcc --version",
        )

    # Fallback: check driver-based CUDA version and runtime package
    # Derive CUDA version silently (get_default_cuda_version() prints warnings to stdout
    # which would corrupt --json output on CPU-only systems)
    cuda_ver = _get_driver_cuda_version()
    runtime_ver = get_cuda_runtime_version()
    if runtime_ver:
        return CheckResult(
            status="OK",
            name="CUDA",
            message=f"runtime {runtime_ver} (nvcc not in PATH; driver supports CUDA {cuda_ver})",
        )
    return CheckResult(
        status="WARN",
        name="CUDA",
        message=f"nvcc not found; driver supports CUDA {cuda_ver}",
        fix_suggestion="Install CUDA toolkit or use container mode",
    )


def check_docker() -> CheckResult:
    """Check Docker installation and runtime"""
    docker_exe = os.environ.get("HOLOHUB_DOCKER_EXE", "docker")
    docker = shutil.which(docker_exe)
    if not docker:
        if is_running_in_docker():
            return CheckResult(
                status="SKIP", name="Docker", message="Not applicable (inside container)"
            )
        return CheckResult(
            status="WARN",
            name="Docker",
            message=f"{docker_exe} not installed (container mode unavailable)",
            fix_suggestion="Install Docker: https://docs.docker.com/engine/install/",
        )

    try:
        proc = subprocess.run(
            [docker_exe, "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            if "permission denied" in stderr.lower() or "connect:" in stderr.lower():
                return CheckResult(
                    status="FAIL",
                    name="Docker",
                    message="Docker permission denied",
                    fix_suggestion="Add user to docker group: sudo usermod -aG docker $USER && newgrp docker",
                )
            return CheckResult(
                status="FAIL",
                name="Docker",
                message="Docker daemon not running",
                fix_suggestion="sudo systemctl start docker",
            )
    except subprocess.TimeoutExpired:
        return CheckResult(
            status="FAIL",
            name="Docker",
            message="Docker daemon not responding (timed out after 10s)",
            fix_suggestion="Check Docker status: sudo systemctl status docker",
        )
    except Exception:
        return CheckResult(
            status="FAIL",
            name="Docker",
            message="Docker daemon not running",
            fix_suggestion="sudo systemctl start docker",
        )

    docker_version = run_info_command([docker_exe, "--version"])
    version_str = ""
    if docker_version:
        match = re.search(r"(\d+\.\d+\.\d+)", docker_version)
        version_str = match.group(1) if match else docker_version.strip()

    ctk = shutil.which("nvidia-ctk")
    ctk_version = ""
    if ctk:
        ctk_out = run_info_command(["nvidia-ctk", "--version"])
        if ctk_out:
            match = re.search(r"(\d+\.\d+\.\d+)", ctk_out)
            ctk_version = match.group(1) if match else ""

    parts = [version_str or "installed (version unknown)"]
    if ctk_version:
        parts.append(f"nvidia-ctk {ctk_version}")

    status = "OK"
    fix = None
    if not ctk:
        status = "WARN"
        fix = "Install NVIDIA Container Toolkit for GPU support in containers"

    return CheckResult(status=status, name="Docker", message=" + ".join(parts), fix_suggestion=fix)


def check_holoscan() -> CheckResult:
    """Check Holoscan SDK availability"""
    sdk_dir = os.environ.get("HOLOHUB_DEFAULT_HSDK_DIR", "/opt/nvidia/holoscan")
    sdk_path = Path(sdk_dir)
    if sdk_path.exists() and is_valid_sdk_installation(sdk_path):
        version = get_sdk_version(sdk_path)
        return CheckResult(status="OK", name="Holoscan", message=f"SDK {version} at {sdk_dir}")

    sdk_root = os.environ.get("HOLOSCAN_SDK_ROOT")
    if sdk_root:
        root_path = Path(sdk_root)
        if root_path.exists() and is_valid_sdk_installation(root_path):
            version = get_sdk_version(root_path)
            return CheckResult(status="OK", name="Holoscan", message=f"SDK {version} at {sdk_root}")
        if root_path.exists():
            resolved = find_hsdk_build_rel_dir(root_path)
            resolved_path = (
                root_path / resolved if not Path(resolved).is_absolute() else Path(resolved)
            )
            if resolved_path.exists() and is_valid_sdk_installation(resolved_path):
                version = get_sdk_version(resolved_path)
                return CheckResult(
                    status="OK",
                    name="Holoscan",
                    message=f"SDK {version} at {resolved_path} (via HOLOSCAN_SDK_ROOT)",
                )

    searched = sdk_dir
    if sdk_root:
        searched += f", HOLOSCAN_SDK_ROOT={sdk_root}"
    return CheckResult(
        status="WARN",
        name="Holoscan",
        message=f"SDK not found (searched: {searched})",
        fix_suggestion="Install Holoscan SDK or use container mode",
    )


def check_holoscan_python() -> CheckResult:
    """Check if the Holoscan Python package is importable and report its location.

    Runs ``python3 -c "import holoscan"`` in a subprocess so a failed import
    (e.g. missing libcudart) does not crash the CLI itself.
    """
    snippet = "import holoscan; " "print(holoscan.__version__); " "print(holoscan.__file__)"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            status="WARN",
            name="Holoscan SDK Python",
            message="import holoscan timed out (>15s)",
        )

    if proc.returncode != 0:
        err = proc.stderr.strip().splitlines()
        short_err = err[-1] if err else "unknown error"
        return CheckResult(
            status="WARN",
            name="Holoscan SDK Python",
            message=f"import holoscan failed ({short_err})",
        )

    lines = proc.stdout.strip().splitlines()
    version = lines[0].strip() if len(lines) > 0 else "unknown"
    location = lines[1].strip() if len(lines) > 1 else "unknown"
    pkg_dir = str(Path(location).parent) if location != "unknown" else "unknown"
    return CheckResult(
        status="OK",
        name="Holoscan SDK Python",
        message=f"{version} ({pkg_dir})",
    )


def check_disk() -> CheckResult:
    """Check free disk space on the filesystem containing the build directory"""
    holohub_root = get_holohub_root()
    build_dir = Path(os.environ.get("HOLOHUB_BUILD_PARENT_DIR", holohub_root / "build"))
    check_path = build_dir if build_dir.exists() else holohub_root

    try:
        stat = os.statvfs(str(check_path))
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        mount_point = check_path.resolve()
        while not mount_point.is_mount():
            mount_point = mount_point.parent

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
            status="OK", name="Disk", message=f"{free_gb:.0f}GB free on {mount_point}"
        )
    except OSError:
        return CheckResult(status="WARN", name="Disk", message="Could not check disk space")


def check_cli() -> CheckResult:
    """Check CLI version and commit"""
    holohub_root = get_holohub_root()

    cli_commit_file = holohub_root / ".cli_commit_hash"
    if cli_commit_file.exists():
        cli_hash = cli_commit_file.read_text().strip()
        return CheckResult(status="OK", name="CLI", message=f"holohub (cli commit {cli_hash})")

    commit_hash = get_git_short_sha(length=7)
    msg = f"holohub (commit {commit_hash})"
    return CheckResult(status="OK", name="CLI", message=msg)


def check_container() -> CheckResult:
    """Check if running inside a container"""
    if is_running_in_docker():
        return CheckResult(status="OK", name="Container", message="Running inside Docker container")
    return CheckResult(status="OK", name="Container", message="Running on host (not in container)")


def check_display() -> CheckResult:
    """Check X11/display availability for visualization apps"""
    display = os.environ.get("DISPLAY")

    x11_socket = False
    if display:
        try:
            screen = display.split(":")[1].split(".")[0]
            x11_socket = os.path.exists(f"/tmp/.X11-unix/X{screen}")
        except (IndexError, ValueError):
            x11_socket = os.path.exists("/tmp/.X11-unix")

    if display and x11_socket:
        return CheckResult(
            status="OK", name="Display", message=f"DISPLAY={display}, X11 socket present"
        )
    if display and not x11_socket:
        return CheckResult(
            status="WARN",
            name="Display",
            message=f"DISPLAY={display} but X11 socket not found",
            fix_suggestion="X11 socket not found - visualization apps may fail. "
            "If using SSH, try: ssh -X or set up X11 forwarding",
        )
    if is_running_in_docker():
        return CheckResult(
            status="SKIP",
            name="Display",
            message="DISPLAY not set (pass -e DISPLAY for visualization apps)",
        )
    return CheckResult(
        status="WARN",
        name="Display",
        message="DISPLAY not set - visualization apps will fail",
        fix_suggestion="export DISPLAY=:0  (or use SSH X11 forwarding: ssh -X)",
    )


def check_devices() -> CheckResult:
    """Scan /dev for known device patterns"""
    devices_found = []

    v4l2_devs = sorted(glob.glob("/dev/video[0-9]*"))
    if v4l2_devs:
        devices_found.append(f"V4L2: {', '.join(v4l2_devs)}")

    aja_devs = sorted(glob.glob("/dev/ajantv2[0-9]*"))
    if aja_devs:
        devices_found.append(f"AJA: {', '.join(aja_devs)}")

    delta_devs = []
    for i in range(4):
        for prefix in ["/dev/delta-x380", "/dev/delta-x370", "/dev/delta-x350"]:
            dev = f"{prefix}{i}"
            if os.path.exists(dev):
                delta_devs.append(dev)
    if delta_devs:
        devices_found.append(f"Deltacast: {', '.join(delta_devs)}")

    ib_devs = sorted(glob.glob("/dev/infiniband/uverbs[0-9]*"))
    if ib_devs:
        devices_found.append(f"ConnectX: {', '.join(ib_devs)}")

    if os.path.isdir("/dev/snd"):
        audio_count = len(glob.glob("/dev/snd/pcm*"))
        if audio_count:
            devices_found.append(f"Audio: {audio_count} PCM device(s)")

    if os.path.exists("/dev/nvgpu/igpu0/nvsched"):
        devices_found.append("iGPU: /dev/nvgpu/igpu0/nvsched")

    if not devices_found:
        return CheckResult(status="SKIP", name="Devices", message="No specialized devices detected")

    return CheckResult(status="OK", name="Devices", message="; ".join(devices_found))


def run_all_checks() -> List[CheckResult]:
    """Run all system info checks and return results"""
    checks = [
        ("GPU", check_gpu),
        ("CUDA", check_cuda),
        ("Docker", check_docker),
        ("Holoscan", check_holoscan),
        ("Holoscan SDK Python", check_holoscan_python),
        ("Disk", check_disk),
        ("CLI", check_cli),
        ("Container", check_container),
        ("Display", check_display),
        ("Devices", check_devices),
    ]

    results = []
    for canonical_name, check_fn in checks:
        try:
            result = check_fn()
            results.append(result)
        except Exception as e:
            results.append(
                CheckResult(
                    status="FAIL",
                    name=canonical_name,
                    message=f"Check failed: {e}",
                )
            )
    return results


def format_results(results: List[CheckResult], elapsed: float) -> str:
    """Format check results for terminal display"""
    status_formats = {
        "OK": Color.green("[OK]  "),
        "WARN": Color.yellow("[WARN]"),
        "FAIL": Color.red("[FAIL]"),
        "SKIP": "[--]  ",
    }

    lines = [f"\nSystem Info Check ({elapsed:.1f}s)\n"]

    fail_count = 0
    warn_count = 0

    for r in results:
        prefix = status_formats.get(r.status, f"[{r.status}]")
        name_padded = f"{r.name:<20}"
        lines.append(f"  {prefix} {name_padded} {r.message}")

        if r.details:
            for detail_line in r.details.split("\n"):
                lines.append(f"                             {detail_line}")

        if r.status == "FAIL":
            fail_count += 1
            if r.fix_suggestion:
                lines.append(f"                             Fix: {Color.yellow(r.fix_suggestion)}")
        elif r.status == "WARN":
            warn_count += 1
            if r.fix_suggestion:
                lines.append(f"                             Hint: {Color.yellow(r.fix_suggestion)}")

    lines.append("")

    if fail_count:
        lines.append(Color.red(f"{fail_count} check(s) failed. See suggestions above."))
    elif warn_count:
        lines.append(Color.yellow(f"All checks passed with {warn_count} warning(s)."))
    else:
        lines.append(Color.green("All checks passed."))

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
