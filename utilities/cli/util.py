#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import grp
import re
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


# Utility Functions
def get_timestamp() -> str:
    """Get current timestamp in the format used by the bash script"""
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def fatal(message: str) -> None:
    """Print fatal error and exit with backtrace"""
    print(f"\033[31m{get_timestamp()} [FATAL] \033[0m{message}", file=sys.stderr)
    print("\nBacktrace: ...", file=sys.stderr)
    traceback.print_list(traceback.extract_stack()[-3:], file=sys.stderr)
    sys.exit(1)


def run_command(
    cmd: List[str], dry_run: bool = False, check: bool = True, **kwargs
) -> subprocess.CompletedProcess:
    """Run a shell command and handle errors"""
    cmd_str = " ".join(str(x) for x in cmd)
    if dry_run:
        print(f"\033[34m{get_timestamp()} \033[36m[dryrun] \033[37m$ \033[32m{cmd_str}\033[0m")
        return subprocess.CompletedProcess(cmd, 0)

    print(f"\033[34m{get_timestamp()} \033[37m$ \033[32m{cmd_str}\033[0m")
    try:
        return subprocess.run(cmd, check=check, **kwargs)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd_str}")
        print(f"Exit code: {e.returncode}")
        sys.exit(e.returncode)


def check_nvidia_ctk() -> None:
    """Check NVIDIA Container Toolkit version"""
    min_version = "1.12.0"
    recommended_version = "1.14.1"

    if not shutil.which("nvidia-ctk"):
        fatal("nvidia-ctk not found. Please install the NVIDIA Container Toolkit.")

    try:
        output = subprocess.check_output(["nvidia-ctk", "--version"], text=True)
        import re

        match = re.search(r"(\d+\.\d+\.\d+)", output)
        if match:
            version = match.group(1)
            from packaging import version as ver

            if ver.parse(version) < ver.parse(min_version):
                fatal(
                    f"Found nvidia-ctk Version {version}. Version {min_version}+ is required ({recommended_version}+ recommended)."
                )
        else:
            print(f"Failed to parse available nvidia-ctk version: {output}")
    except subprocess.CalledProcessError:
        fatal(f"Could not determine nvidia-ctk version. Version {min_version}+ required.")


def get_host_gpu() -> str:
    """Determine if running on dGPU or iGPU"""
    if not shutil.which("nvidia-smi"):
        print(
            "Could not find any GPU drivers on host. Defaulting build to target dGPU/CPU stack.",
            file=sys.stderr,
        )
        return "dgpu"

    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
        )
        if not output or "Orin (nvgpu)" in output.decode():
            return "igpu"
    except subprocess.CalledProcessError:
        return "dgpu"

    return "dgpu"


def get_compute_capacity() -> str:
    """Get GPU compute capacity"""
    if not shutil.which("nvidia-smi"):
        return "0.0"
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
        )
        return output.decode().strip().split("\n")[0]
    except subprocess.CalledProcessError:
        return "0.0"


def get_group_id(group: str) -> Optional[int]:
    """Get group ID for a given group name"""
    try:
        return grp.getgrnam(group).gr_gid
    except KeyError:
        return None


def normalize_language(language: str) -> Optional[str]:
    """Normalize language name"""
    if language.lower() == "cpp" or language.lower() == "c++":
        return "cpp"
    elif language.lower() == "python" or language.lower() == "py":
        return "python"
    return None


def list_metadata_json_dir(*paths: Path) -> List[Tuple[str, str]]:
    """List all metadata.json files in given paths"""
    results = []
    for path in paths:
        for json_path in path.rglob("metadata.json"):
            json_dir = json_path.parent
            dir_name = json_dir.name

            if "{{" in dir_name and "}}" in dir_name:
                continue  # Skip templates

            if dir_name in ["cpp", "python"]:
                language = f"({dir_name})"
                name = json_dir.parent.name
            else:
                language = ""
                name = dir_name

            results.append((name, language))

    return sorted(results)


def install_cuda_dependencies_package(
    package_name: str, preferred_version: str, optional: bool = False, dry_run: bool = False
) -> bool:
    """Install CUDA dependencies package with version checking

    Args:
        package_name: Name of the package to install
        preferred_version: Preferred version string to match
        optional: Whether the package is optional (default: False)

    Returns:
        bool: True if package was installed or already present, False if optional package was skipped

    Raises:
        SystemExit: If non-optional package cannot be installed
    """
    # Check if package is already installed
    try:
        output = subprocess.check_output(
            ["apt", "list", "--installed", package_name], text=True, stderr=subprocess.DEVNULL
        )
        # Extract installed version from apt list output using regex
        # Example output: "libcudnn9-cuda-12/unknown,now 9.5.1.17-1 amd64 [installed,upgradable to: 9.8.0.87-1]"
        installed_version = re.search(rf"{package_name}/.*?now\s+([\d\.-]+)", output)
        if installed_version:
            installed_version = installed_version.group(1)
            print(f"Package {package_name} found with version {installed_version}")
            return True
    except subprocess.CalledProcessError:
        pass

    # Check available versions
    try:
        available_versions = subprocess.check_output(
            ["apt", "list", "-a", package_name], text=True, stderr=subprocess.DEVNULL
        )

        # Find matching version
        matching_version = None
        for line in available_versions.splitlines():
            if preferred_version in line and package_name in line:
                matching_version = line.split()[1]  # Get version from second column
                break

        if not matching_version:
            if optional:
                print(f"Package {package_name} {preferred_version} not found. Skipping.")
                return False
            else:
                fatal(
                    f"{package_name} {preferred_version} is not installable.\n"
                    f"You might want to try to install a newer version manually and rerun the setup:\n"
                    f"  sudo apt install {package_name}"
                )

        # Install the package
        print(f"Installing {package_name}={matching_version}")
        run_command(
            [
                "sudo",
                "apt",
                "install",
                "--no-install-recommends",
                "-y",
                f"{package_name}={matching_version}",
            ],
            dry_run=dry_run,
        )
        return True

    except subprocess.CalledProcessError as e:
        if optional:
            print(f"Error checking available versions for {package_name}: {e}")
            return False
        else:
            fatal(f"Error checking available versions for {package_name}: {e}")
