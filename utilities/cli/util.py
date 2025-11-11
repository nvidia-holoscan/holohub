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

import functools
import grp
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Union

DEFAULT_BASE_SDK_VERSION = "3.8.0"

PROJECT_PREFIXES = {
    "application": "APP",
    "benchmark": "APP",
    "operator": "OP",
    "package": "PKG",
    "workflow": "APP",
    "default": "APP",  # specified type but not recognized
}

BUILD_TYPES = {
    "debug": "Debug",
    "release": "Release",
    "rel-debug": "RelWithDebInfo",
    "relwithdebinfo": "RelWithDebInfo",
    "default": "Release",
}


class Color:
    """Utility class for terminal color formatting"""

    # ANSI color codes
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Text attributes
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @staticmethod
    def format(text: str, color: str, bold: bool = False) -> str:
        """Format text with color and optional bold attribute"""
        result = color
        if bold:
            result += Color.BOLD
        result += text + Color.RESET
        return result

    def _create_color_method(color_code: str):
        """Create a color method for the given color code"""

        def color_method(text: str, bold: bool = False) -> str:
            return Color.format(text, color_code, bold)

        return color_method

    # Create color methods dynamically
    red = _create_color_method(RED)
    green = _create_color_method(GREEN)
    yellow = _create_color_method(YELLOW)
    blue = _create_color_method(BLUE)
    cyan = _create_color_method(CYAN)
    white = _create_color_method(WHITE)


_sudo_available = None  # Cache for sudo availability check
_apt_updated = False  # track whether apt update has been called


# Utility Functions
def get_timestamp() -> str:
    """Get current timestamp in the format used by the bash script"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def format_cmd(command: str, is_dryrun: bool = False) -> str:
    """Format command output with consistent timestamp and color formatting"""
    timestamp = Color.blue(get_timestamp())
    if is_dryrun:
        dryrun_tag = Color.cyan("[dryrun]")
        return f"{timestamp} {dryrun_tag} {Color.white('$')} {Color.green(command)}"
    return f"{timestamp} {Color.white('$')} {Color.green(command)}"


def info(message: str) -> None:
    """Print informational message with consistent formatting"""
    print(f"{Color.yellow('INFO:')} {message}")


def get_env_bool(
    env_var_name: str,
    default: bool = True,
    false_values: Tuple[str, ...] = ("false", "no", "n", "0", "f"),
) -> Tuple[str, bool]:
    """Check environment variable as boolean flag"""
    env_value = os.environ.get(env_var_name, str(default).lower())
    is_true = env_value.lower() not in false_values
    return env_value, is_true


def check_skip_builds(args) -> Tuple[bool, bool]:
    """Checking skip build flags and printing info messages"""
    holohub_always_build, always_build = get_env_bool("HOLOHUB_ALWAYS_BUILD", default=True)
    skip_builds = not always_build
    skip_docker_build = skip_builds or getattr(args, "no_docker_build", False)
    skip_local_build = skip_builds or getattr(args, "no_local_build", False)
    if skip_builds:
        info(f"Skipping build due to HOLOHUB_ALWAYS_BUILD={holohub_always_build}")
    else:
        if getattr(args, "no_local_build", False):
            info("Skipping local build due to --no-local-build")
        if getattr(args, "no_docker_build", False):
            info("Skipping container build due to --no-docker-build")
    return skip_docker_build, skip_local_build


def fatal(message: str) -> None:
    """Print fatal error and exit with backtrace"""
    print(
        f"{Color.red(get_timestamp())} {Color.red('[FATAL]', bold=True)} {message}", file=sys.stderr
    )
    print("\nBacktrace: ...", file=sys.stderr)
    traceback.print_list(traceback.extract_stack()[-3:], file=sys.stderr)
    sys.exit(1)


def warn(message: str) -> None:
    print(f"{Color.yellow('WARNING:')} {message}")


def _get_holohub_root() -> Path:
    """Get the HoloHub repository root path."""
    env_root = os.environ.get("HOLOHUB_ROOT")
    if env_root:
        env_path = Path(env_root).expanduser()
        if env_path.exists() and env_path.is_dir():
            return env_path
        warn(
            f"Environment variable HOLOHUB_ROOT='{env_root}' is invalid. "
            f"Falling back to default path: {Path(__file__).parent.parent.parent}"
        )
    return Path(__file__).parent.parent.parent


HOLOHUB_ROOT = _get_holohub_root()


def get_holohub_root() -> Path:
    return HOLOHUB_ROOT


def get_holohub_setup_scripts_dir() -> Path:
    return Path(
        os.environ.get("HOLOHUB_SETUP_SCRIPTS_DIR", HOLOHUB_ROOT / "utilities" / "setup")
    ).expanduser()


def _get_maybe_sudo() -> str:
    """Get sudo command if available, with caching to avoid repeated subprocess calls"""
    global _sudo_available

    if _sudo_available is not None:
        return _sudo_available
    _sudo_available = "sudo" if shutil.which("sudo") else ""
    return _sudo_available


def _classify_sudo_requirement(cmd: Union[str, List[str]]) -> Tuple[bool, str]:
    """Classify command sudo requirement and return (needs_sudo, reason)"""
    cmd_parts = cmd.split() if isinstance(cmd, str) else [str(x) for x in cmd]
    if not cmd_parts:
        return False, ""
    # Already has sudo
    if cmd_parts[0] == "sudo":
        return True, "Command already includes sudo"
    cmd_name = cmd_parts[0]
    # Commands that always need sudo
    always_sudo = {
        "apt": "Package management requires root privileges",
        "apt-get": "Package management requires root privileges",
        "dpkg": "Package database access requires root privileges",
        "chmod": "Changing file permissions requires root privileges",
        "chown": "Changing file ownership requires root privileges",
    }
    if cmd_name in always_sudo:
        return True, always_sudo[cmd_name]
    # Commands that need sudo for system paths
    if cmd_name in ["ln", "cp", "mv", "rm", "mkdir"]:
        if any(
            arg.startswith(("/etc/", "/usr/", "/var/", "/opt/", "/sys/", "/proc/"))
            for arg in cmd_parts[1:]
        ):
            return True, "Writing to system directories requires root privileges"
    # Shell commands with system redirections
    if isinstance(cmd, str) and ("tee /" in cmd or ">/etc/" in cmd or ">/usr/" in cmd):
        return True, "Writing to system locations requires root privileges"

    return False, ""


def _process_command_with_sudo(
    cmd: Union[str, List[str]], maybe_sudo: str
) -> Union[str, List[str]]:
    """Process command and add sudo if needed and available"""
    needs_sudo, _ = _classify_sudo_requirement(cmd)
    if not needs_sudo or not maybe_sudo:
        return cmd

    # Check if already has sudo anywhere in the command
    if isinstance(cmd, str):
        if cmd.strip().startswith("sudo ") or " sudo " in cmd:
            return cmd  # Don't add sudo if it's already present anywhere
        return f"{maybe_sudo} {cmd}"
    else:
        if cmd and (str(cmd[0]) == "sudo" or "sudo" in [str(x) for x in cmd]):
            return cmd  # Don't add sudo if it's already present anywhere
        return [maybe_sudo] + cmd


def run_command(
    cmd: Union[str, List[str]], dry_run: bool = False, check: bool = True, **kwargs
) -> subprocess.CompletedProcess:
    """Run a command and handle errors"""
    # Process the command and add sudo if needed
    processed_cmd = _process_command_with_sudo(cmd, _get_maybe_sudo())
    if isinstance(processed_cmd, str):
        cmd_str = processed_cmd
    else:
        cmd_list = [f'"{x}"' if " " in str(x) else str(x) for x in processed_cmd]
        cmd_str = format_long_command(cmd_list) if dry_run else " ".join(cmd_list)

    needs_sudo, sudo_reason = _classify_sudo_requirement(cmd)  # Add reason for sudo usage
    if needs_sudo:
        print(Color.yellow(f"[SUDO REQUIRED] {sudo_reason}"))
    if dry_run:
        print(format_cmd(cmd_str, is_dryrun=True))
        return subprocess.CompletedProcess(cmd_str, 0)

    print(format_cmd(cmd_str))
    try:
        return subprocess.run(processed_cmd, check=check, **kwargs)
    except subprocess.CalledProcessError as e:
        print(f"Non-zero exit code running command: {cmd_str}")
        print(f"Exit code: {e.returncode}")
        sys.exit(e.returncode)


def run_info_command(cmd: List[str]) -> Optional[str]:
    """Run a command for information gathering and return stripped output or None if failed"""
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def parse_semantic_version(version: str) -> Tuple[int, int, int]:
    """
    Parse semantic version string MAJOR.MINOR.PATCH into tuple of integers for comparison

    Note: Implementing our own version parsing to avoid dependency on PyPI 'packaging' module.

    ref: https://semver.org/
    """
    match = re.match(r"^(\d+\.\d+\.\d+).*", version.strip())
    if not match:
        raise ValueError(f"Failed to parse semantic version string: {version}")
    return tuple(map(int, match.group(1).split(".")))


def check_nvidia_ctk(min_version: str = "1.12.0", recommended_version: str = "1.14.1") -> None:
    """Check NVIDIA Container Toolkit version"""

    if not shutil.which("nvidia-ctk"):
        fatal("nvidia-ctk not found. Please install the NVIDIA Container Toolkit.")

    try:
        output = subprocess.check_output(["nvidia-ctk", "--version"], text=True)
        match = re.search(r"(\d+\.\d+\.\d+)", output)
        if match:
            version = match.group(1)
            try:
                version_check = parse_semantic_version(version) < parse_semantic_version(
                    min_version
                )
            except ValueError:
                version_check = False

            if version_check:
                fatal(
                    f"Found nvidia-ctk {version}. Version {min_version}+ is required "
                    f"({recommended_version}+ recommended)."
                )
        else:
            print(f"Failed to parse available nvidia-ctk version: {output}")
    except subprocess.CalledProcessError:
        fatal(f"Could not determine nvidia-ctk version. Version {min_version}+ required.")


def get_gpu_name() -> Optional[str]:
    """
    Helper function to get GPU name from nvidia-smi.  Returns None if nvidia-smi is not available.
    """
    if not shutil.which("nvidia-smi"):
        return None
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return output.strip() if output else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_host_gpu() -> str:
    """Determine if running on dGPU or iGPU"""
    gpu_name = get_gpu_name()
    if gpu_name is None:
        print(
            "Could not find any GPU drivers on host. Defaulting build to target dGPU/CPU stack.",
            file=sys.stderr,
        )
        return "dgpu"

    # Check for iGPU (Orin integrated GPU)
    if "Orin (nvgpu)" in gpu_name:
        return "igpu"
    return "dgpu"


def get_default_cuda_version() -> str:
    """
    Get default CUDA version based on NVIDIA driver version.

    Returns:
        - "13" if driver version >= 580 or if nvidia-smi is not available
        - "12" if driver version < 580
    """
    # Default to CUDA 13 if nvidia-smi is not available
    if not shutil.which("nvidia-smi"):
        warn("nvidia-smi not found, default CUDA version is 13")
        return "13"

    # Check the driver version using nvidia-smi
    driver_version = run_info_command(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )

    if not driver_version:
        warn("Unable to detect NVIDIA driver version, default CUDA version is 13")
        return "13"

    try:
        driver_major_version = int(driver_version.split(".")[0])
        if driver_major_version >= 580:
            return "13"
        else:
            return "12"
    except (ValueError, IndexError):
        warn(f"Unable to parse driver version '{driver_version}', default CUDA version is 13")
        return "13"


def get_cuda_tag(cuda_version: Optional[Union[str, int]] = None, sdk_version: str = "3.6.1") -> str:
    """
    Determine the CUDA container tag based on CUDA version and GPU type.

    SDK version support:
    - SDK < 3.6.1: Old format (dgpu/igpu)
    - SDK == 3.6.1: only cuda13-dgpu available
    - SDK >= 3.7.0: Full CUDA support
      - cuda13: CUDA 13 (x86_64, Jetson Thor)
      - cuda12-dgpu: CUDA 12 dGPU (x86_64, IGX Orin dGPU, Clara AGX dGPU, GH200)
      - cuda12-igpu: CUDA 12 iGPU (Jetson Orin, IGX Orin iGPU, Clara AGX iGPU)

    Args:
        cuda_version: CUDA major version (e.g., 12, 13). If None, uses platform default.
        sdk_version: SDK version string (e.g., "3.6.0", "3.6.1", "3.7.0").

    Returns:
        The appropriate container tag string
    """
    try:
        sdk_ver = parse_semantic_version(sdk_version)
    except (ValueError, IndexError):
        sdk_ver = parse_semantic_version(DEFAULT_BASE_SDK_VERSION)
    if sdk_ver < (3, 6, 1):
        return get_host_gpu()
    if sdk_ver == (3, 6, 1):
        return "cuda13-dgpu"
    if cuda_version is None:
        cuda_version = get_default_cuda_version()
    cuda_str = str(cuda_version)
    if cuda_str == "13":
        return "cuda13"
    if cuda_str == "12":
        return f"cuda12-{get_host_gpu()}"
    return f"cuda{cuda_str}-{get_host_gpu()}"


def get_host_arch() -> str:
    """Get host architecture"""
    machine = platform.machine().lower()
    if machine in ["x86_64", "amd64"]:
        return "x86_64"
    if machine in ["aarch64", "arm64"]:
        return "aarch64"
    return machine


def get_arch_gpu_str() -> str:
    """Get architecture+GPU string like bash get_arch+gpu_str()"""
    arch = get_host_arch()
    if arch == "aarch64":
        gpu = get_host_gpu()
        return f"{arch}-{gpu}"
    return arch


def _is_valid_sdk_installation(path: Union[str, Path]) -> bool:
    """
    Validate if a directory contains a valid Holoscan SDK installation.
    """
    path = Path(path) if isinstance(path, str) else path
    if not path.exists() or not path.is_dir() or not (path / "lib").exists():
        return False
    # Check for at least one of these to confirm it's a Holoscan SDK
    return (path / "lib" / "cmake" / "holoscan" / "holoscan-config.cmake").exists() or (
        path / "lib" / "cmake" / "holoscan" / "HoloscanConfig.cmake"
    ).exists()


def find_hsdk_build_rel_dir(local_sdk_root: Optional[Union[str, Path]] = None) -> str:
    """
    Find a suitable SDK installation or build directory.
    https://github.com/nvidia-holoscan/holoscan-sdk/blob/9c5b3c3d4831f2e65ebda6b79ae9b1c5517c6a7c/run#L226-L228

    Search order:
    1. Direct SDK installation directory
    2. Environment variable `HOLOSCAN_SDK_ROOT` SDK root directory
    3. Assuming the direct or env var is the src code root, searching for immediate subdirectories:
        3.1 Install directory (prefer)
        3.2 Build directory (fallback)

    Args:
        local_sdk_root: Path to SDK root directory, or direct SDK installation/build directory

    Returns:
        Relative path to the SDK directory from the root, or absolute path if passed directly
    """
    search_paths = []

    # Handle user-provided path
    if local_sdk_root:
        local_sdk_root = Path(local_sdk_root) if isinstance(local_sdk_root, str) else local_sdk_root
        if local_sdk_root.exists():
            # Check if this is a direct SDK installation directory
            if _is_valid_sdk_installation(local_sdk_root):
                return str(local_sdk_root)
            else:
                # Treat as SDK root directory to search
                search_paths.append(local_sdk_root)

    # Add environment variable path
    if os.environ.get("HOLOSCAN_SDK_ROOT"):
        env_path = Path(os.environ["HOLOSCAN_SDK_ROOT"])
        if env_path.exists():
            if _is_valid_sdk_installation(env_path):
                return str(env_path)
            else:
                search_paths.append(env_path)

    # Search within SDK root directories
    arch_gpu = get_arch_gpu_str()
    for sdk_path in search_paths:
        for install_dir in [f"install-{arch_gpu}", "install"]:
            if _is_valid_sdk_installation(sdk_path / install_dir):
                return install_dir
        for install_dir in sorted([d.name for d in sdk_path.glob("install-*") if d.is_dir()]):
            if _is_valid_sdk_installation(sdk_path / install_dir):
                return install_dir
        for build_dir in [f"build-{arch_gpu}", "build"]:
            if _is_valid_sdk_installation(sdk_path / build_dir):
                return build_dir
        for build_dir in sorted([d.name for d in sdk_path.glob("build-*") if d.is_dir()]):
            if _is_valid_sdk_installation(sdk_path / build_dir):
                return build_dir
    info(
        f"Valid SDK installation not found. Looking for 'install-{arch_gpu}' or 'build-{arch_gpu}'."
    )
    return f"build-{arch_gpu}"


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


def normalize_language(language: str | None) -> str:
    """Normalize language name"""
    # Handle empty language
    if not language:
        return ""

    # Handle invalid language type
    if not isinstance(language, str):
        print(f"WARNING: Language must be a string, got {type(language)}: {language}")
        return ""

    # Normalize language name
    language = language.lower()
    if language in ["cpp", "c++"]:
        return "cpp"
    if language in ["python", "py"]:
        return "python"
    raise ValueError(f"Invalid language: {language}")


def list_normalized_languages(language: str | list[str] | None) -> list[str]:
    """Make list of normalized languages from a single language or list of languages"""
    if isinstance(language, list):
        return [normalize_language(lang) for lang in language]
    return [normalize_language(language)]


def determine_project_prefix(project_type: str) -> str:
    type_str = project_type.lower().strip()
    if type_str in PROJECT_PREFIXES:
        return PROJECT_PREFIXES[type_str]
    return PROJECT_PREFIXES["default"]


def get_buildtype_str(build_type: Optional[str]) -> str:
    """Get CMake build type string"""
    if not build_type:
        return os.environ.get("CMAKE_BUILD_TYPE", BUILD_TYPES["default"])
    build_type_str = build_type.lower().strip()
    return BUILD_TYPES.get(build_type_str, BUILD_TYPES["default"])


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


class PackageInstallationError(Exception):
    """Raised when a package cannot be installed via apt"""

    def __init__(self, package_name: str, version_pattern: str, message: str = None):
        self.package_name = package_name
        self.version_pattern = version_pattern
        super().__init__(
            message or f"Failed to install package {package_name} matching {version_pattern}"
        )


def get_installed_package_version(package_name: str) -> Optional[str]:
    """Get the installed version of a package"""
    try:
        result = subprocess.run(
            ["dpkg-query", "-W", "-f=${Version}", package_name],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def get_available_package_versions(package_name: str) -> List[str]:
    """Get available versions of a package from apt"""
    try:
        result = subprocess.run(
            ["apt-cache", "madison", package_name], capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            return []

        versions = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                parts = line.split("|")
                if len(parts) >= 2:
                    version = parts[1].strip()
                    if version:
                        versions.append(version)
        return versions
    except Exception:
        return []


def ensure_apt_updated(dry_run: bool = False) -> None:
    """Ensure apt package list is updated, but only once per session"""
    global _apt_updated
    if not _apt_updated:
        run_command(["apt-get", "update"], dry_run=dry_run)
        _apt_updated = True


def install_packages_if_missing(
    packages: List[str], dry_run: bool = False, apt_options: List[str] = None
) -> List[str]:
    """Install packages only if they're not already installed

    Args:
        packages: List of package names to install (can include version specs like "pkg=1.0*")
            Note: If package has a version spec, it always runs sudo apt install to ensure version.
        dry_run: Whether to perform a dry run
        apt_options: Additional options for apt install

    Returns:
        List of packages that were actually installed (or would be installed in dry run)
    """
    if apt_options is None:
        apt_options = ["--no-install-recommends", "-y"]

    packages_to_install = []

    for package_spec in packages:
        package_name = package_spec.split("=")[0]

        if "=" in package_spec:
            packages_to_install.append(package_spec)
            info(f"Installing {package_spec}")
        else:
            if get_installed_package_version(package_name):
                info(f"Package {package_name} is already installed")
            else:
                packages_to_install.append(package_spec)

    if packages_to_install:
        ensure_apt_updated(dry_run=dry_run)
        install_cmd = ["apt", "install"] + apt_options + packages_to_install
        run_command(install_cmd, dry_run=dry_run)

    return packages_to_install


def install_cuda_dependencies_package(
    package_name: str,
    version_pattern: str = r"\d+\.\d+\.\d+",
    dry_run: bool = False,
) -> str:
    """Install CUDA dependencies package with version checking

    Args:
        package_name: Name of the package to install
        version_pattern: Regular expression for package version to install
        dry_run: Whether to perform a dry run

    Returns:
        str: Installed package version

    Raises:
        PackageInstallationError: If package cannot be installed
    """
    installed_version = get_installed_package_version(package_name)
    if installed_version:
        if re.search(version_pattern, installed_version):
            info(f"Package {package_name} version {installed_version} already installed")
            return installed_version
        else:
            info(f"{package_name} version {installed_version} not match pattern {version_pattern}")

    available_versions = get_available_package_versions(package_name)
    if not available_versions:
        raise PackageInstallationError(
            package_name, version_pattern, f"No versions available for {package_name}"
        )

    matching_versions = [v for v in available_versions if re.search(version_pattern, v)]
    if not matching_versions:
        raise PackageInstallationError(
            package_name,
            version_pattern,
            f"{package_name} has no versions matching pattern {version_pattern}.\n"
            f"Available versions: {', '.join(available_versions[:5])}\n"
            f"You might need to install manually: sudo apt install {package_name}",
        )

    target_version = matching_versions[0]
    install_packages_if_missing(
        [f"{package_name}={target_version}"],
        apt_options=["--no-install-recommends", "-y", "--allow-downgrades"],
        dry_run=dry_run,
    )

    return target_version


def format_long_command(cmd: List[str], max_line_length: int = 80) -> str:
    """Format a long command into multiple lines for better readability

    Args:
        cmd: Command to format as a list of strings
        max_line_length: Maximum line length before wrapping

    Returns:
        Formatted command string with line continuations
    """
    if not cmd:
        return ""

    # Check if total command length exceeds max length
    total_length = sum(len(arg) + 1 for arg in cmd) - 1
    if total_length <= max_line_length:
        return " ".join(cmd)

    # Start with the first command
    formatted = cmd[0]
    current_line = cmd[0]

    # Common patterns that suggest good break points
    break_patterns = {
        "--",  # Long options
        "-",  # Short options
        "&&",  # Command chaining
        "||",  # Command chaining
        "|",  # Pipes
        ";",  # Command separator
        ">",  # Output redirection
        "<",  # Input redirection
        ">>",  # Append redirection
        "2>",  # Error redirection
    }

    for i, arg in enumerate(cmd[1:]):
        # Check if this is a good place to break
        should_break = (
            # Break if we exceed max length
            len(current_line) + len(arg) + 1 > max_line_length
            or
            # Break before common command separators
            any(arg.startswith(pattern) for pattern in break_patterns)
            or
            # Break after common command separators
            any(cmd[i].endswith(pattern) for pattern in break_patterns)
        )

        if should_break:
            formatted += " \\\n    " + arg
            current_line = arg
        else:
            formatted += " " + arg
            current_line += " " + arg

    return formatted


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings."""
    s1 = s1.lower()
    s2 = s2.lower()

    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def list_cmake_dir_options(script_dir: Path, cmake_function: str) -> List[str]:
    """Get list of directories from CMakeLists.txt files"""
    results = []
    for cmakelists in script_dir.rglob("CMakeLists.txt"):
        with open(cmakelists) as f:
            content = f.read()
            for line in content.splitlines():
                if cmake_function in line:
                    try:
                        name = line.split("(")[1].split(")")[0].strip()
                        results.append(name)
                    except IndexError:
                        continue
    return sorted(results)


@functools.lru_cache(maxsize=32)
def resolve_path_prefix(prefix: Optional[str] = None) -> str:
    """Resolve the path prefix for HoloHub placeholders"""
    if prefix is None:
        prefix = os.environ.get("HOLOHUB_PATH_PREFIX", "holohub_")
    if not prefix.endswith("_"):
        prefix = prefix + "_"
    return prefix


def build_holohub_path_mapping(
    holohub_root: Path,
    project_data: Optional[dict] = None,
    build_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    prefix: Optional[str] = None,
) -> dict[str, str]:
    """Build a mapping of HoloHub placeholders to their resolved paths

    Args:
        holohub_root: Root directory of HoloHub
        project_data: Optional project metadata dictionary
        build_dir: Optional build directory path
        data_dir: Optional data directory path
        prefix: Prefix for placeholder keys. If None, reads from HOLOHUB_PATH_PREFIX
                environment variable (default: "holohub_")

    Returns:
        Dictionary mapping placeholder names to their resolved paths
    """
    prefix = resolve_path_prefix(prefix)

    if data_dir is None:
        data_dir = holohub_root / "data"

    path_mapping = {
        f"{prefix}root": str(holohub_root),
        f"{prefix}data_dir": str(data_dir),
    }
    if not project_data:
        return path_mapping
    # Add project-specific mappings if project_data is provided
    app_source_path = project_data.get("source_folder", "")
    if app_source_path:
        path_mapping[f"{prefix}app_source"] = str(app_source_path)
    if build_dir:
        path_mapping[f"{prefix}bin"] = str(build_dir)
        if app_source_path:
            try:
                app_build_dir = build_dir / Path(app_source_path).relative_to(holohub_root)
                path_mapping[f"{prefix}app_bin"] = str(app_build_dir)
            except ValueError:
                # Handle case where app_source_path is not relative to holohub_root
                path_mapping[f"{prefix}app_bin"] = str(build_dir)
    elif project_data.get("project_name"):
        # If no build_dir provided but we have project name, try to infer it
        project_name = project_data["project_name"]
        inferred_build_dir = holohub_root / "build" / project_name
        path_mapping[f"{prefix}bin"] = str(inferred_build_dir)
        if app_source_path:
            try:
                app_build_dir = inferred_build_dir / Path(app_source_path).relative_to(holohub_root)
                path_mapping[f"{prefix}app_bin"] = str(app_build_dir)
            except ValueError:
                path_mapping[f"{prefix}app_bin"] = str(inferred_build_dir)
    return path_mapping


def docker_args_to_devcontainer_format(docker_args: List[str]) -> List[str]:
    """Convert Docker argument format to devcontainer format (--flag value -> --flag=value)"""
    standalone = {"--rm", "--init", "--no-cache"}
    result, i = [], 0
    while i < len(docker_args):
        curr = docker_args[i]
        if (
            i + 1 < len(docker_args)
            and curr.startswith("--")
            and "=" not in curr
            and curr not in standalone
            and not docker_args[i + 1].startswith("-")
        ):
            result.append(f"{curr}={docker_args[i + 1]}")
            i += 2
        else:
            result.append(curr)
            i += 1
    return result


def get_entrypoint_command_args(
    img: str, command: str, docker_opts: str, dry_run: bool = False
) -> tuple[str, List[str]]:
    """Determine how to execute a shell command in a Docker container."""

    # Check if user provided a custom entrypoint
    entrypoint = None
    if "--entrypoint" in docker_opts:
        try:
            tokens = shlex.split(docker_opts)
            for i, token in enumerate(tokens):
                if token == "--entrypoint" and i + 1 < len(tokens):
                    entrypoint = tokens[i + 1]
                    break
                elif token.startswith("--entrypoint="):
                    entrypoint = token.split("=", 1)[1]
                    break
        except ValueError:
            pass

    if entrypoint:  # If user provided a custom entrypoint
        if entrypoint in ["/bin/sh", "/bin/bash", "sh", "bash"]:
            return "", ["-c", command]  # Shell needs -c to execute command string
        return "", shlex.split(command)  # For non-shell user entrypoints, pass command as arguments

    entrypoint = get_container_entrypoint(img, dry_run=dry_run)
    if not entrypoint:  # Image has no entrypoint, use default "/bin/bash -c"
        return "", ["/bin/bash", "-c", command]
    # Image has an ENTRYPOINT
    if entrypoint in [["/bin/sh", "-c"], ["/bin/bash", "-c"], ["sh", "-c"], ["bash", "-c"]]:
        return "", [command]  # Shell is already configured to take command string
    if entrypoint[0] in ["/bin/sh", "/bin/bash", "sh", "bash"]:
        return "", ["-c", command]  # Shell needs -c to execute command string
    return "--entrypoint=/bin/bash", ["-c", command]  # bash is used to run local build/run command


def get_container_entrypoint(img: str, dry_run: bool = False) -> Optional[List[str]]:
    """Check if container image has an entrypoint defined"""
    if dry_run:
        print(
            Color.yellow(
                "Inspect docker image entrypoint: "
                f"docker inspect --format={{{{json .Config.Entrypoint}}}} {img}"
            )
        )
        return None

    try:
        docker_exe = os.environ.get("HOLOHUB_DOCKER_EXE", "docker")
        result = run_command(
            [docker_exe, "inspect", "--format={{json .Config.Entrypoint}}", img],
            capture_output=True,
            check=False,
            dry_run=dry_run,
        )
        if result.returncode != 0:
            return None
        entrypoint_json = result.stdout.strip()
        if entrypoint_json in ["<no value>", "[]", "null", "''"]:
            return None
        parsed = json.loads(entrypoint_json)
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed
        return None
    except Exception:
        pass
    return None


def get_image_pythonpath(img: str, dry_run: bool = False) -> str:
    """Get PYTHONPATH from the Docker image environment"""
    if dry_run:
        print(
            Color.yellow(
                "Inspect docker image PYTHONPATH: docker inspect "
                f"--format '{{{{range .Config.Env}}}}{{{{println .}}}}{{{{end}}}}' {img}"
            )
        )
        return ""
    try:
        docker_exe = os.environ.get("HOLOHUB_DOCKER_EXE", "docker")
        result = run_command(
            [docker_exe, "inspect", "--format", "{{range .Config.Env}}{{println .}}{{end}}", img],
            check=False,
            capture_output=True,
            dry_run=dry_run,
        )
        if result.returncode != 0:
            return ""
        for line in result.stdout.decode().strip().split("\n"):
            if line.startswith("PYTHONPATH="):
                return line[len("PYTHONPATH=") :]
    except (subprocess.CalledProcessError, AttributeError):
        pass
    return ""


def replace_placeholders(text: str, path_mapping: dict[str, str]) -> str:
    """Replace placeholders in text using the provided path mapping"""
    if not text:
        return text
    result = text
    for placeholder, replacement in path_mapping.items():
        bracketed_placeholder = f"<{placeholder}>"
        result = result.replace(bracketed_placeholder, replacement)
    return result


def launch_vscode(workspace_path: str, dry_run: bool = False) -> None:
    """Install VS Code Remote Development extension and launch VS Code with new window"""
    print("Installing VS Code Remote Development extension...")
    run_command(
        [
            "code",
            "--force",
            "--install-extension",
            "ms-vscode-remote.vscode-remote-extensionpack",
        ],
        dry_run=dry_run,
    )
    run_command(["code", "--new-window", workspace_path], dry_run=dry_run)


def open_url(url: str, dry_run: bool = False) -> bool:
    """Open a URL using the system's default URL opener"""
    if shutil.which("open"):
        run_command(["open", url], check=False, dry_run=dry_run)
        return True
    elif shutil.which("xdg-open"):
        run_command(["xdg-open", url], check=False, dry_run=dry_run)
        return True
    if not dry_run:
        print("Could not automatically open URL.")
        print(f"Please manually open: {url}")
    return False


def launch_vscode_devcontainer(
    workspace_path: str, workspace_name: str = "holohub", dry_run: bool = False
) -> None:
    """Launch VS Code with dev container and open the dev container URL"""
    hash_hex = str(workspace_path).encode().hex()
    url = f"vscode://vscode-remote/dev-container+{hash_hex}/workspace/{workspace_name}"

    if dry_run:
        print(f"Dryrun URL: {url}")
    else:
        print(f"Launching VSCode Dev Container from: {workspace_path}")
        print(f"Connecting to {url}...")
    launch_vscode(workspace_path, dry_run=dry_run)
    open_url(url, dry_run=dry_run)


def get_devcontainer_config(
    holohub_root: Path, project_name: Optional[str] = None, dry_run: bool = False
) -> str:
    """Get devcontainer configuration content"""

    default_config_path = holohub_root / ".devcontainer"
    if (
        project_name
        and (holohub_root / ".devcontainer" / project_name / "devcontainer.json").exists()
    ):
        dev_container_path = holohub_root / ".devcontainer" / project_name
        print(f"Using application-specific DevContainer configuration: {dev_container_path}")
    else:
        dev_container_path = default_config_path
        print(f"Using top-level DevContainer configuration: {dev_container_path}")

    devcontainer_json_src = dev_container_path / "devcontainer.json"

    if dry_run:
        print(f"Would read and modify {devcontainer_json_src}")
        print("Would substitute environment variables and launch VS Code")
        return ""
    else:
        with open(devcontainer_json_src, "r") as f:
            devcontainer_content = f.read()

    return devcontainer_content


def collect_system_info() -> None:
    """Collect and display system information"""
    print(f"\n{Color.blue('System Information:')}")
    print(f"  OS: {platform.system()} {platform.release()} {platform.machine()}")
    print(f"  Platform: {platform.platform()}")


def collect_python_info() -> None:
    """Collect and display Python information"""
    print(f"\n{Color.blue('Python Information:')}")
    print(f"  Version: {sys.version}")
    print(f"  Executable: {sys.executable} Path: {sys.path[0] if sys.path else 'N/A'}")


def collect_holohub_info(
    holohub_root: Path, build_dir: Path, data_dir: Path, sdk_dir: Path
) -> None:
    """Collect and display HoloHub information"""
    print(f"\n{Color.blue('HoloHub Information:')}")
    print(f"  HOLOHUB_ROOT: {holohub_root}")
    print(f"  HOLOHUB_BUILD_PARENT_DIR: {build_dir}")
    print(f"  HOLOHUB_DATA_DIR: {data_dir}")
    print(f"  HOLOHUB_SDK_DIR: {sdk_dir}")


def collect_git_info(holohub_root: Path) -> None:
    """Collect and display Git repository information"""
    print(f"\n{Color.blue('Git Repository Information:')}")
    if not holohub_root.exists() or not holohub_root.is_dir():
        print(f"  HoloHub root directory does not exist or is not a directory: {holohub_root}")
        return
    original_cwd = os.getcwd()
    try:
        os.chdir(holohub_root)
    except Exception as e:
        print(f"  Cannot access HoloHub directory: {e}")
        return
    try:
        git_branch = run_info_command(["git", "branch", "--show-current"])
        git_commit_full = run_info_command(["git", "rev-parse", "HEAD"])
        git_status = run_info_command(["git", "status", "--porcelain"])
        if git_branch is None or git_commit_full is None or git_status is None:
            print("  Git information not available")
            return
        git_commit = git_commit_full[:8]
        print(f"  Branch: {git_branch} Commit: {git_commit}")
        print(f"  Modified: {git_status.splitlines()}")
    finally:
        try:
            os.chdir(original_cwd)  # try to restore the original working directory
        except Exception:
            pass


def collect_docker_info() -> None:
    """Collect and display Docker information"""
    print(f"\n{Color.blue('Docker Information:')}")
    docker_exe = os.environ.get("HOLOHUB_DOCKER_EXE", "docker")
    docker_version = run_info_command([docker_exe, "--version"])
    docker_info = run_info_command([docker_exe, "info", "--format", "{{.ServerVersion}}"])
    if docker_version is None or docker_info is None:
        print("  Docker not available")
        return
    print(f"  Version: {docker_version} Server Version: {docker_info}")
    nvidia_ctk_version = run_info_command(["nvidia-ctk", "--version"])
    if nvidia_ctk_version is not None:
        print(f"  NVIDIA Container Toolkit: {nvidia_ctk_version.strip()}")


def collect_cuda_gpu_info() -> None:
    """Collect and display CUDA/GPU information"""
    print(f"\n{Color.blue('CUDA/GPU Information:')}")
    nvidia_smi = run_info_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if nvidia_smi is None:
        print("  NVIDIA GPU/CUDA not available")
        return
    for i, line in enumerate(nvidia_smi.split("\n")):
        if line.strip():
            parts = line.split(",")
            if len(parts) >= 3:
                print(f"  GPU {i}: {parts[0].strip()}")
                print(f"    Driver: {parts[1].strip()}")
                print(f"    Memory: {parts[2].strip()} MB")
    cuda_version = run_info_command(["nvcc", "--version"])
    if cuda_version is None:
        print("  nvcc not available")
        return
    version_line = [line for line in cuda_version.split("\n") if "release" in line.lower()]
    print(f"  NVCC: {version_line[0].strip()}")
    nvcc_path = run_info_command(["which", "nvcc"])
    if nvcc_path is not None:
        print(f"  NVCC Path: {nvcc_path}")


def collect_environment_variables() -> None:
    """Collect and display environment variables"""
    print(f"\n{Color.blue('HoloHub Environment Variables:')}")
    holohub_env_vars = [
        "HOLOHUB_CMD_NAME",
        "HOLOHUB_BUILD_LOCAL",
        "HOLOHUB_ALWAYS_BUILD",
        "HOLOHUB_BUILD_PARENT_DIR",
        "HOLOHUB_DATA_DIR",
        "HOLOHUB_DEFAULT_HSDK_DIR",
        "HOLOHUB_CTEST_SCRIPT",
        "HOLOHUB_REPO_PREFIX",
        "HOLOHUB_CONTAINER_PREFIX",
        "HOLOHUB_WORKSPACE_NAME",
        "HOLOHUB_HOSTNAME_PREFIX",
        "HOLOHUB_BASE_IMAGE",
        "HOLOHUB_DOCKER_EXE",
        "HOLOHUB_SDK_PATH",
        "HOLOHUB_BASE_SDK_VERSION",
        "HOLOHUB_BENCHMARKING_SUBDIR",
        "HOLOHUB_DEFAULT_DOCKERFILE",
        "HOLOHUB_BASE_IMAGE_FORMAT",
        "HOLOHUB_DEFAULT_IMAGE_FORMAT",
        "HOLOHUB_DEFAULT_DOCKER_BUILD_ARGS",
        "HOLOHUB_DEFAULT_DOCKER_RUN_ARGS",
        "HOLOHUB_DOCS_URL",
        "HOLOHUB_CLI_DOCS_URL",
        "HOLOHUB_DATA_PATH",
        "HOLOHUB_SETUP_SCRIPTS_DIR",
        # Legacy variables
        "HOLOHUB_APP_NAME",
        "HOLOHUB_CONTAINER_BASE_NAME",
        "HOLOHUB_PATH_PREFIX",
    ]
    for var in sorted(holohub_env_vars):
        print(f"  {var}: {os.environ.get(var) or '(not set)'}")

    print(f"\n{Color.blue('Holoscan Environment Variables:')}")
    holoscan_env_vars = ["HOLOSCAN_SDK_VERSION", "HOLOSCAN_INPUT_PATH"]
    for var in sorted(holoscan_env_vars):
        print(f"  {var}: {os.environ.get(var) or '(not set)'}")

    print(f"\n{Color.blue('Other Relevant Environment Variables:')}")
    other_env_vars = [
        "PYTHONPATH",
        "PATH",
        "LD_LIBRARY_PATH",
        "CMAKE_BUILD_TYPE",
        "DOCKER_BUILDKIT",
        "XDG_SESSION_TYPE",
        "XDG_RUNTIME_DIR",
    ]
    for var in sorted(other_env_vars):
        print(f"  {var}: {os.environ.get(var) or '(not set)'}")


def is_running_in_docker() -> bool:
    """Check if the current process is inside a Docker container"""
    try:
        if os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
            return True
        with open("/proc/1/cgroup", "r") as f:
            return any(indicator in f.read() for indicator in ["docker", "containerd", "kubepods"])

    except (OSError, IOError):
        return False


def collect_env_info() -> None:
    """Collect and display comprehensive environment information"""
    collect_system_info()
    collect_python_info()
    collect_docker_info()
    collect_cuda_gpu_info()
    collect_environment_variables()


def normalize_args_str(args):
    """Convert arguments to string format, handling both string and array inputs"""
    if isinstance(args, str):
        return os.path.expandvars(args)
    elif isinstance(args, list):
        expanded_args = [os.path.expandvars(arg) for arg in args]
        return " ".join(expanded_args)
    return ""


def get_ubuntu_codename() -> str:
    """Get Ubuntu codename from os-release"""
    try:
        with open("/etc/os-release") as f:
            content = f.read()
        match = re.search(r"UBUNTU_CODENAME=(\w+)", content)
        return match.group(1) if match else "jammy"
    except (FileNotFoundError, AttributeError):
        return "jammy"


def setup_cmake(min_version: str = "3.26.4", dry_run: bool = False) -> None:
    """Setup CMake from Kitware if needed"""
    cmake_ver = get_installed_package_version("cmake")
    if cmake_ver and parse_semantic_version(cmake_ver) >= parse_semantic_version(min_version):
        return
    ubuntu_codename = get_ubuntu_codename()
    install_packages_if_missing(["gpg"], dry_run=dry_run)
    maybe_sudo = _get_maybe_sudo()
    run_command(
        "wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | "
        "gpg --dearmor - | "
        f"{maybe_sudo} tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null",
        dry_run=dry_run,
        shell=True,
    )
    run_command(
        f'echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] '
        f'https://apt.kitware.com/ubuntu/ {ubuntu_codename} main" | '
        f"{maybe_sudo} tee /etc/apt/sources.list.d/kitware.list >/dev/null",
        dry_run=dry_run,
        shell=True,
    )
    install_packages_if_missing(["cmake", "cmake-curses-gui"], dry_run=dry_run)


def setup_python_dev(min_version: str = "3.10.0", dry_run: bool = False) -> None:
    """Setup Python development packages"""
    python_version = sys.version_info
    python_dev_package = f"python3.{python_version.minor}-dev"
    pydev_ver = get_installed_package_version(python_dev_package)
    if not pydev_ver:
        pydev_ver = get_installed_package_version("python3-dev")
    if not pydev_ver or parse_semantic_version(pydev_ver) < parse_semantic_version(min_version):
        install_packages_if_missing([python_dev_package], dry_run=dry_run)


def setup_ngc_cli(dry_run: bool = False) -> None:
    """Setup NGC CLI if not present"""
    if shutil.which("ngc"):
        return

    arch_suffix = "arm64" if platform.machine() == "aarch64" else "linux"
    ngc_url = (
        "https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli"
        f"/versions/3.64.3/files/ngccli_{arch_suffix}.zip"
    )
    ngc_filename = f"ngccli_{arch_suffix}.zip"

    try:
        run_command(
            ["wget", "--quiet", "--content-disposition", ngc_url, "-O", ngc_filename],
            dry_run=dry_run,
        )
        run_command(["unzip", "-q", ngc_filename], dry_run=dry_run)
        run_command(["chmod", "u+x", "ngc-cli/ngc"], dry_run=dry_run)

        # Use absolute path for symlink
        abs_path = os.path.abspath("ngc-cli/ngc")
        run_command(["ln", "-s", abs_path, "/usr/local/bin/ngc"], dry_run=dry_run)

    except Exception as e:
        fatal(f"Failed to install NGC CLI: {e}")


def get_cuda_runtime_version() -> Optional[str]:
    """Get CUDA runtime version from dpkg"""
    try:
        result = subprocess.run(["dpkg", "-l"], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return None

        cuda_pattern = re.search(r"cuda-cudart-[0-9]+-[0-9]+.*\n", result.stdout)
        if cuda_pattern:
            version_match = re.search(r"[0-9]+\.[0-9]+\.[0-9]+", cuda_pattern.group(0))
            return version_match.group(0) if version_match else None
    except Exception:
        pass
    return None


def setup_cuda_dependencies(dry_run: bool = False) -> None:
    """Setup CUDA dependencies if CUDA runtime is available"""
    cuda_runtime_version = get_cuda_runtime_version()
    if cuda_runtime_version:
        cuda_major_version = cuda_runtime_version.split(".")[0]
        setup_cuda_packages(cuda_major_version, dry_run)
    else:
        info("CUDA Runtime package not found, skipping CUDA package installation")


def setup_cuda_packages(cuda_major_version: str, dry_run: bool = False) -> None:
    """Install CUDA packages for Holoscan SDK development"""

    # Attempt to install cudnn9
    CUDNN_9_PATTERN = r"9\.[0-9]+\.[0-9]+\.[0-9]+\-[0-9]+"
    try:
        installed_cudnn9_version = install_cuda_dependencies_package(
            package_name=f"libcudnn9-cuda-{cuda_major_version}",
            version_pattern=CUDNN_9_PATTERN,
            dry_run=dry_run,
        )
        install_cuda_dependencies_package(
            package_name=f"libcudnn9-dev-cuda-{cuda_major_version}",
            version_pattern=re.escape(installed_cudnn9_version),
            dry_run=dry_run,
        )
    except PackageInstallationError as e:
        info(f"cuDNN 9.x installation failed, falling back to cuDNN 8.x: {e}")
        try:
            # Fall back to cudnn8
            CUDNN_8_PATTERN = rf"8\.[0-9]+\.[0-9]+\.[0-9]+\-[0-9]\+cuda{cuda_major_version}\.[0-9]+"
            installed_cudnn8_version = install_cuda_dependencies_package(
                package_name="libcudnn8",
                version_pattern=CUDNN_8_PATTERN,
                dry_run=dry_run,
            )
            install_cuda_dependencies_package(
                package_name="libcudnn8-dev",
                version_pattern=re.escape(installed_cudnn8_version),
                dry_run=dry_run,
            )
        except PackageInstallationError as e:
            info(f"cuDNN 8.x installation failed: {e}.")
            info("cuDNN packages may need to be installed manually.")

    # Install TensorRT dependencies
    NVINFER_PATTERN = rf"\d+\.[0-9]+\.[0-9]+\.[0-9]+-[0-9]\+cuda{cuda_major_version}\.[0-9]+"
    try:

        installed_libnvinferversion = install_cuda_dependencies_package(
            package_name="libnvinfer10",
            version_pattern=NVINFER_PATTERN,
            dry_run=dry_run,
        )
        libnvinfer_pattern = re.escape(installed_libnvinferversion)

        install_packages_if_missing(
            [
                f"libnvinfer-bin={installed_libnvinferversion}",
                f"libnvinfer-lean10={installed_libnvinferversion}",
                f"libnvinfer-plugin10={installed_libnvinferversion}",
                f"libnvinfer-vc-plugin10={installed_libnvinferversion}",
                f"libnvinfer-dispatch10={installed_libnvinferversion}",
                f"libnvonnxparsers10={installed_libnvinferversion}",
            ],
            apt_options=["--no-install-recommends", "-y", "--allow-downgrades"],
            dry_run=dry_run,
        )

        for trt_package_name in [
            "libnvinfer-headers-dev",
            "libnvinfer-dev",
            "libnvinfer-headers-plugin-dev",
            "libnvinfer-plugin-dev",
            "libnvonnxparsers-dev",
        ]:
            install_cuda_dependencies_package(
                package_name=trt_package_name,
                version_pattern=libnvinfer_pattern,
                dry_run=dry_run,
            )
    except PackageInstallationError as e:
        info(f"TensorRT installation failed: {e}")
        info("Continuing with setup - TensorRT packages may need to be installed manually")
