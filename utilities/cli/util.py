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
import os
import re
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

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


# Utility Functions
def get_timestamp() -> str:
    """Get current timestamp in the format used by the bash script"""
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def format_cmd(command: str, is_dryrun: bool = False) -> str:
    """Format command output with consistent timestamp and color formatting"""
    timestamp = Color.blue(get_timestamp())
    if is_dryrun:
        dryrun_tag = Color.cyan("[dryrun]")
        return f"{timestamp} {dryrun_tag} {Color.white('$')} {Color.green(command)}"
    return f"{timestamp} {Color.white('$')} {Color.green(command)}"


def fatal(message: str) -> None:
    """Print fatal error and exit with backtrace"""
    print(
        f"{Color.red(get_timestamp())} {Color.red('[FATAL]', bold=True)} {message}", file=sys.stderr
    )
    print("\nBacktrace: ...", file=sys.stderr)
    traceback.print_list(traceback.extract_stack()[-3:], file=sys.stderr)
    sys.exit(1)


def run_command(
    cmd: Union[str, List[str]], dry_run: bool = False, check: bool = True, **kwargs
) -> subprocess.CompletedProcess:
    """Run a command and handle errors"""
    if isinstance(cmd, str):
        cmd_str = cmd
    else:
        cmd_list = [f'"{x}"' if " " in str(x) else str(x) for x in cmd]
        cmd_str = format_long_command(cmd_list) if dry_run else " ".join(cmd_list)
    if dry_run:
        print(format_cmd(cmd_str, is_dryrun=True))
        return subprocess.CompletedProcess(cmd_str, 0)

    print(format_cmd(cmd_str))
    try:
        return subprocess.run(cmd, check=check, **kwargs)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd_str}")
        print(f"Exit code: {e.returncode}")
        sys.exit(e.returncode)


def check_nvidia_ctk(min_version: str = "1.12.0", recommended_version: str = "1.14.1") -> None:
    """Check NVIDIA Container Toolkit version"""

    if not shutil.which("nvidia-ctk"):
        fatal("nvidia-ctk not found. Please install the NVIDIA Container Toolkit.")

    try:
        output = subprocess.check_output(["nvidia-ctk", "--version"], text=True)
        import re

        match = re.search(r"(\d+\.\d+\.\d+)", output)
        if match:
            version = match.group(1)

            try:
                from packaging import version as ver

                version_check = ver.parse(version) < ver.parse(min_version)
            except ImportError:

                def parse_version(v):
                    try:
                        return tuple(map(int, v.split(".")))
                    except ValueError:
                        return (10, 0, 0)

                version_check = parse_version(version) < parse_version(min_version)

            if version_check:
                fatal(
                    f"Found nvidia-ctk {version}. Version {min_version}+ is required "
                    f"({recommended_version}+ recommended)."
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
    if not isinstance(language, str):
        return None
    if language.lower() == "cpp" or language.lower() == "c++":
        return "cpp"
    elif language.lower() == "python" or language.lower() == "py":
        return "python"
    return None


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
