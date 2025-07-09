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
import json
import os
import platform
import re
import shlex
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


class PackageInstallationError(Exception):
    """Raised when a package cannot be installed via apt"""

    def __init__(self, package_name: str, version_pattern: str, message: str = None):
        self.package_name = package_name
        self.version_pattern = version_pattern
        super().__init__(
            message or f"Failed to install package {package_name} matching {version_pattern}"
        )


def install_cuda_dependencies_package(
    package_name: str,
    version_pattern: str = r"\d+\.\d+\.\d+",
    dry_run: bool = False,
) -> str:
    """Install CUDA dependencies package with version checking

    Procedure:
    1. If package is already installed, return the version
    2. If the package is not installed and the preferred version is available, install it and return the version.
    3. If the package is not installed and the preferred version is not available, throw.

    Args:
        package_name: Name of the package to install
        version_pattern: Regular expression for package version to get if not already installed.
            The latest version matching the pattern will be installed.

    Returns:
        str: Installed package version

    Raises:
        PackageInstallationError: If package cannot be installed
    """

    # Check if package is already installed
    try:
        output = subprocess.check_output(
            ["apt", "list", "--installed", package_name], text=True, stderr=subprocess.DEVNULL
        ).split()
        # Extract installed version from apt list output using regex
        # Example output: "libcudnn9-cuda-12/unknown,now 9.5.1.17-1 amd64 [installed,upgradable to: 9.8.0.87-1]"
        if len(output) >= 3 and re.match(r"\d+\.\d+\.\d+.*", output[2]):
            installed_version = output[2]
            info(f"Package {package_name} found with version {installed_version}")
            return installed_version
    except subprocess.CalledProcessError:
        # Package not installed, continue to attempt installation
        pass

    # Check available versions
    try:
        # apt list -a sorts in descending order by default
        available_versions = subprocess.check_output(
            ["apt", "list", "-a", package_name], text=True, stderr=subprocess.DEVNULL
        )
        matching_version = re.findall(
            f"^{re.escape(package_name)}/.*?({version_pattern}).*$",
            available_versions,
            re.MULTILINE,
        )
        matching_version = matching_version[0] if matching_version else None

        if not matching_version:
            raise PackageInstallationError(
                package_name,
                version_pattern,
                f"{package_name} is not installable with pattern {version_pattern}.\n"
                f"You might want to try to install a newer version manually and rerun the setup:\n"
                f"  sudo apt install {package_name}",
            )

        # Install the package
        info(f"Installing {package_name}={matching_version}")
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
        return matching_version

    except subprocess.CalledProcessError as e:
        raise PackageInstallationError(
            package_name,
            version_pattern,
            f"Error checking available versions for {package_name}: {e}",
        )


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


def build_holohub_path_mapping(
    holohub_root: Path,
    project_data: Optional[dict] = None,
    build_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
) -> dict[str, str]:
    """Build a mapping of HoloHub placeholders to their resolved paths"""
    if data_dir is None:
        data_dir = holohub_root / "data"

    path_mapping = {
        "holohub_root": str(holohub_root),
        "holohub_data_dir": str(data_dir),
    }
    if not project_data:
        return path_mapping
    # Add project-specific mappings if project_data is provided
    app_source_path = project_data.get("source_folder", "")
    if app_source_path:
        path_mapping["holohub_app_source"] = str(app_source_path)
    if build_dir:
        path_mapping["holohub_bin"] = str(build_dir)
        if app_source_path:
            try:
                app_build_dir = build_dir / Path(app_source_path).relative_to(holohub_root)
                path_mapping["holohub_app_bin"] = str(app_build_dir)
            except ValueError:
                # Handle case where app_source_path is not relative to holohub_root
                path_mapping["holohub_app_bin"] = str(build_dir)
    elif project_data.get("project_name"):
        # If no build_dir provided but we have project name, try to infer it
        project_name = project_data["project_name"]
        inferred_build_dir = holohub_root / "build" / project_name
        path_mapping["holohub_bin"] = str(inferred_build_dir)
        if app_source_path:
            try:
                app_build_dir = inferred_build_dir / Path(app_source_path).relative_to(holohub_root)
                path_mapping["holohub_app_bin"] = str(app_build_dir)
            except ValueError:
                path_mapping["holohub_app_bin"] = str(inferred_build_dir)
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
            "Inspect docker image entrypoint: "
            f"docker inspect --format={{{{json .Config.Entrypoint}}}} {img}"
        )
        return None

    try:
        result = run_command(
            ["docker", "inspect", "--format={{json .Config.Entrypoint}}", img],
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
        result = run_command(
            ["docker", "inspect", "--format", "{{range .Config.Env}}{{println .}}{{end}}", img],
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
    docker_version = run_info_command(["docker", "--version"])
    docker_info = run_info_command(["docker", "info", "--format", "{{.ServerVersion}}"])
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
        "HOLOHUB_BASE_IMAGE",
        "HOLOHUB_APP_NAME",
        "HOLOHUB_CONTAINER_BASE_NAME",
        "HOLOHUB_ALWAYS_BUILD",
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
