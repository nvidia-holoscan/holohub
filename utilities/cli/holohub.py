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

import sys

# Python version check - must be before other imports that use Python 3.10+ features
PYTHON_MIN_VERSION = (3, 10, 0)
if sys.version_info < PYTHON_MIN_VERSION:
    sys_major, sys_minor, sys_micro = sys.version_info[:3]
    print(
        f"Error: Python {'.'.join(map(str, PYTHON_MIN_VERSION))} or higher required, "
        f"found {sys_major}.{sys_minor}.{sys_micro}",
        file=sys.stderr,
    )
    sys.exit(1)

# ruff: noqa: E402  # Imports after python version check
import argparse
import datetime
import filecmp
import os
import shlex
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import utilities.cli.util as holohub_cli_util
import utilities.metadata.gather_metadata as metadata_util
from utilities.cli.container import HoloHubContainer
from utilities.cli.util import Color


class HoloHubCLI:
    """Command-line interface for HoloHub"""

    HOLOHUB_ROOT = holohub_cli_util.get_holohub_root()
    DEFAULT_BUILD_PARENT_DIR = Path(
        os.environ.get("HOLOHUB_BUILD_PARENT_DIR", HOLOHUB_ROOT / "build")
    )
    DEFAULT_DATA_DIR = Path(os.environ.get("HOLOHUB_DATA_DIR", HOLOHUB_ROOT / "data"))
    DEFAULT_SDK_DIR = os.environ.get("HOLOHUB_DEFAULT_HSDK_DIR", "/opt/nvidia/holoscan/lib")
    # Allow overriding the default CTest script path via environment variable
    DEFAULT_CTEST_SCRIPT = os.environ.get(
        "HOLOHUB_CTEST_SCRIPT", "utilities/testing/holohub.container.ctest"
    )

    def __init__(self):
        self.script_name = os.environ.get("HOLOHUB_CMD_NAME", "./holohub")
        self.parser = self._create_parser()
        # Cache for resolved projects to avoid duplicate lookups
        self._project_data: dict[tuple[str, str], dict] = {}
        self._collect_metadata()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser with all supported commands"""
        parser = argparse.ArgumentParser(
            prog=self.script_name,
            description=f"{self.script_name} CLI tool for managing Holoscan-based applications and containers",
        )
        subparsers = parser.add_subparsers(dest="command", required=True)

        # Store subparsers for error handling
        self.subparsers = {}

        # Common container arguments parent parsers
        container_build_argparse = HoloHubContainer.get_build_argparse()
        container_run_argparse = HoloHubContainer.get_run_argparse()
        # Add create command
        create = subparsers.add_parser("create", help="Create a new Holoscan application")
        self.subparsers["create"] = create
        create.add_argument("project", help="Name of the project to create")
        create.add_argument(
            "--template",
            default=str(HoloHubCLI.HOLOHUB_ROOT / "applications" / "template"),
            help="Path to the template directory to use",
        )
        create.add_argument(
            "--language",
            choices=["cpp", "python"],
            default="cpp",
            help="Programming language for the project",
        )
        create.add_argument(
            "--dryrun", action="store_true", help="Print commands without executing them"
        )
        create.add_argument(
            "--directory",
            type=Path,
            default=self.HOLOHUB_ROOT / "applications",
            help="Path to the directory to create the project in",
        )
        create.add_argument(
            "--context",
            action="append",
            help='Additional context variables for cookiecutter in format key=value. \
                Example: --context description=\'My project desc\' \
                    --context tags=[\\"tag1\\", \\"tag2\\"]',
        )
        create.add_argument(
            "-i",
            "--interactive",
            action="store",
            nargs="?",
            const=True,
            default=True,
            type=lambda x: x.lower() not in ("false", "no", "n", "0", "f"),
            help="Interactive mode for setting cookiecutter properties (use -i False to disable)",
        )
        create.set_defaults(func=self.handle_create)

        # build-container command
        build_container = subparsers.add_parser(
            "build-container",
            help="Build the development container",
            parents=[container_build_argparse],
        )
        self.subparsers["build-container"] = build_container
        build_container.add_argument("project", nargs="?", help="Project to build container for")
        build_container.add_argument(
            "--verbose", action="store_true", help="Print variables passed to docker build command"
        )
        build_container.add_argument(
            "--dryrun", action="store_true", help="Print commands without executing them"
        )
        build_container.add_argument(
            "--language", choices=["cpp", "python"], help="Specify language implementation"
        )
        build_container.set_defaults(func=self.handle_build_container)

        # run-container command
        run_container = subparsers.add_parser(
            "run-container",
            help="Build and launch the development container",
            parents=[container_build_argparse, container_run_argparse],
            epilog="Any trailing arguments after ' -- ' are forwarded to 'docker run'",
        )
        self.subparsers["run-container"] = run_container
        run_container.add_argument("project", nargs="?", help="Project to run container for")
        run_container.add_argument(
            "--verbose", action="store_true", help="Print variables passed to docker run command"
        )
        run_container.add_argument(
            "--dryrun", action="store_true", help="Print commands without executing them"
        )
        run_container.add_argument(
            "--language", choices=["cpp", "python"], help="Specify language implementation"
        )
        run_container.add_argument(
            "--no-docker-build", action="store_true", help="Skip building the container"
        )
        run_container.set_defaults(func=self.handle_run_container)

        # build command
        build = subparsers.add_parser(
            "build",
            help="Build a project",
            parents=[container_build_argparse, container_run_argparse],
        )
        self.subparsers["build"] = build
        build.add_argument("project", help="Project to build")
        build.add_argument("mode", nargs="?", help="Mode to build (optional)")
        build.add_argument(
            "--local", action="store_true", help="Build locally instead of in container"
        )
        build.add_argument("--verbose", action="store_true", help="Print extra output")
        build.add_argument(
            "--build-type",
            help="Build type (debug, release, rel-debug). "
            "If not specified, uses CMAKE_BUILD_TYPE environment variable or defaults to 'release'",
        )
        build.add_argument(
            "--build-with",
            dest="with_operators",
            help="Optional operators that should be built, separated by semicolons (;)",
        )
        build.add_argument(
            "--dryrun", action="store_true", help="Print commands without executing them"
        )
        build.add_argument(
            "--pkg-generator", default="DEB", help="Package generator for cpack (default: DEB)"
        )
        build.add_argument(
            "--parallel", help="Number of parallel build jobs (e.g. --parallel $(($(nproc)-1)))"
        )
        build.add_argument(
            "--language", choices=["cpp", "python"], help="Specify language implementation"
        )
        build.add_argument(
            "--benchmark",
            action="store_true",
            help="Build for Holoscan Flow Benchmarking. Valid for applications/workflows only",
        )
        build.add_argument(
            "--no-docker-build", action="store_true", help="Skip building the container"
        )
        build.add_argument(
            "--configure-args",
            action="append",
            help="Additional configuration arguments for cmake "
            "example: --configure-args='-DCUSTOM_OPTION=ON' --configure-args='-Dtest=ON'",
        )
        build.set_defaults(func=self.handle_build)

        # run command
        run = subparsers.add_parser(
            "run",
            help="Build and run a project",
            parents=[container_build_argparse, container_run_argparse],
        )
        self.subparsers["run"] = run
        run.add_argument("project", help="Project to run")
        run.add_argument("mode", nargs="?", help="Mode to run (optional)")
        run.add_argument("--local", action="store_true", help="Run locally instead of in container")
        run.add_argument("--verbose", action="store_true", help="Print extra output")
        run.add_argument(
            "--dryrun", action="store_true", help="Print commands without executing them"
        )
        run.add_argument(
            "--language", choices=["cpp", "python"], help="Specify language implementation"
        )
        run.add_argument(
            "--build-type",
            help="Build type (debug, release, rel-debug). "
            "If not specified, uses CMAKE_BUILD_TYPE environment variable or defaults to 'release'",
        )
        run.add_argument(
            "--run-args",
            help="Additional arguments to pass to the application executable, "
            "example: --run-args=--flag or --run-args '-c config/file'",
        )

        run.add_argument(
            "--build-with",
            dest="with_operators",
            help="Optional operators that should be built, separated by semicolons (;)",
        )
        run.add_argument(
            "--parallel", help="Number of parallel build jobs (e.g. --parallel $(($(nproc)-1)))"
        )
        run.add_argument(
            "--pkg-generator", default="DEB", help="Package generator for cpack (default: DEB)"
        )
        run.add_argument(
            "--no-local-build",
            action="store_true",
            help="Skip building and just run the application",
        )
        run.add_argument(
            "--no-docker-build", action="store_true", help="Skip building the container"
        )
        run.add_argument(
            "--configure-args",
            action="append",
            help="Additional configuration arguments for cmake "
            "example: --configure-args='-DCUSTOM_OPTION=ON' --configure-args='-Dtest=ON'",
        )
        run.set_defaults(func=self.handle_run)

        # list command
        list_cmd = subparsers.add_parser("list", help="List all available targets")
        self.subparsers["list"] = list_cmd
        list_cmd.set_defaults(func=self.handle_list)

        # modes command
        modes = subparsers.add_parser("modes", help="List available modes for an application")
        self.subparsers["modes"] = modes
        modes.add_argument("project", help="Project to list modes for")
        modes.add_argument(
            "--language", choices=["cpp", "python"], help="Specify language implementation"
        )
        modes.set_defaults(func=self.handle_modes)

        # autocompletion_list command (for bash completion)
        autocomp_cmd = subparsers.add_parser(
            "autocompletion_list", help="List targets for autocompletion"
        )
        self.subparsers["autocompletion_list"] = autocomp_cmd
        autocomp_cmd.set_defaults(func=self.handle_autocompletion_list)

        # lint command
        lint = subparsers.add_parser("lint", help="Run linting tools")
        self.subparsers["lint"] = lint
        lint.add_argument("path", nargs="?", default=".", help="Path to lint")
        lint.add_argument("--fix", action="store_true", help="Fix linting issues")
        lint.add_argument(
            "--install-dependencies",
            action="store_true",
            help="Install linting dependencies (may require `sudo` privileges)",
        )
        lint.add_argument(
            "--dryrun", action="store_true", help="Print commands without executing them"
        )
        lint.set_defaults(func=self.handle_lint)

        # setup command
        setup = subparsers.add_parser(
            "setup", help="Install HoloHub recommended packages for development."
        )
        self.subparsers["setup"] = setup
        setup.add_argument(
            "--dryrun", action="store_true", help="Print commands without executing them"
        )
        setup.add_argument(
            "--list-scripts",
            action="store_true",
            help="List all setup scripts found in the HOLOHUB_SETUP_SCRIPTS_DIR directory. "
            + "Run scripts directly or with `./holohub setup --scripts <script_name>`.",
        )
        setup.add_argument(
            "--scripts",
            action="append",
            help="Named dependency installation scripts to run. Can be specified multiple times. "
            + "Searches in the directory path specified by the HOLOHUB_SETUP_SCRIPTS_DIR environment variable. "
            + "Omit to install default recommended packages for Holoscan SDK development.",
        )
        setup.set_defaults(func=self.handle_setup)

        # Add env-info command
        env_info = subparsers.add_parser(
            "env-info", help="Display environment debugging information"
        )
        self.subparsers["env-info"] = env_info
        env_info.set_defaults(func=self.handle_env_info)

        # Add install command
        install = subparsers.add_parser(
            "install",
            help="Install a project",
            parents=[container_build_argparse, container_run_argparse],
        )
        self.subparsers["install"] = install
        install.add_argument("project", help="Project to install")
        install.add_argument("mode", nargs="?", help="Mode to install (optional)")
        install.add_argument(
            "--local", action="store_true", help="Install locally instead of in container"
        )
        install.add_argument(
            "--build-type",
            help="Build type (debug, release, rel-debug). "
            "If not specified, uses CMAKE_BUILD_TYPE environment variable or defaults to 'release'",
        )
        install.add_argument(
            "--language", choices=["cpp", "python"], help="Specify language implementation"
        )
        install.add_argument(
            "--build-with",
            dest="with_operators",
            help="Optional operators that should be built, separated by semicolons (;)",
        )
        install.add_argument("--verbose", action="store_true", help="Print extra output")
        install.add_argument(
            "--dryrun", action="store_true", help="Print commands without executing them"
        )
        install.add_argument(
            "--parallel", help="Number of parallel build jobs (e.g. --parallel $(($(nproc)-1)))"
        )
        install.add_argument(
            "--no-docker-build", action="store_true", help="Skip building the container"
        )
        install.add_argument(
            "--configure-args",
            action="append",
            help="Additional configuration arguments for cmake "
            "example: --configure-args='-DCUSTOM_OPTION=ON' --configure-args='-Dtest=ON'",
        )
        install.set_defaults(func=self.handle_install)

        # Add test command
        test = subparsers.add_parser(
            "test", help="Test a project", parents=[container_build_argparse]
        )
        self.subparsers["test"] = test
        test.add_argument("project", nargs="?", help="Project to test")
        test.add_argument("--verbose", action="store_true", help="Print extra output")
        test.add_argument(
            "--dryrun", action="store_true", help="Print commands without executing them"
        )
        test.add_argument("--clear-cache", action="store_true", help="Clear cache folders")
        test.add_argument("--site-name", help="Site name")
        test.add_argument("--cdash-url", help="CDash URL")
        test.add_argument("--platform-name", help="Platform name")
        test.add_argument(
            "--cmake-options",
            action="append",
            help="CMake options, "
            "example: --cmake-options='-DCUSTOM_OPTION=ON' --cmake-options='-DDEBUG_MODE=1'",
        )
        test.add_argument(
            "--ctest-options",
            action="append",
            help="CTest options, "
            "example: --ctest-options='-DGPU_TYPE=rtx4090' --ctest-options='-DDEBUG_MODE=ON'",
        )
        test.add_argument("--no-xvfb", action="store_true", help="Do not use xvfb")
        test.add_argument("--ctest-script", help="CTest script")
        test.add_argument(
            "--no-docker-build", action="store_true", help="Skip building the container"
        )
        test.add_argument(
            "--build-name-suffix",
            help="Suffix to use for ctest build name (defaulting to the image tag)",
        )
        test.set_defaults(func=self.handle_test)

        # Add clear-cache command
        clear_cache = subparsers.add_parser("clear-cache", help="Clear cache folders")
        self.subparsers["clear-cache"] = clear_cache
        clear_cache.add_argument(
            "--dryrun", action="store_true", help="Print commands without executing them"
        )
        clear_cache.set_defaults(func=self.handle_clear_cache)

        # Add vscode command
        vscode = subparsers.add_parser(
            "vscode",
            help="Launch VS Code in Dev Container",
            parents=[container_build_argparse],
        )
        self.subparsers["vscode"] = vscode
        vscode.add_argument("project", nargs="?", help="Project to launch VS Code for")
        vscode.add_argument(
            "--language", choices=["cpp", "python"], help="Specify language implementation"
        )
        vscode.add_argument("--docker-opts", help="Additional options to pass to the Docker launch")
        vscode.add_argument(
            "--verbose", action="store_true", help="Print variables passed to docker run command"
        )
        vscode.add_argument(
            "--dryrun", action="store_true", help="Print commands without executing them"
        )
        vscode.add_argument(
            "--no-docker-build", action="store_true", help="Skip building the container"
        )
        vscode.set_defaults(func=self.handle_vscode)

        return parser

    def _collect_metadata(self) -> None:
        """Create an unstructured database of metadata for all projects"""

        EXCLUDE_PATHS = ["applications/holoviz/template", "applications/template"]
        # Known exceptions, such as template files that do not represent a standalone project

        app_paths = (
            HoloHubCLI.HOLOHUB_ROOT / "applications",
            HoloHubCLI.HOLOHUB_ROOT / "benchmarks",
            HoloHubCLI.HOLOHUB_ROOT / "gxf_extensions",
            HoloHubCLI.HOLOHUB_ROOT / "operators",
            HoloHubCLI.HOLOHUB_ROOT / "pkg",
            HoloHubCLI.HOLOHUB_ROOT / "workflows",
        )
        self.projects = metadata_util.gather_metadata(app_paths, exclude_paths=EXCLUDE_PATHS)

    def find_project(self, project_name: str, language: Optional[str] = None) -> dict:
        """Find a project by name"""
        normalized_language = holohub_cli_util.normalize_language(language)

        cache_key = (project_name, normalized_language)
        if cache_key in self._project_data:
            return self._project_data[cache_key]

        # Find all projects with the given name
        candidates = [p for p in self.projects if p.get("project_name") == project_name]
        if candidates:
            available_lang = []
            for p in candidates:
                for lang in holohub_cli_util.list_normalized_languages(
                    p.get("metadata", {}).get("language", None)
                ):
                    available_lang.append(lang)
            available_lang = sorted(list(set(available_lang)))

            # Determine target language (if unspecified, prefer cpp then first available)
            if normalized_language:
                target_lang = normalized_language
            elif "python" in available_lang:
                target_lang = "python"
            else:
                target_lang = available_lang[0] if available_lang else ""
            # Warn if ambiguous and no language specified
            if not normalized_language and len(available_lang) > 1:
                msg = f"'{project_name}' has multiple languages: {', '.join(available_lang)}.\n"
                msg += f"Defaulting to '{target_lang}'. Use --language to select explicitly.\n\n"
                print(Color.green(msg))
            for p in candidates:
                if target_lang in holohub_cli_util.list_normalized_languages(
                    p.get("metadata", {}).get("language", None)
                ):
                    self._project_data[cache_key] = p  # Return candidate matching target_lang
                    return p
            if normalized_language:  # If target_lang specified but not found
                holohub_cli_util.fatal(
                    f"Project '{project_name}' (language: {normalized_language}) not found. "
                    f"Available: {', '.join(available_lang) if available_lang else 'unknown'}"
                )
            # No language info or no match found; return first candidate
            fallback_candidate = candidates[0]
            fallback_lang = fallback_candidate.get("metadata", {}).get("language", None)
            if not fallback_lang:
                msg = f"Returning '{project_name}' with missing or unknown language metadata.\n"
                msg += "Consider specifying --language for more consistent results.\n"
                holohub_cli_util.warn(msg)
            self._project_data[cache_key] = fallback_candidate
            return self._project_data[cache_key]
        # If project not found, suggest similar names
        distances = [
            (
                p["project_name"],
                holohub_cli_util.levenshtein_distance(project_name, p["project_name"]),
                p.get("source_folder", ""),
                p.get("metadata", {}).get("language", ""),
            )
            for p in self.projects
        ]
        distances.sort(key=lambda x: x[1])  # Sort by distance
        closest_matches = [
            (name, folder, lang) for name, dist, folder, lang in distances[:3] if dist <= 3
        ]  # Show up to 3 matches
        msg = f"Project '{project_name}' (language: {normalized_language}) not found."
        if closest_matches:
            msg += "\nDid you mean:"
            for name, folder, lang in closest_matches:
                details = []
                if lang:
                    details.append(f"language: {lang}")
                if folder:
                    details.append(f"source: {folder}")
                msg += f"\n  '{name}'" + (f" ({', '.join(details)})" if details else "")
        holohub_cli_util.fatal(msg)
        return None

    def resolve_mode(self, project_data: dict, requested_mode: Optional[str] = None) -> tuple:
        """
        Resolve mode from metadata and validate
        Returns: (mode_name, mode_config) or (None, None) for legacy behavior
        """
        modes = project_data.get("metadata", {}).get("modes", {})
        if not modes:
            return None, None  # No modes defined - should use legacy behavior

        if requested_mode is None:
            # Validate that multiple modes have a default_mode specified
            application_metadata = project_data.get("metadata", {})
            if len(modes) > 1 and "default_mode" not in application_metadata:
                available = ", ".join(modes.keys())
                holohub_cli_util.fatal(
                    f"Multiple modes found ({available}) but no 'default_mode' specified. "
                    f"Please add a 'default_mode' field to specify which mode to use by default."
                )

            if "default_mode" in application_metadata:
                requested_mode = application_metadata["default_mode"]
                # Validate that default_mode references an existing mode
                if requested_mode not in modes:
                    available = ", ".join(modes.keys())
                    holohub_cli_util.fatal(
                        f"Invalid default_mode '{requested_mode}' in metadata among {available}"
                    )
            else:
                requested_mode = list(modes.keys())[0]
        if requested_mode not in modes:
            available = ", ".join(modes.keys())
            holohub_cli_util.fatal(
                f"Mode '{requested_mode}' not found. Available modes: {available}"
            )
        return requested_mode, modes[requested_mode]

    def _is_implicit_default(self, project_data: dict, user_requested_mode: Optional[str]) -> bool:
        """
        Check if we're using a default mode without explicitly requesting it.
        This enables the implicit default mode behavior where CLI args are allowed.
        """
        if user_requested_mode is not None:
            return False  # Mode was explicitly requested by user
        return bool(project_data.get("metadata", {}).get("modes", {}))  # has default mode

    def validate_mode(
        self,
        args: argparse.Namespace,
        mode_name: Optional[str],
        mode_config: dict,
        project_data: dict,
        requested_mode: Optional[str],
    ) -> None:
        """Validate that when mode is specified, no conflicting CLI parameters are provided"""
        if not mode_name or not mode_config:
            return  # No mode specified, allow CLI parameters

        # Check if this is an implicit default mode selection - if so, allow all CLI parameter overrides
        if self._is_implicit_default(project_data, requested_mode):
            # For implicit default modes, allow all CLI parameter overrides
            # This enables the use case: ./holohub run app --run-args="..." --build-with="..." etc.
            return

        conflicting_params = []

        # Check build-related parameters
        if "build" in mode_config:
            build_config = mode_config["build"]
            if "depends" in build_config and getattr(args, "with_operators", None):
                conflicting_params.append("--build-with")
            if "docker_build_args" in build_config and getattr(args, "build_args", None):
                conflicting_params.append("--build-args")
            if "cmake_options" in build_config and getattr(args, "configure_args", None):
                conflicting_params.append("--configure-args")

        # Check run-related parameters
        if "run" in mode_config:
            run_config = mode_config["run"]
            if "docker_run_args" in run_config and getattr(args, "docker_opts", None):
                conflicting_params.append("--docker-opts")
            if "command" in run_config and getattr(args, "run_args", None):
                conflicting_params.append("--run-args")

        if conflicting_params:
            params_str = ", ".join(conflicting_params)
            holohub_cli_util.fatal(
                f"Cannot specify CLI parameters {params_str} when using explicit mode '{mode_name}'. "
                f"All configuration must be provided through the mode definition.\n"
                f"See {os.environ.get('HOLOHUB_CLI_DOCS_URL', 'https://github.com/nvidia-holoscan/holohub/blob/main/utilities/cli/README.md')}"
            )

    def get_effective_build_config(
        self,
        args: argparse.Namespace,
        mode_config: dict,
        project_data: dict = None,
        requested_mode: Optional[str] = None,
    ) -> dict:
        """Get effective build configuration combining CLI args and mode config without mutation"""
        config = {
            "with_operators": getattr(args, "with_operators", None),
            "docker_opts": getattr(args, "docker_opts", ""),
            "build_args": getattr(args, "build_args", ""),
            "configure_args": getattr(args, "configure_args", None),
        }
        if not mode_config:
            return config

        # Check if this is an implicit default mode - if so, CLI args take precedence
        is_implicit_default = project_data and self._is_implicit_default(
            project_data, requested_mode
        )

        # Apply build configuration (mode values override CLI values unless it's an implicit default)
        if "build" in mode_config:
            build_config = mode_config["build"]

            # For depends/with_operators: use CLI if provided and implicit default, otherwise use mode
            if "depends" in build_config:
                if is_implicit_default and config["with_operators"]:
                    mode_deps = [dep.strip() for dep in build_config["depends"] if dep.strip()]
                    msg = f"CLI args --build-with='{config['with_operators']}' "
                    msg += f"overrides mode depends: {', '.join(mode_deps)}"
                    holohub_cli_util.warn(msg)
                else:
                    mode_deps = [dep.strip() for dep in build_config["depends"] if dep.strip()]
                    config["with_operators"] = ";".join(mode_deps) if mode_deps else ""

            # For docker_build_args: use CLI if provided and implicit default, otherwise use mode
            if "docker_build_args" in build_config:
                if is_implicit_default and config["build_args"]:
                    mode_args = holohub_cli_util.normalize_args_str(
                        build_config["docker_build_args"]
                    )
                    msg = f"CLI args --build-args='{config['build_args']}' "
                    msg += f"overrides mode --build-args: {mode_args}"
                    holohub_cli_util.warn(msg)
                else:
                    config["build_args"] = holohub_cli_util.normalize_args_str(
                        build_config["docker_build_args"]
                    )

            # For cmake_options: use CLI if provided and implicit default, otherwise use mode
            if "cmake_options" in build_config:
                if is_implicit_default and config["configure_args"]:
                    mode_opts = (
                        " ".join(build_config["cmake_options"])
                        if isinstance(build_config["cmake_options"], list)
                        else build_config["cmake_options"]
                    )
                    cli_opts = (
                        " ".join(config["configure_args"])
                        if isinstance(config["configure_args"], list)
                        else config["configure_args"]
                    )
                    msg = f"CLI args --configure-args='{cli_opts}' "
                    msg += f"overrides mode --configure-args: {mode_opts}"
                    holohub_cli_util.warn(msg)
                else:
                    config["configure_args"] = build_config["cmake_options"]

        # Apply run.docker_run_args for build container (Docker run arguments)
        if "run" in mode_config and "docker_run_args" in mode_config["run"]:
            # For docker_opts: use CLI if provided and implicit default, otherwise use mode
            if is_implicit_default and getattr(args, "docker_opts", ""):
                mode_opts = holohub_cli_util.normalize_args_str(
                    mode_config["run"]["docker_run_args"]
                )
                msg = f"CLI args --docker-opts='{getattr(args, 'docker_opts', '')}' "
                msg += f"overrides mode --docker-opts: {mode_opts}"
                holohub_cli_util.warn(msg)
            else:
                config["docker_opts"] = holohub_cli_util.normalize_args_str(
                    mode_config["run"]["docker_run_args"]
                )

        return config

    def get_effective_run_config(
        self,
        args: argparse.Namespace,
        mode_config: dict,
        project_data: dict = None,
        requested_mode: Optional[str] = None,
    ) -> dict:
        """Get effective run configuration combining CLI args and mode config without mutation"""
        config = {
            "run_args": getattr(args, "run_args", "") or "",
            "docker_opts": getattr(args, "docker_opts", ""),
        }

        if mode_config and "run" in mode_config:
            run_config = mode_config["run"]

            # Check if this is an implicit default mode - if so, CLI args take precedence for docker_opts
            is_implicit_default = project_data and self._is_implicit_default(
                project_data, requested_mode
            )

            if "command" in run_config:
                config["command"] = run_config["command"]
            if "workdir" in run_config:
                config["workdir"] = run_config["workdir"]

            # For run_args: show warning if CLI overrides mode command in implicit default mode
            if "command" in run_config and is_implicit_default and getattr(args, "run_args", ""):
                msg = (
                    f"CLI args --run-args='{getattr(args, 'run_args', '')}' "
                    f"will be appended to mode command"
                )
                holohub_cli_util.warn(msg)

            # For docker_run_args: use CLI if provided and implicit default, otherwise use mode
            if "docker_run_args" in run_config:
                if is_implicit_default and getattr(args, "docker_opts", ""):
                    mode_opts = holohub_cli_util.normalize_args_str(run_config["docker_run_args"])
                    msg = (
                        f"CLI args --docker-opts='{getattr(args, 'docker_opts', '')}' "
                        f"overrides mode --docker-opts: {mode_opts}"
                    )
                    holohub_cli_util.warn(msg)
                else:
                    config["docker_opts"] = holohub_cli_util.normalize_args_str(
                        run_config["docker_run_args"]
                    )
        return config

    def _make_project_container(
        self, project_name: Optional[str] = None, language: Optional[str] = None
    ) -> HoloHubContainer:
        """Define a project container"""
        if not project_name:
            return HoloHubContainer(project_metadata=None)
        project_data = self.find_project(project_name=project_name, language=language)
        return HoloHubContainer(project_metadata=project_data, language=language)

    def handle_build_container(self, args: argparse.Namespace) -> None:
        """Handle build-container command"""
        container = self._make_project_container(
            project_name=args.project,
            language=args.language if hasattr(args, "language") else None,
        )
        container.dryrun = args.dryrun
        container.build(
            docker_file=args.docker_file,
            base_img=args.base_img,
            img=args.img,
            no_cache=args.no_cache,
            build_args=args.build_args,
            cuda_version=getattr(args, "cuda", None),
            extra_scripts=getattr(args, "extra_scripts", []),
        )

    def handle_run_container(self, args: argparse.Namespace) -> None:
        """Handle run-container command"""
        skip_docker_build, _ = holohub_cli_util.check_skip_builds(args)
        container = self._make_project_container(
            project_name=args.project, language=args.language if hasattr(args, "language") else None
        )
        container.dryrun = args.dryrun
        container.verbose = args.verbose
        if not skip_docker_build:
            container.build(
                docker_file=args.docker_file,
                base_img=args.base_img,
                img=args.img,
                no_cache=args.no_cache,
                build_args=args.build_args,
                cuda_version=getattr(args, "cuda", None),
                extra_scripts=getattr(args, "extra_scripts", []),
            )
        else:
            if hasattr(args, "cuda") and args.cuda is not None:
                container.cuda_version = args.cuda

        trailing_args = getattr(args, "_trailing_args", [])
        docker_opts = args.docker_opts
        if trailing_args:  # additional commands requires a bash entrypoint
            command = holohub_cli_util.normalize_args_str(trailing_args)
            docker_opts_extra, extra_args = holohub_cli_util.get_entrypoint_command_args(
                args.img or container.image_name, command, docker_opts, dry_run=args.dryrun
            )
            if docker_opts_extra:
                docker_opts = f"{docker_opts} {docker_opts_extra}".strip()
            trailing_args = extra_args

        container.run(
            img=args.img,
            local_sdk_root=args.local_sdk_root,
            enable_x11=getattr(args, "enable_x11", True),
            ssh_x11=getattr(args, "ssh_x11", False),
            use_tini=args.init,
            persistent=args.persistent,
            nsys_profile=getattr(args, "nsys_profile", False),
            nsys_location=getattr(args, "nsys_location", ""),
            as_root=args.as_root,
            docker_opts=docker_opts,
            add_volumes=args.add_volume,
            enable_mps=getattr(args, "mps", False),
            extra_args=trailing_args,
        )

    def handle_test(self, args: argparse.Namespace) -> None:
        """Handle test command"""
        skip_docker_build, _ = holohub_cli_util.check_skip_builds(args)
        container = self._make_project_container(
            project_name=args.project, language=args.language if hasattr(args, "language") else None
        )
        if args.clear_cache:
            self.handle_clear_cache(args)

        container.dryrun = args.dryrun
        container.verbose = args.verbose

        if not skip_docker_build:
            container.build(
                docker_file=args.docker_file,
                base_img=args.base_img,
                img=args.img,
                no_cache=args.no_cache,
                build_args=args.build_args,
                cuda_version=getattr(args, "cuda", None),
                extra_scripts=getattr(args, "extra_scripts", []),
            )
        else:
            if hasattr(args, "cuda") and args.cuda is not None:
                container.cuda_version = args.cuda

        xvfb = "" if args.no_xvfb else "xvfb-run -a"

        # TAG is used in CTest scripts by default
        if getattr(args, "build_name_suffix", None):
            tag = args.build_name_suffix
        else:
            if skip_docker_build:
                image_name = getattr(args, "img", None) or container.image_name
            else:
                image_name = args.base_img or container.default_base_image()
            tag = image_name.split(":")[-1]

        ctest_cmd = f"{xvfb} ctest "
        if args.project:
            ctest_cmd += f"-DAPP={args.project} "
        ctest_cmd += f"-DTAG={tag} "

        if args.cmake_options:
            cmake_opts = ";".join(args.cmake_options)
            ctest_cmd += f'-DCONFIGURE_OPTIONS="{cmake_opts}" '

        if getattr(args, "ctest_options", None):
            ctest_cmd += " ".join(args.ctest_options) + " "

        if args.cdash_url:
            ctest_cmd += f"-DCTEST_SUBMIT_URL={args.cdash_url} "

        if args.site_name:
            ctest_cmd += f"-DCTEST_SITE={args.site_name} "

        if args.platform_name:
            ctest_cmd += f"-DPLATFORM_NAME={args.platform_name} "

        if args.ctest_script:
            ctest_cmd += f"-S {args.ctest_script} "
        else:
            ctest_cmd += f"-S {self.DEFAULT_CTEST_SCRIPT} "

        if args.verbose:
            ctest_cmd += "-VV "

        container.run(
            img=getattr(args, "img", None),
            use_tini=True,
            docker_opts="--entrypoint=bash",
            extra_args=["-c", ctest_cmd],
        )

    def build_project_locally(
        self,
        project_name: str,
        language: Optional[str] = None,
        build_type: Optional[str] = None,
        with_operators: Optional[str] = None,
        dryrun: bool = False,
        pkg_generator: str = "DEB",
        parallel: Optional[str] = None,
        benchmark: bool = False,
        configure_args: Optional[list[str]] = None,
    ) -> tuple[Path, dict]:
        """Helper method to build a project locally"""
        project_data = self.find_project(project_name=project_name, language=language)
        project_type = project_data.get("project_type", "application")

        # Handle benchmark patching before building
        app_source_path = None
        if benchmark:
            if project_type in ["application", "workflow", "benchmark"]:
                app_source_path = project_data.get("source_folder", "")
                patch_script = (
                    HoloHubCLI.HOLOHUB_ROOT
                    / "benchmarks/holoscan_flow_benchmarking/patch_application.sh"
                )
                holohub_cli_util.run_command(
                    [str(patch_script), str(app_source_path)], dry_run=dryrun
                )
                print("Building for Holoscan Flow Benchmarking")
            else:
                holohub_cli_util.fatal(
                    "--benchmark option is only available for applications/workflows"
                )

        build_type = holohub_cli_util.get_buildtype_str(build_type)
        build_dir = HoloHubCLI.DEFAULT_BUILD_PARENT_DIR / project_name
        build_dir.mkdir(parents=True, exist_ok=True)

        proj_prefix = holohub_cli_util.determine_project_prefix(project_type)
        cmake_args = [
            "cmake",
            "-B",
            str(build_dir),
            "-S",
            str(HoloHubCLI.HOLOHUB_ROOT),
            "--no-warn-unused-cli",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DPython3_ROOT_DIR={os.path.dirname(os.path.dirname(sys.executable))}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DCMAKE_PREFIX_PATH={HoloHubCLI.DEFAULT_SDK_DIR}",
            f"-DHOLOHUB_DATA_DIR:PATH={HoloHubCLI.DEFAULT_DATA_DIR}",
            f"-D{proj_prefix}_{project_name}=ON",
        ]
        # Add benchmark-specific CMake flags
        if benchmark:
            cmake_args.append(
                f"-DCMAKE_CXX_FLAGS=-I{HoloHubCLI.HOLOHUB_ROOT}/benchmarks/holoscan_flow_benchmarking"
            )

        # use -G Ninja if available
        if shutil.which("ninja"):
            cmake_args.extend(["-G", "Ninja"])
        # Add optional operators if specified
        if with_operators:
            cmake_args.append(f'-DHOLOHUB_BUILD_OPERATORS="{with_operators}"')

        normalized_language = holohub_cli_util.normalize_language(
            project_data.get("metadata", {}).get("language", None)
        )
        if normalized_language == "python":
            cmake_args.append("-DHOLOHUB_BUILD_PYTHON=ON")
            cmake_args.append("-DHOLOHUB_BUILD_CPP=OFF")
        elif normalized_language == "cpp":
            cmake_args.append("-DHOLOHUB_BUILD_PYTHON=OFF")
            cmake_args.append("-DHOLOHUB_BUILD_CPP=ON")

        if configure_args:
            cmake_args.extend(configure_args)

        holohub_cli_util.run_command(cmake_args, dry_run=dryrun)

        # Build the project with optional parallel jobs
        build_cmd = ["cmake", "--build", str(build_dir), "--config", build_type]
        # Determine the number of parallel jobs (user input > env var > CPU count):
        if parallel is not None:
            build_njobs = str(parallel)
        else:
            build_njobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", str(os.cpu_count()))
        build_cmd.extend(["-j", build_njobs])

        holohub_cli_util.run_command(build_cmd, dry_run=dryrun)

        # If this is a package, run cpack
        if project_type == "package":
            pkg_build_dir = build_dir / "pkg"
            if pkg_build_dir.exists():
                for cpack_config in pkg_build_dir.glob("CPackConfig-*.cmake"):
                    holohub_cli_util.run_command(
                        ["cpack", "--config", str(cpack_config), "-G", pkg_generator],
                        dry_run=dryrun,
                    )

        # Handle benchmark restoration after building
        if (
            benchmark
            and app_source_path
            and project_type in ["application", "workflow", "benchmark"]
        ):
            restore_script = (
                HoloHubCLI.HOLOHUB_ROOT
                / "benchmarks/holoscan_flow_benchmarking/restore_application.sh"
            )
            holohub_cli_util.run_command(
                [str(restore_script), str(app_source_path)], dry_run=dryrun
            )

        return build_dir, project_data

    def handle_build(self, args: argparse.Namespace) -> None:
        """Handle build command"""
        skip_docker_build, _ = holohub_cli_util.check_skip_builds(args)

        # Handle mode-specific configuration
        project_data = self.find_project(args.project, language=args.language)
        mode_name, mode_config = self.resolve_mode(project_data, getattr(args, "mode", None))

        self.validate_mode(args, mode_name, mode_config, project_data, getattr(args, "mode", None))

        # Apply mode-specific build configuration
        build_args = self.get_effective_build_config(
            args, mode_config, project_data, getattr(args, "mode", None)
        )

        if mode_config:
            print(f"Building {args.project} in '{mode_name}' mode")

        if args.local or os.environ.get("HOLOHUB_BUILD_LOCAL"):
            self.build_project_locally(
                project_name=args.project,
                language=args.language if hasattr(args, "language") else None,
                build_type=args.build_type,
                with_operators=build_args.get("with_operators"),
                dryrun=args.dryrun,
                pkg_generator=getattr(args, "pkg_generator", "DEB"),
                parallel=getattr(args, "parallel", None),
                benchmark=getattr(args, "benchmark", False),
                configure_args=build_args.get("configure_args"),
            )
        else:
            # Build in container
            container = self._make_project_container(
                project_name=args.project,
                language=args.language if hasattr(args, "language") else None,
            )
            container.dryrun = args.dryrun
            container.verbose = args.verbose
            if not skip_docker_build:
                container.build(
                    docker_file=args.docker_file,
                    base_img=args.base_img,
                    img=args.img,
                    no_cache=args.no_cache,
                    build_args=build_args.get("build_args"),
                    cuda_version=getattr(args, "cuda", None),
                    extra_scripts=getattr(args, "extra_scripts", []),
                )
            else:
                if hasattr(args, "cuda") and args.cuda is not None:
                    container.cuda_version = args.cuda

            # Build command with all necessary arguments
            build_cmd = f"{self.script_name} build {args.project}"
            # Only add mode name if it was explicitly requested by user (not implicitly resolved)
            if mode_name and getattr(args, "mode", None) is not None:
                build_cmd += f" {mode_name}"
            build_cmd += " --local"
            if args.build_type:
                build_cmd += f" --build-type {args.build_type}"
            if args.with_operators:
                build_cmd += f' --build-with "{args.with_operators}"'
            if hasattr(args, "pkg_generator"):
                build_cmd += f" --pkg-generator {args.pkg_generator}"
            if hasattr(args, "language") and args.language:
                build_cmd += f" --language {args.language}"
            if getattr(args, "parallel", None):
                build_cmd += f" --parallel {args.parallel}"
            if args.verbose:
                build_cmd += " --verbose"
            if getattr(args, "benchmark", False):
                build_cmd += " --benchmark"
            if getattr(args, "configure_args", None):
                for configure_arg in args.configure_args:
                    build_cmd += f" --configure-args={shlex.quote(configure_arg)}"

            img = getattr(args, "img", None) or container.image_name
            docker_opts = build_args.get("docker_opts", "")
            docker_opts_extra, extra_args = holohub_cli_util.get_entrypoint_command_args(
                img, build_cmd, docker_opts, dry_run=args.dryrun
            )
            if docker_opts_extra:
                docker_opts = f"{docker_opts} {docker_opts_extra}".strip()
            container.run(
                img=getattr(args, "img", None),
                local_sdk_root=getattr(args, "local_sdk_root", None),
                enable_x11=getattr(args, "enable_x11", True),
                ssh_x11=getattr(args, "ssh_x11", False),
                use_tini=getattr(args, "init", False),
                persistent=getattr(args, "persistent", False),
                nsys_profile=getattr(args, "nsys_profile", False),
                nsys_location=getattr(args, "nsys_location", ""),
                as_root=getattr(args, "as_root", False),
                docker_opts=docker_opts,
                add_volumes=getattr(args, "add_volume", None),
                enable_mps=getattr(args, "mps", False),
                extra_args=extra_args,
            )

    def handle_run(self, args: argparse.Namespace) -> None:
        """Handle run command"""
        skip_docker_build, skip_local_build = holohub_cli_util.check_skip_builds(args)
        is_local_mode = args.local or os.environ.get("HOLOHUB_BUILD_LOCAL")

        # Handle mode-specific configuration
        project_data = self.find_project(args.project, language=args.language)
        mode_name, mode_config = self.resolve_mode(project_data, getattr(args, "mode", None))

        self.validate_mode(args, mode_name, mode_config, project_data, getattr(args, "mode", None))

        # Apply mode-specific build configuration for build process
        build_args = self.get_effective_build_config(
            args, mode_config, project_data, getattr(args, "mode", None)
        )

        # Apply mode-specific run configuration
        run_args = self.get_effective_run_config(
            args, mode_config, project_data, getattr(args, "mode", None)
        )

        if mode_config:
            print(f"Running {args.project} in '{mode_name}' mode")

        if is_local_mode:
            if args.docker_opts:
                holohub_cli_util.fatal(
                    "Container arguments were provided with `--docker-opts` but a non-containerized build was requested."
                )
            if skip_local_build:
                # Skip building; reuse previously resolved project_data and build directory
                build_dir = HoloHubCLI.DEFAULT_BUILD_PARENT_DIR / args.project
                if not build_dir.is_dir() and not args.dryrun:
                    holohub_cli_util.fatal(
                        f"The build directory {build_dir} for this application does not exist.\n"
                        f"Did you forget to build the application first? Try running:\n"
                        f"  {self.script_name} build {args.project}"
                    )
            else:
                build_dir, project_data = self.build_project_locally(
                    project_name=args.project,
                    language=args.language if hasattr(args, "language") else None,
                    build_type=args.build_type,
                    with_operators=build_args.get("with_operators"),
                    dryrun=args.dryrun,
                    pkg_generator=getattr(args, "pkg_generator", "DEB"),
                    parallel=getattr(args, "parallel", None),
                    configure_args=build_args.get("configure_args"),
                )

            language = holohub_cli_util.normalize_language(
                project_data.get("metadata", {}).get("language", None)
            )

            if mode_config and "run" in mode_config:
                run_config = mode_config["run"]  # Use mode-specific run configuration
            else:  # Fall back to legacy run configuration
                run_config = project_data.get("metadata", {}).get("run", {})
                if not run_config:
                    holohub_cli_util.fatal(
                        f"Project '{args.project}' does not have a run configuration"
                    )

            prefix = holohub_cli_util.resolve_path_prefix(None)
            path_mapping = holohub_cli_util.build_holohub_path_mapping(
                holohub_root=HoloHubCLI.HOLOHUB_ROOT,
                project_data=project_data,
                build_dir=build_dir,
                data_dir=HoloHubCLI.DEFAULT_DATA_DIR,
                prefix=prefix,
            )
            if path_mapping:
                mapping_info = ";\n".join(
                    f"<{key}>: {value}" for key, value in path_mapping.items()
                )
                print(
                    holohub_cli_util.format_cmd(
                        f"Path mappings: \n{mapping_info}", is_dryrun=args.dryrun
                    )
                )
            # Process command template using the path mapping
            cmd = holohub_cli_util.replace_placeholders(run_config["command"], path_mapping)

            # Use effective run args (which may come from mode or CLI)
            effective_run_args = run_args.get("run_args")
            if effective_run_args:
                cmd_args = shlex.split(effective_run_args)
                if isinstance(cmd, str):  # Ensure cmd is a list of arguments
                    cmd = shlex.split(cmd)
                cmd.extend(cmd_args)

            if language == "cpp":
                if not build_dir.is_dir() and not args.dryrun:
                    holohub_cli_util.fatal(
                        f"The build directory {build_dir} for this application does not exist.\n"
                        f"Did you forget to '{self.script_name} build {args.project}'?"
                    )

            workdir_spec = run_config.get("workdir", f"{prefix}app_bin")
            if not workdir_spec:
                target_dir = Path(path_mapping.get(f"{prefix}root", "."))
            elif workdir_spec in path_mapping:
                target_dir = Path(path_mapping[workdir_spec])
            else:
                target_dir = Path(workdir_spec)
            print(holohub_cli_util.format_cmd("cd " + str(target_dir), is_dryrun=args.dryrun))
            if not args.dryrun:
                os.chdir(target_dir)

            # Set up environment
            env = os.environ.copy()
            env["PYTHONPATH"] = (
                f"{os.environ.get('PYTHONPATH', '')}:{HoloHubCLI.DEFAULT_SDK_DIR}/../python/lib:{build_dir}/python/lib:{HoloHubCLI.HOLOHUB_ROOT}"
            )
            env["HOLOHUB_DATA_PATH"] = str(HoloHubCLI.DEFAULT_DATA_DIR)
            env["HOLOSCAN_INPUT_PATH"] = os.environ.get(
                "HOLOSCAN_INPUT_PATH", str(HoloHubCLI.DEFAULT_DATA_DIR)
            )

            if run_config.get("env", None) is not None:
                env.update(run_config["env"])

            # Print environment setup
            if args.verbose or args.dryrun:
                print(
                    holohub_cli_util.format_cmd(
                        "export PYTHONPATH=" + env["PYTHONPATH"], is_dryrun=args.dryrun
                    )
                )
                print(
                    holohub_cli_util.format_cmd(
                        "export HOLOHUB_DATA_PATH=" + env["HOLOHUB_DATA_PATH"],
                        is_dryrun=args.dryrun,
                    )
                )
                print(
                    holohub_cli_util.format_cmd(
                        "export HOLOSCAN_INPUT_PATH=" + env["HOLOSCAN_INPUT_PATH"],
                        is_dryrun=args.dryrun,
                    )
                )

            # Handle Nsight Systems profiling
            if args.nsys_profile:
                if (
                    not shutil.which("nsys")
                    and not os.path.isdir("/opt/nvidia/nsys-host")
                    and not args.dryrun
                ):
                    holohub_cli_util.fatal(
                        "Nsight Systems CLI command 'nsys' not found. No Nsight installation from the host is also mounted."
                    )
                nsys_cmd = "/opt/nvidia/nsys-host/bin/nsys" if not shutil.which("nsys") else "nsys"

                # Check perf_event_paranoid level
                if not args.dryrun:
                    try:
                        with open("/proc/sys/kernel/perf_event_paranoid") as f:
                            if int(f.read()) > 2:
                                holohub_cli_util.fatal(
                                    "For Nsight Systems profiling the Linux operating system's perf_event_paranoid level must be 2 or less."
                                )
                    except (IOError, ValueError):
                        pass

                cmd = f"{nsys_cmd} profile --trace=cuda,vulkan,nvtx,osrt {cmd}"

            cmd_to_run = cmd if isinstance(cmd, list) else shlex.split(cmd)
            holohub_cli_util.run_command(cmd_to_run, env=env, dry_run=args.dryrun)
        else:
            container = self._make_project_container(
                project_name=args.project,
                language=args.language if hasattr(args, "language") else None,
            )
            container.dryrun = args.dryrun
            container.verbose = args.verbose
            if not skip_docker_build:
                container.build(
                    docker_file=args.docker_file,
                    base_img=args.base_img,
                    img=args.img,
                    no_cache=args.no_cache,
                    build_args=build_args.get("build_args"),
                    cuda_version=getattr(args, "cuda", None),
                    extra_scripts=getattr(args, "extra_scripts", []),
                )
            else:
                if hasattr(args, "cuda") and args.cuda is not None:
                    container.cuda_version = args.cuda

            language = holohub_cli_util.normalize_language(
                container.project_metadata.get("metadata", {}).get("language", None)
            )

            run_cmd = f"{self.script_name} run {args.project}"
            # Only add mode name if it was explicitly requested by user (not implicitly resolved)
            if mode_name and getattr(args, "mode", None) is not None:
                run_cmd += f" {mode_name}"
            run_cmd += f" --language {language} --local"
            if args.verbose:
                run_cmd += " --verbose"
            if args.build_type:
                run_cmd += f" --build-type {args.build_type}"
            if getattr(args, "pkg_generator", None) and args.pkg_generator != "DEB":
                run_cmd += f" --pkg-generator {args.pkg_generator}"
            if args.nsys_profile:
                run_cmd += " --nsys-profile"
            if skip_local_build:
                run_cmd += " --no-local-build"
            if hasattr(args, "with_operators") and args.with_operators:
                run_cmd += f' --build-with "{args.with_operators}"'
            if hasattr(args, "run_args") and args.run_args:
                run_cmd += f" --run-args={shlex.quote(args.run_args)}"
            if getattr(args, "parallel", None):
                run_cmd += f" --parallel {args.parallel}"
            if getattr(args, "configure_args", None):
                for configure_arg in args.configure_args:
                    run_cmd += f" --configure-args={shlex.quote(configure_arg)}"

            img = getattr(args, "img", None) or container.image_name
            docker_opts = build_args.get("docker_opts", "")
            docker_opts_extra, extra_args = holohub_cli_util.get_entrypoint_command_args(
                img, run_cmd, docker_opts, dry_run=args.dryrun
            )
            if docker_opts_extra:
                docker_opts = f"{docker_opts} {docker_opts_extra}".strip()
            container.run(
                img=getattr(args, "img", None),
                local_sdk_root=getattr(args, "local_sdk_root", None),
                enable_x11=getattr(args, "enable_x11", True),
                ssh_x11=getattr(args, "ssh_x11", False),
                use_tini=getattr(args, "init", False),
                persistent=getattr(args, "persistent", False),
                nsys_profile=getattr(args, "nsys_profile", False),
                nsys_location=getattr(args, "nsys_location", ""),
                as_root=getattr(args, "as_root", False),
                docker_opts=docker_opts,
                add_volumes=getattr(args, "add_volume", None),
                enable_mps=getattr(args, "mps", False),
                extra_args=extra_args,
            )

    def handle_list(self, args: argparse.Namespace) -> None:
        """Handle list command"""
        LIST_TYPES = [
            "application",
            "benchmark",
            "gxf_extension",
            "package",
            "operator",
            "tutorial",
            "workflow",
        ]
        grouped_metadata = defaultdict(list)
        for project in self.projects:
            grouped_metadata[project.get("project_type", "")].append(project)

        for project_type in LIST_TYPES:
            if project_type not in grouped_metadata:
                continue
            print(f"\n{Color.white(f'== {project_type.upper()}S =================', bold=True)}\n")
            for project in sorted(grouped_metadata[project_type], key=lambda x: x["project_name"]):
                language = project.get("metadata", {}).get("language", "")
                language = f"({language})" if language else ""
                print(f'{project["project_name"]} {language}')

        print(f"\n{Color.white('=================================', bold=True)}\n")

    def handle_modes(self, args: argparse.Namespace) -> None:
        """Handle modes command"""
        project_data = self.find_project(args.project, language=args.language)
        modes = project_data.get("metadata", {}).get("modes", {})

        if not modes:
            print(f"No modes defined for {args.project}")
            return

        print(f"\n{Color.white(f'Available modes for {args.project}:', bold=True)}\n")

        for mode_name, mode_config in modes.items():
            description = mode_config.get("description", "No description")
            print(f"  {Color.green(mode_name, bold=True)} - {description}")
            requirements = mode_config.get("requirements", [])
            if requirements:
                req_list = ", ".join(requirements)
                print(f"    Requirements: {req_list}")

            print()  # Empty line between modes

    def handle_autocompletion_list(self, args: argparse.Namespace) -> None:
        """Handle autocompletion_list command - output project names and commands for bash completion"""
        project_names = set()
        for project in self.projects:
            project_names.add(project["project_name"])
        for name in sorted(project_names):
            print(name)
        commands = [
            "build-container",
            "run-container",
            "build",
            "run",
            "list",
            "lint",
            "setup",
            "install",
            "create",
            "cpp",
            "python",
            "autocompletion_list",
        ]
        for cmd in commands:
            print(cmd)

    def handle_lint(self, args: argparse.Namespace) -> None:
        """Handle lint command"""
        if args.install_dependencies:
            self._install_lint_deps(args.dryrun)
            return

        exit_code = 0

        # Add ~/.local/bin to PATH for pip-installed executables
        env = os.environ.copy()
        local_bin_path = Path.home() / ".local" / "bin"
        if str(local_bin_path) not in env.get("PATH", ""):
            env["PATH"] = str(local_bin_path) + ":" + env.get("PATH", "")
            print(f"Added {local_bin_path} to PATH.")

        # Set cache directories to /tmp when in Docker container to avoid permission issues
        if holohub_cli_util.is_running_in_docker():
            env["RUFF_CACHE_DIR"] = "/tmp/.ruff_cache"
            env["BLACK_CACHE_DIR"] = "/tmp/.black_cache"
            print(f"Set cache directories to {env['RUFF_CACHE_DIR']} and {env['BLACK_CACHE_DIR']}")

        # Change to script directory
        print(
            holohub_cli_util.format_cmd("cd " + str(HoloHubCLI.HOLOHUB_ROOT), is_dryrun=args.dryrun)
        )
        if not args.dryrun:
            os.chdir(HoloHubCLI.HOLOHUB_ROOT)

        if args.fix:
            # Fix Python
            holohub_cli_util.run_command(
                ["ruff", "check", "--fix", "--ignore", "E712", args.path],
                check=False,
                dry_run=args.dryrun,
                env=env,
            )
            holohub_cli_util.run_command(
                ["isort", args.path], check=False, dry_run=args.dryrun, env=env
            )
            holohub_cli_util.run_command(
                ["black", args.path], check=False, dry_run=args.dryrun, env=env
            )
            holohub_cli_util.run_command(
                [
                    "codespell",
                    "-w",
                    "-i",
                    "3",
                    args.path,
                    "--ignore-words",
                    "codespell_ignore_words.txt",
                    "--exclude-file",
                    "codespell.txt",
                    "--skip=*.onnx,*.min.js,*.min.js.map,Contrastive_learning_Notebook.ipynb,./data,./applications/holochat/models",
                ],
                check=False,
                dry_run=args.dryrun,
                env=env,
            )

            # Fix C++ with clang-format
            try:
                cmd = [
                    sys.executable,
                    "-m",
                    "cpplint",
                    "--exclude",
                    "build",
                    "--exclude",
                    "install",
                    "--exclude",
                    "build-*",
                    "--exclude",
                    "install-*",
                    "--exclude",
                    "applications/holoviz/template/cookiecutter*",
                    "--exclude",
                    ".ruff_cache",
                    "--exclude",
                    ".local",
                    "--recursive",
                    args.path,
                ]
                if args.dryrun:
                    holohub_cli_util.run_command(cmd, dry_run=True)
                    cpp_files = ""
                else:
                    cpp_files = subprocess.check_output(
                        cmd, stderr=subprocess.PIPE, text=True, env=env
                    )

                for file in cpp_files.splitlines():
                    if ":" in file:  # Only process files with issues
                        file = file.split(":")[0]
                        holohub_cli_util.run_command(
                            [
                                "clang-format",
                                "--style=file",
                                "--sort-includes=0",
                                "--lines=20:10000",
                                "-i",
                                file,
                            ],
                            check=False,
                            dry_run=args.dryrun,
                            env=env,
                        )
            except subprocess.CalledProcessError:
                pass

        else:
            # Run linting checks
            print(Color.blue("Linting Python"))
            if (
                holohub_cli_util.run_command(
                    ["ruff", "check", "--ignore", "E712", args.path],
                    check=False,
                    dry_run=args.dryrun,
                    env=env,
                ).returncode
                != 0
            ):
                exit_code = 1
            if (
                holohub_cli_util.run_command(
                    ["isort", "-c", args.path], check=False, dry_run=args.dryrun, env=env
                ).returncode
                != 0
            ):
                exit_code = 1
            if (
                holohub_cli_util.run_command(
                    ["black", "--check", args.path], check=False, dry_run=args.dryrun, env=env
                ).returncode
                != 0
            ):
                exit_code = 1

            print(Color.blue("Linting C++"))
            if (
                holohub_cli_util.run_command(
                    [
                        "python",
                        "-m",
                        "cpplint",
                        "--quiet",
                        "--exclude",
                        "build",
                        "--exclude",
                        "install",
                        "--exclude",
                        "build-*",
                        "--exclude",
                        "install-*",
                        "--exclude",
                        ".vscode-server",
                        "--exclude",
                        "applications/holoviz/template/cookiecutter*",
                        "--exclude",
                        ".ruff_cache",
                        "--exclude",
                        ".local",
                        "--recursive",
                        args.path,
                    ],
                    check=False,
                    dry_run=args.dryrun,
                    env=env,
                ).returncode
                != 0
            ):
                exit_code = 1

            print(Color.blue("Code spelling"))
            if (
                holohub_cli_util.run_command(
                    [
                        "codespell",
                        args.path,
                        "--skip=*.onnx,*.min.js,*.min.js.map,Contrastive_learning_Notebook.ipynb,./data",
                        "--ignore-words",
                        "codespell_ignore_words.txt",
                        "--exclude-file",
                        "codespell.txt",
                    ],
                    check=False,
                    dry_run=args.dryrun,
                    env=env,
                ).returncode
                != 0
            ):
                exit_code = 1

            print(Color.blue("Linting CMake"))
            cmake_files = list(Path(args.path).rglob("CMakeLists.txt"))
            cmake_files.extend(Path(args.path).rglob("*.cmake"))
            excluded_paths = ["build", "install", "tmp"]
            cmake_files = [
                f
                for f in cmake_files
                if not any(
                    excluded_dir in f.parts or any(part.startswith(".") for part in f.parts)
                    for excluded_dir in excluded_paths
                )
            ]
            if cmake_files:
                batch_size = 100
                cmake_lint_failed = False
                for i in range(0, len(cmake_files), batch_size):
                    if (
                        holohub_cli_util.run_command(
                            [
                                "cmakelint",
                                "--filter=-whitespace/indent,-linelength,-readability/wonkycase,-convention/filename,-package/stdargs",
                                *[str(f) for f in cmake_files[i : i + batch_size]],
                            ],
                            check=False,
                            dry_run=args.dryrun,
                            env=env,
                        ).returncode
                        != 0
                    ):
                        cmake_lint_failed = True

                if cmake_lint_failed:
                    exit_code = 1

        if exit_code == 0 and not args.dryrun:
            print(Color.green("Everything looks good!"))
        sys.exit(exit_code)

    def _install_lint_deps(self, dry_run: bool = False) -> None:
        """Install linting dependencies"""
        print(holohub_cli_util.format_cmd("cd " + str(HoloHubCLI.HOLOHUB_ROOT), is_dryrun=dry_run))
        if not dry_run:
            os.chdir(HoloHubCLI.HOLOHUB_ROOT)

        print("Install Lint Dependencies for Python")
        holohub_cli_util.run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(HoloHubCLI.HOLOHUB_ROOT / "utilities/requirements.lint.txt"),
            ],
            dry_run=dry_run,
        )

    def _install_template_deps(self, dry_run: bool = False) -> None:
        """Install template dependencies"""
        print(holohub_cli_util.format_cmd("cd " + str(HoloHubCLI.HOLOHUB_ROOT), is_dryrun=dry_run))
        if not dry_run:
            os.chdir(HoloHubCLI.HOLOHUB_ROOT)

        print("Install Template Dependencies")
        holohub_cli_util.run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(HoloHubCLI.HOLOHUB_ROOT / "utilities" / "requirements.template.txt"),
            ],
            dry_run=dry_run,
        )

    def handle_setup(self, args: argparse.Namespace) -> None:
        """Handle setup command"""

        if args.list_scripts:
            setup_scripts_dir = holohub_cli_util.get_holohub_setup_scripts_dir()
            print(
                holohub_cli_util.format_cmd(
                    f"Listing setup scripts available in {setup_scripts_dir}"
                )
            )
            print(Color.green("Use with `./holohub setup --scripts <script_name>`"))
            for script in setup_scripts_dir.glob("*.sh"):
                print(f"  {script.stem}")
            sys.exit(0)

        if args.scripts:
            for script in args.scripts:
                script_path = holohub_cli_util.get_holohub_setup_scripts_dir() / f"{script}.sh"
                if any(sep in script for sep in ("/", "\\")):
                    holohub_cli_util.fatal(
                        f"Invalid script name '{script}': path separators are not allowed"
                    )
                script_path = (
                    holohub_cli_util.get_holohub_setup_scripts_dir().resolve() / f"{script}.sh"
                )
                if not script_path.exists():
                    holohub_cli_util.fatal(
                        f"Script {script}.sh not found in {holohub_cli_util.get_holohub_setup_scripts_dir()}"
                    )
                holohub_cli_util.run_command(["bash", str(script_path)], dry_run=args.dryrun)
            sys.exit(0)

        if not args.scripts:
            holohub_cli_util.install_packages_if_missing(
                ["wget", "xvfb", "git", "unzip", "ffmpeg", "ninja-build", "libv4l-dev"],
                dry_run=args.dryrun,
            )

            holohub_cli_util.setup_cmake(dry_run=args.dryrun)
            holohub_cli_util.setup_python_dev(dry_run=args.dryrun)
            holohub_cli_util.setup_ngc_cli(dry_run=args.dryrun)
            holohub_cli_util.setup_cuda_dependencies(dry_run=args.dryrun)

            source = f"{HoloHubCLI.HOLOHUB_ROOT}/utilities/holohub_autocomplete"
            dest_folder = "/etc/bash_completion.d"
            dest = f"{dest_folder}/holohub_autocomplete"
            if (
                not os.path.exists(dest) or not filecmp.cmp(source, dest, shallow=False)
            ) and os.path.exists(dest_folder):
                holohub_cli_util.run_command(["cp", source, dest_folder], dry_run=args.dryrun)

            if not args.dryrun:
                print(
                    Color.blue("\nTo enable ./holohub autocomplete in your current shell session:")
                )
                print("  source /etc/bash_completion.d/holohub_autocomplete")
                print("Or add it to your shell profile:")
                print("  echo '. /etc/bash_completion.d/holohub_autocomplete' >> ~/.bashrc")
                print("  source ~/.bashrc")

                print(Color.green("Setup for HoloHub is ready. Happy Holocoding!"))

    def handle_env_info(self, args: argparse.Namespace) -> None:
        """Handle env-info command to collect debugging information"""
        print(holohub_cli_util.format_cmd("Environment Information"))
        holohub_cli_util.collect_holohub_info(
            holohub_root=self.HOLOHUB_ROOT,
            build_dir=self.DEFAULT_BUILD_PARENT_DIR,
            data_dir=self.DEFAULT_DATA_DIR,
            sdk_dir=self.DEFAULT_SDK_DIR,
        )
        holohub_cli_util.collect_git_info(holohub_root=self.HOLOHUB_ROOT)
        holohub_cli_util.collect_env_info()
        print(
            holohub_cli_util.format_cmd(
                "Complete (Before sharing, please review and remove sensitive information)"
            )
        )

    def handle_install(self, args: argparse.Namespace) -> None:
        """Handle install command"""
        skip_docker_build, _ = holohub_cli_util.check_skip_builds(args)

        # Handle mode-specific configuration (if project has modes)
        project_data = self.find_project(args.project, language=getattr(args, "language", None))
        mode_name, mode_config = self.resolve_mode(project_data, getattr(args, "mode", None))
        self.validate_mode(args, mode_name, mode_config, project_data, getattr(args, "mode", None))
        build_args = self.get_effective_build_config(
            args, mode_config, project_data, getattr(args, "mode", None)
        )

        if mode_config:
            print(f"Installing {args.project} in '{mode_name}' mode")

        if args.local or os.environ.get("HOLOHUB_BUILD_LOCAL"):
            # Build and install locally
            build_dir, project_data = self.build_project_locally(
                project_name=args.project,
                language=getattr(args, "language", None),
                build_type=args.build_type,
                with_operators=build_args.get("with_operators"),
                dryrun=args.dryrun,
                parallel=getattr(args, "parallel", None),
                configure_args=build_args.get("configure_args"),
            )
            # Install the project
            holohub_cli_util.run_command(
                ["cmake", "--install", str(build_dir)], dry_run=args.dryrun
            )
            if not args.dryrun:
                print(f"{Color.green('Successfully installed')} {args.project}")
        else:
            # Install in container
            container = self._make_project_container(
                project_name=args.project,
                language=getattr(args, "language", None),
            )
            container.dryrun = args.dryrun
            container.verbose = args.verbose
            if not skip_docker_build:
                container.build(
                    docker_file=args.docker_file,
                    base_img=args.base_img,
                    img=args.img,
                    no_cache=args.no_cache,
                    build_args=build_args.get("build_args"),
                    cuda_version=getattr(args, "cuda", None),
                    extra_scripts=getattr(args, "extra_scripts", []),
                )
            else:
                if hasattr(args, "cuda") and args.cuda is not None:
                    container.cuda_version = args.cuda

            # Install command with all necessary arguments
            install_cmd = f"{self.script_name} install {args.project} --local"
            if args.build_type:
                install_cmd += f" --build-type {args.build_type}"
            if getattr(args, "language", None):
                install_cmd += f" --language {args.language}"
            if getattr(args, "with_operators", None):
                install_cmd += f' --build-with "{args.with_operators}"'
            if getattr(args, "parallel", None):
                install_cmd += f" --parallel {args.parallel}"
            if args.verbose:
                install_cmd += " --verbose"
            if getattr(args, "configure_args", None):
                for configure_arg in args.configure_args:
                    install_cmd += f" --configure-args={shlex.quote(configure_arg)}"

            img = getattr(args, "img", None) or container.image_name
            docker_opts = build_args.get("docker_opts", "")
            docker_opts_extra, extra_args = holohub_cli_util.get_entrypoint_command_args(
                img, install_cmd, docker_opts, dry_run=args.dryrun
            )
            if docker_opts_extra:
                docker_opts = f"{docker_opts} {docker_opts_extra}".strip()
            container.run(
                img=getattr(args, "img", None),
                local_sdk_root=getattr(args, "local_sdk_root", None),
                enable_x11=getattr(args, "enable_x11", True),
                ssh_x11=getattr(args, "ssh_x11", False),
                use_tini=getattr(args, "init", False),
                persistent=getattr(args, "persistent", False),
                nsys_profile=getattr(args, "nsys_profile", False),
                nsys_location=getattr(args, "nsys_location", ""),
                as_root=getattr(args, "as_root", False),
                docker_opts=docker_opts,
                add_volumes=getattr(args, "add_volume", None),
                enable_mps=getattr(args, "mps", False),
                extra_args=extra_args,
            )

    def handle_clear_cache(self, args: argparse.Namespace) -> None:
        """Handle clear-cache command"""
        if args.dryrun:
            print(Color.blue("Would clear cache folders:"))
        else:
            print(Color.blue("Clearing cache..."))

        cache_dirs = [
            self.DEFAULT_BUILD_PARENT_DIR,
            self.DEFAULT_DATA_DIR,
        ]
        for pattern in ["build", "build-*", "data", "data-*", "install"]:
            for path in HoloHubCLI.HOLOHUB_ROOT.glob(pattern):
                if path.is_dir() and path not in cache_dirs:
                    cache_dirs.append(path)
        for path in set(cache_dirs):
            if path.exists() and path.is_dir():
                if args.dryrun:
                    print(f"  {Color.yellow('Would remove:')} {path}")
                else:
                    print(f"  {Color.red('Removing:')} {path}")
                    shutil.rmtree(path)

    def _add_to_cmakelists(self, project_name: str) -> None:
        """Add a new application to applications/CMakeLists.txt if it doesn't exist"""
        cmakelists_path = self.HOLOHUB_ROOT / "applications" / "CMakeLists.txt"
        if not cmakelists_path.exists():
            return
        with open(cmakelists_path, "r") as f:
            lines = f.readlines()
        target_line = f"add_holohub_application({project_name})"
        if any(target_line in line.strip() for line in lines):
            return
        try:
            with open(cmakelists_path, "a") as f:
                f.write(f"add_holohub_application({project_name})\n")
        except Exception as e:
            print(Color.red(f"Failed to add application to applications/CMakeLists.txt: {str(e)}"))
            print(Color.red("Please add the application manually to applications/CMakeLists.txt"))

    def handle_vscode(self, args: argparse.Namespace) -> None:
        """Builds a dev container and launches VS Code with proper devcontainer configuration."""
        if not shutil.which("code") and not args.dryrun:
            holohub_cli_util.fatal(
                "Please install VS Code to use VS Code Dev Container. "
                "Follow the instructions at https://code.visualstudio.com/Download"
            )

        skip_docker_build, _ = holohub_cli_util.check_skip_builds(args)
        container = self._make_project_container(
            project_name=args.project, language=getattr(args, "language", None)
        )
        container.dryrun = args.dryrun
        container.verbose = args.verbose
        dev_container_tag = "holohub-dev-container"
        if args.project:
            dev_container_tag += f"-{args.project}"
        dev_container_tag += ":dev"

        if not skip_docker_build:
            print(f"Building base Dev Container {dev_container_tag}...")
            container.build(
                docker_file=args.docker_file,
                base_img=args.base_img,
                img=dev_container_tag,
                no_cache=args.no_cache,
                build_args=args.build_args,
                cuda_version=getattr(args, "cuda", None),
                extra_scripts=getattr(args, "extra_scripts", []),
            )
        else:
            if hasattr(args, "cuda") and args.cuda is not None:
                container.cuda_version = args.cuda
            print(f"Skipping build, using existing Dev Container {dev_container_tag}...")
        devcontainer_env_options = container.get_devcontainer_args(
            docker_opts=getattr(args, "docker_opts", None) or ""
        )
        # Enable X11 access for Docker containers
        container.enable_x11_access()

        devcontainer_content = holohub_cli_util.get_devcontainer_config(
            holohub_root=self.HOLOHUB_ROOT, project_name=args.project, dry_run=args.dryrun
        )
        devcontainer_content = devcontainer_content.replace(
            "${localWorkspaceFolder}", str(self.HOLOHUB_ROOT)
        )
        devcontainer_content = devcontainer_content.replace('//"<env>"', devcontainer_env_options)
        os.environ["HOLOHUB_BASE_IMAGE"] = dev_container_tag
        if args.project:
            os.environ["HOLOHUB_APP_NAME"] = args.project

        if not args.dryrun:
            tmpdir = tempfile.mkdtemp()
            workspace_name = self.HOLOHUB_ROOT.name
            tmp_workspace = Path(tmpdir) / workspace_name
            tmp_workspace.mkdir()
            tmp_devcontainer = tmp_workspace / ".devcontainer"
            tmp_devcontainer.mkdir()
            devcontainer_json_dst = tmp_devcontainer / "devcontainer.json"
            with open(devcontainer_json_dst, "w") as f:
                f.write(devcontainer_content)
            print(f"Created temporary workspace: {tmp_devcontainer}")
        else:
            tmp_workspace = "<tmp_workspace>"
        holohub_cli_util.launch_vscode_devcontainer(str(tmp_workspace), dry_run=args.dryrun)

    def handle_create(self, args: argparse.Namespace) -> None:
        """Handle create command"""
        # Ensure template directory exists
        template_dir = self.HOLOHUB_ROOT / args.template
        if not template_dir.exists() and not args.dryrun:
            holohub_cli_util.fatal(f"Template directory {template_dir} does not exist")

        if not args.directory.exists() and not args.dryrun:
            holohub_cli_util.fatal(f"Project output directory {args.directory} does not exist")

        # Define minimal context with required fields
        context = {
            "project_name": args.project,
            "project_slug": args.project.lower().replace(" ", "_"),
            "language": args.language.lower() if args.language else None,  # Only set if provided
            "holoscan_version": HoloHubContainer.BASE_SDK_VERSION,
            "year": datetime.datetime.now().year,
        }

        # Add any additional context variables from command line
        if args.context:
            for ctx_var in args.context:
                try:
                    key, value = ctx_var.split("=", 1)
                    context[key] = value
                except ValueError:
                    holohub_cli_util.fatal(
                        f"Invalid context variable format: {ctx_var}. Expected key=value"
                    )

        # Print summary if dryrun
        if args.dryrun:
            print(Color.green("Would create project folder with these parameters (dryrun):"))
            print(f"Directory: {args.directory / context['project_slug']}")
            for key, value in context.items():
                print(f"  {key}: {value}")
            if args.directory == self.HOLOHUB_ROOT / "applications":
                print(Color.green("Would modify `applications/CMakeLists.txt`: "))
                print(f"    add_holohub_application({context['project_slug']})")
            return

        try:
            import cookiecutter.main
        except ImportError:
            self._install_template_deps(args.dryrun)

        import cookiecutter.main

        project_dir = args.directory / context["project_slug"]
        if project_dir.exists():
            holohub_cli_util.fatal(f"Project directory {project_dir} already exists")

        try:
            # Let cookiecutter handle all file generation
            cookiecutter.main.cookiecutter(
                str(template_dir),
                no_input=not args.interactive,
                extra_context=context,
                output_dir=str(args.directory),
            )
        except Exception as e:
            holohub_cli_util.fatal(f"Failed to create project: {str(e)}")

        # Add to CMakeLists.txt if in applications directory
        if args.directory == self.HOLOHUB_ROOT / "applications":
            self._add_to_cmakelists(context["project_slug"])

        # Get the actual project directory after cookiecutter runs
        project_dir = args.directory / context["project_slug"]
        metadata_path = project_dir / "metadata.json"
        src_dir = project_dir / "src"
        main_file = next(src_dir.glob(f"{context['project_slug']}.*"), None)

        msg_next = ""
        if "applications" in args.template:
            msg_next = (
                f"Possible next steps:\n"
                f"- Add operators to {main_file}\n"
                f"- Update project metadata in {metadata_path}\n"
                f"- Review source code license files and headers (e.g. {project_dir / 'LICENSE'})\n"
                f"- Build and run the application:\n"
                f"   {self.script_name} run {context['project_slug']}"
            )

        print(
            Color.green(f"Successfully created new project: {args.project}"),
            f"\nDirectory: {project_dir}\n\n{msg_next}",
        )

    def _suggest_command(self, invalid_command: str) -> list[str]:
        """Suggest similar command names using existing levenshtein_distance utility"""
        available_commands = list(self.subparsers.keys())
        distances = [
            (cmd, holohub_cli_util.levenshtein_distance(invalid_command, cmd))
            for cmd in available_commands
        ]
        distances.sort(key=lambda x: x[1])
        return [cmd for cmd, dist in distances[:2] if dist <= 2]  # Show up to 2 matches

    def _check_for_dash_prefix_issue(self, cmd_args: List[str]) -> Optional[str]:
        """
        Check if the parsing error is likely due to dash-prefixed arguments
        """
        DASH_VALUE_ARGS = ["--run-args", "--build-args", "--docker-opts", "--configure-args"]
        for i, arg in enumerate(cmd_args):
            if arg in DASH_VALUE_ARGS and "=" not in arg:
                if i + 1 < len(cmd_args) and cmd_args[i + 1].startswith("-"):
                    next_arg = cmd_args[i + 1]
                    return (
                        f"💡 Tip: ambiguous dash-prefixed arguments, use the equals format:\n"
                        f"   Instead of: {arg} {next_arg}\n"
                        f"   Use: {arg}={next_arg}"
                    )
        return None

    def run(self) -> None:
        """Main entry point for the CLI"""

        trailing_docker_args = []  # Handle " -- " separator for run-container command forwarding
        cmd_args = sys.argv[1:]  # Skip script name, return a copy of the args
        if len(cmd_args) >= 2 and cmd_args[0] == "run-container" and "--" in cmd_args:
            sep = cmd_args.index("--")
            cmd_args, trailing_docker_args = cmd_args[:sep], cmd_args[sep + 1 :]

        potential_command = cmd_args[0] if cmd_args else None
        dash_suggestion = None
        if potential_command and potential_command in self.subparsers:
            dash_suggestion = self._check_for_dash_prefix_issue(cmd_args)

        try:
            args = self.parser.parse_args(cmd_args)
            if trailing_docker_args:
                args._trailing_args = trailing_docker_args  # " -- " used for run-container command
        except SystemExit as e:
            if len(cmd_args) > 0 and e.code != 0:  # exit code is 0 => help was successfully shown
                if dash_suggestion:
                    print(f"\n{dash_suggestion}\n", file=sys.stderr)

                if potential_command and not potential_command.startswith("-"):
                    if potential_command in self.subparsers:
                        # Valid subcommand but parsing failed
                        print(f"\n💡 For more help with '{potential_command}':", file=sys.stderr)
                        print(f"  {self.script_name} {potential_command} --help\n", file=sys.stderr)
                        sys.exit(e.code if e.code is not None else 1)
                    else:  # Invalid subcommand - suggest similar ones
                        suggestions = self._suggest_command(potential_command)
                        if suggestions:
                            print("\n💡 Did you mean:", file=sys.stderr)
                            for cmd in suggestions:
                                print(f"  {self.script_name} {cmd}", file=sys.stderr)
                            print(file=sys.stderr)
                        sys.exit(1)
            raise
        if hasattr(args, "func"):
            args.func(args)
        else:
            self.parser.print_help()
            sys.exit(1)


def main():
    cli = HoloHubCLI()
    cli.run()


if __name__ == "__main__":
    main()
