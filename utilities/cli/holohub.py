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

import argparse
import datetime
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import utilities.cli.util as holohub_cli_util
import utilities.metadata.gather_metadata as metadata_util
from utilities.cli.container import HoloHubContainer, base_sdk_version
from utilities.cli.util import Color


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


class HoloHubCLI:
    """Command-line interface for HoloHub"""

    HOLOHUB_ROOT = Path(__file__).parent.parent.parent
    DEFAULT_BUILD_PARENT_DIR = HOLOHUB_ROOT / "build"
    DEFAULT_DATA_DIR = HOLOHUB_ROOT / "data"
    DEFAULT_SDK_DIR = "/opt/nvidia/holoscan/lib"

    def __init__(self):
        self.parser = self._create_parser()
        self._collect_metadata()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser with all supported commands"""
        parser = argparse.ArgumentParser(
            description="HoloHub CLI tool for managing HoloHub applications and containers"
        )
        subparsers = parser.add_subparsers(dest="command", required=True)

        # Add create command
        create = subparsers.add_parser("create", help="Create a new Holoscan application")
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
            "build-container", help="Build the development container"
        )
        build_container.add_argument("project", nargs="?", help="Project to build container for")
        build_container.add_argument("--base_img", help="Fully qualified base image name")
        build_container.add_argument("--docker_file", help="Path to Dockerfile to use")
        build_container.add_argument("--img", help="Specify fully qualified container name")
        build_container.add_argument(
            "--verbose", action="store_true", help="Print variables passed to docker build command"
        )
        build_container.add_argument(
            "--no-cache", action="store_true", help="Do not use cache when building the image"
        )
        build_container.add_argument(
            "--build-args", help="Provides extra arguments to docker build command"
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
            "run-container", help="Build and launch the development container"
        )
        run_container.add_argument("project", nargs="?", help="Project to run container for")
        run_container.add_argument("--img", help="Fully qualified image name")
        run_container.add_argument(
            "--local_sdk_root",
            help="Path to Holoscan SDK used for building local Holoscan SDK container",
        )
        run_container.add_argument("--init", action="store_true", help="Support tini entry point")
        run_container.add_argument(
            "--persistent", action="store_true", help="Does not delete container after it is run"
        )
        run_container.add_argument(
            "--verbose", action="store_true", help="Print variables passed to docker run command"
        )
        run_container.add_argument(
            "--dryrun", action="store_true", help="Print commands without executing them"
        )
        run_container.add_argument("--add-volume", action="append", help="Mount additional volume")
        run_container.add_argument(
            "--as_root", action="store_true", help="Run the container with root permissions"
        )
        run_container.add_argument(
            "--docker_opts", help="Additional options to pass to the Docker launch"
        )
        run_container.add_argument(
            "--language", choices=["cpp", "python"], help="Specify language implementation"
        )
        run_container.set_defaults(func=self.handle_run_container)

        # build command
        build = subparsers.add_parser("build", help="Build a project")
        build.add_argument("project", help="Project to build")
        build.add_argument(
            "--local", action="store_true", help="Build locally instead of in container"
        )
        build.add_argument("--verbose", action="store_true", help="Print extra output")
        build.add_argument(
            "--build-type",
            choices=["debug", "release", "rel-debug"],
            help="Build type (debug, release, rel-debug)",
        )
        build.add_argument(
            "--dryrun", action="store_true", help="Print commands without executing them"
        )
        build.set_defaults(func=self.handle_build)

        # run command
        run = subparsers.add_parser("run", help="Build and run a project")
        run.add_argument("project", help="Project to run")
        run.add_argument("--local", action="store_true", help="Run locally instead of in container")
        run.add_argument("--verbose", action="store_true", help="Print extra output")
        run.add_argument(
            "--dryrun", action="store_true", help="Print commands without executing them"
        )
        run.add_argument(
            "--nsys-profile", action="store_true", help="Enable Nsight Systems profiling"
        )
        run.add_argument(
            "--language", choices=["cpp", "python"], help="Specify language implementation"
        )
        run.set_defaults(func=self.handle_run)

        # list command
        list_cmd = subparsers.add_parser("list", help="List all available targets")
        list_cmd.set_defaults(func=self.handle_list)

        # lint command
        lint = subparsers.add_parser("lint", help="Run linting tools")
        lint.add_argument("path", nargs="?", default=".", help="Path to lint")
        lint.add_argument("--fix", action="store_true", help="Fix linting issues")
        lint.add_argument(
            "--install-dependencies", action="store_true", help="Install linting dependencies"
        )
        lint.add_argument(
            "--dryrun", action="store_true", help="Print commands without executing them"
        )
        lint.set_defaults(func=self.handle_lint)

        # setup command
        setup = subparsers.add_parser("setup", help="Install HoloHub main required packages")
        setup.add_argument(
            "--dryrun", action="store_true", help="Print commands without executing them"
        )
        setup.set_defaults(func=self.handle_setup)

        # Add install command
        install = subparsers.add_parser("install", help="Install a project")
        install.add_argument("project", help="Project to install")
        install.add_argument(
            "--local", action="store_true", help="Install locally instead of in container"
        )
        install.add_argument(
            "--build-type",
            choices=["debug", "release", "rel-debug"],
            help="Build type (debug, release, rel-debug)",
        )
        install.add_argument(
            "--dryrun", action="store_true", help="Print commands without executing them"
        )
        install.set_defaults(func=self.handle_install)

        # Add test command
        test = subparsers.add_parser("test", help="Test a project")
        test.add_argument("project", nargs="?", help="Project to test")
        test.add_argument("--base_img", help="Fully qualified base image name")
        test.add_argument("--build_args", help="Additional options to pass to the Docker build")
        test.add_argument("--verbose", action="store_true", help="Print extra output")
        test.add_argument(
            "--dryrun", action="store_true", help="Print commands without executing them"
        )
        test.add_argument("--clear_cache", action="store_true", help="Clear cache folders")
        test.add_argument("--site_name", help="Site name")
        test.add_argument("--cdash_url", help="CDash URL")
        test.add_argument("--platform_name", help="Platform name")
        test.add_argument("--cmake_options", help="CMake options")
        test.add_argument("--no_xvfb", action="store_true", help="Do not use xvfb")
        test.add_argument("--ctest_script", help="CTest script")
        test.set_defaults(func=self.handle_test)

        # Add clear-cache command
        clear_cache = subparsers.add_parser("clear-cache", help="Clear cache folders")
        clear_cache.add_argument(
            "--dryrun", action="store_true", help="Print commands without executing them"
        )
        clear_cache.set_defaults(func=self.handle_clear_cache)

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

    def _find_project(self, project_name: str, language: Optional[str] = None) -> dict:
        """Find a project by name"""
        normalized_language = holohub_cli_util.normalize_language(language) if language else None

        # First try exact match
        for project in self.projects:
            if project["project_name"] == project_name:
                if (
                    normalized_language
                    and holohub_cli_util.normalize_language(project["metadata"]["language"])
                    != normalized_language
                ):
                    continue
                return project
        # If project not found, suggest similar names
        distances = [
            (
                p["project_name"],
                holohub_cli_util.levenshtein_distance(project_name, p["project_name"]),
                p.get("source_folder", ""),
            )
            for p in self.projects
        ]
        distances.sort(key=lambda x: x[1])  # Sort by distance
        closest_matches = [
            (name, folder) for name, dist, folder in distances[:1] if dist <= 3
        ]  # Get the closest match with distance <= 3
        msg = f"Project '{project_name}' not found."
        if closest_matches:
            msg += f"\nDid you mean: '{closest_matches[0][0]}' (source: {closest_matches[0][1]})"
        holohub_cli_util.fatal(msg)
        return None

    def _make_project_container(
        self, project_name: Optional[str] = None, language: Optional[str] = None
    ) -> HoloHubContainer:
        """Define a project container"""
        if not project_name:
            return HoloHubContainer(project_metadata=None)
        project_data = self._find_project(project_name=project_name, language=language)
        return HoloHubContainer(project_metadata=project_data)

    def handle_build_container(self, args: argparse.Namespace) -> None:
        """Handle build-container command"""
        container = self._make_project_container(
            project_name=args.project, language=args.language if hasattr(args, "language") else None
        )
        container.dryrun = args.dryrun
        container.build(
            docker_file=args.docker_file,
            base_img=args.base_img,
            img=args.img,
            no_cache=args.no_cache,
            build_args=args.build_args,
        )

    def handle_run_container(self, args: argparse.Namespace) -> None:
        """Handle run-container command"""
        container = self._make_project_container(
            project_name=args.project, language=args.language if hasattr(args, "language") else None
        )

        container.dryrun = args.dryrun
        container.build()
        container.run(
            img=args.img,
            local_sdk_root=args.local_sdk_root,
            use_tini=args.init,
            persistent=args.persistent,
            as_root=args.as_root,
            docker_opts=args.docker_opts,
            add_volumes=args.add_volume,
            verbose=args.verbose,
        )

    def handle_test(self, args: argparse.Namespace) -> None:
        """Handle test command"""
        container = self._make_project_container(
            project_name=args.project, language=args.language if hasattr(args, "language") else None
        )

        if args.clear_cache:
            for pattern in ["build", "build-*", "install"]:
                for path in HoloHubCLI.HOLOHUB_ROOT.glob(pattern):
                    if path.is_dir():
                        if args.dryrun:
                            print(f"  {Color.yellow('Would remove:')} {path}")
                        else:
                            shutil.rmtree(path)

        container.dryrun = args.dryrun

        container.build(
            base_img=args.base_img,
            build_args=args.build_args,
        )

        # Construct the ctest command line
        # If we should run without xvfb
        xvfb = "xvfb-run -a"
        if args.no_xvfb:
            xvfb = ""

        base_img = args.base_img or container.default_base_image()

        img_tag = base_img.split(":")[-1]

        ctest_cmd = f"{xvfb} ctest " f"-DAPP={args.project} " f"-DTAG={img_tag} "

        if args.cmake_options:
            ctest_cmd += f'-DCONFIGURE_OPTIONS="{args.cmake_options}" '

        if args.cdash_url:
            ctest_cmd += f"-DCTEST_SUBMIT_URL={args.cdash_url} "

        if args.site_name:
            ctest_cmd += f"-DCTEST_SITE={args.site_name} "

        if args.platform_name:
            ctest_cmd += f"-DPLATFORM_NAME={args.platform_name} "

        if args.ctest_script:
            ctest_cmd += f"-S {args.ctest_script} "
        else:
            ctest_cmd += "-S utilities/testing/holohub.container.ctest "

        if args.verbose:
            ctest_cmd += "-VV "

        container.run(
            use_tini=True,
            docker_opts="--entrypoint=bash",
            extra_args=["-c", ctest_cmd],
            verbose=args.verbose,
        )

    def get_buildtype_str(self, build_type: Optional[str]) -> str:
        """Get CMake build type string"""
        if not build_type:
            return os.environ.get("CMAKE_BUILD_TYPE", "Release")

        build_type_map = {
            "debug": "Debug",
            "release": "Release",
            "rel-debug": "RelWithDebInfo",
            "relwithdebinfo": "RelWithDebInfo",
        }
        return build_type_map.get(build_type.lower(), "Release")

    def _build_project_locally(
        self,
        project_name: str,
        language: Optional[str] = None,
        build_type: Optional[str] = None,
        dryrun: bool = False,
    ) -> tuple[Path, dict]:
        """Helper method to build a project locally"""
        project_data = self._find_project(project_name=project_name, language=language)
        build_type = self.get_buildtype_str(build_type)
        build_dir = HoloHubCLI.DEFAULT_BUILD_PARENT_DIR / project_name
        build_dir.mkdir(parents=True, exist_ok=True)

        holohub_cli_util.run_command(
            [
                "cmake",
                "-B",
                str(build_dir),
                "-S",
                str(HoloHubCLI.HOLOHUB_ROOT),
                "-G",
                "Ninja",
                f"-DCMAKE_BUILD_TYPE={build_type}",
                f"-DCMAKE_PREFIX_PATH={HoloHubCLI.DEFAULT_SDK_DIR}",
                f"-DHOLOHUB_DATA_DIR:PATH={HoloHubCLI.DEFAULT_DATA_DIR}",
                f"-DAPP_{project_name}=ON",
            ],
            dry_run=dryrun,
        )
        holohub_cli_util.run_command(
            ["cmake", "--build", str(build_dir), "--config", build_type], dry_run=dryrun
        )

        return build_dir, project_data

    def handle_build(self, args: argparse.Namespace) -> None:
        """Handle build command"""
        if args.local or os.environ.get("HOLOHUB_BUILD_LOCAL"):
            self._build_project_locally(
                project_name=args.project,
                language=args.language if hasattr(args, "language") else None,
                build_type=args.build_type,
                dryrun=args.dryrun,
            )
        else:
            # Build in container
            container = self._make_project_container(
                project_name=args.project,
                language=args.language if hasattr(args, "language") else None,
            )
            container.dryrun = args.dryrun
            container.build()

            container.run(
                docker_opts="--entrypoint=bash",
                extra_args=["-c", f"./holohub build {args.project} --local"],
                verbose=args.verbose,
            )

    def handle_run(self, args: argparse.Namespace) -> None:
        """Handle run command"""
        if args.local or os.environ.get("HOLOHUB_BUILD_LOCAL"):
            build_dir, project_data = self._build_project_locally(
                project_name=args.project,
                language=args.language if hasattr(args, "language") else None,
                build_type="Release",  # Default to Release for run
                dryrun=args.dryrun,
            )

            language = holohub_cli_util.normalize_language(
                project_data.get("metadata", {}).get("language", None)
            )

            run_config = project_data.get("metadata", {}).get("run", {})
            if not run_config:
                holohub_cli_util.fatal(
                    f"Project '{args.project}' does not have a run configuration"
                )

            app_source_path = project_data.get("source_folder", "")

            # Process command template
            cmd = run_config["command"]
            cmd = cmd.replace("<holohub_data_dir>", str(HoloHubCLI.DEFAULT_DATA_DIR))
            cmd = cmd.replace("<holohub_app_source>", str(app_source_path))

            cmd = cmd.replace("<holohub_bin>", str(build_dir))

            app_build_dir = build_dir / app_source_path.relative_to(HoloHubCLI.HOLOHUB_ROOT)
            cmd = cmd.replace("<holohub_app_bin>", str(app_build_dir))

            if language == "cpp":
                if not build_dir.is_dir() and not args.dryrun:
                    holohub_cli_util.fatal(
                        f"The build directory {build_dir} for this application does not exist.\n"
                        f"Did you forget to './holohub build {args.project}'?"
                    )

            # Handle workdir
            workdir = run_config.get("workdir", "holohub_app_bin")
            if workdir == "holohub_app_source":
                if not args.dryrun:
                    os.chdir(project_data.get("source_folder", ""))
                print(
                    f"{Color.blue(holohub_cli_util.get_timestamp())} {Color.white('$')} {Color.green('cd ' + str(project_data.get('source_folder', '')))}"
                )
            elif workdir == "holohub_bin":
                if not args.dryrun:
                    os.chdir(build_dir)
                print(
                    f"{Color.blue(holohub_cli_util.get_timestamp())} {Color.white('$')} {Color.green('cd ' + str(build_dir))}"
                )
            else:  # default to app binary directory
                target_dir = (
                    build_dir if language == "cpp" else project_data.get("source_folder", "")
                )
                if not args.dryrun:
                    os.chdir(target_dir)
                print(
                    f"{Color.blue(holohub_cli_util.get_timestamp())} {Color.white('$')} {Color.green('cd ' + str(target_dir))}"
                )

            # Set up environment
            env = os.environ.copy()
            env["PYTHONPATH"] = (
                f"{os.environ.get('PYTHONPATH', '')}:{HoloHubCLI.DEFAULT_SDK_DIR}/../python/lib:{build_dir}/python/lib:{HoloHubCLI.HOLOHUB_ROOT}"
            )
            env["HOLOHUB_DATA_PATH"] = str(HoloHubCLI.DEFAULT_DATA_DIR)
            env["HOLOSCAN_INPUT_PATH"] = os.environ.get(
                "HOLOSCAN_INPUT_PATH", str(HoloHubCLI.DEFAULT_DATA_DIR)
            )

            # Print environment setup
            if args.verbose or args.dryrun:
                print(
                    f"{Color.blue(holohub_cli_util.get_timestamp())} {Color.white('$')} {Color.green('export PYTHONPATH=' + env['PYTHONPATH'])}"
                )
                print(
                    f"{Color.blue(holohub_cli_util.get_timestamp())} {Color.white('$')} {Color.green('export HOLOHUB_DATA_PATH=' + env['HOLOHUB_DATA_PATH'])}"
                )
                print(
                    f"{Color.blue(holohub_cli_util.get_timestamp())} {Color.white('$')} {Color.green('export HOLOSCAN_INPUT_PATH=' + env['HOLOSCAN_INPUT_PATH'])}"
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

            holohub_cli_util.run_command(cmd.split(), env=env, dry_run=args.dryrun)
        else:
            container = self._make_project_container(
                project_name=args.project,
                language=args.language if hasattr(args, "language") else None,
            )
            container.dryrun = args.dryrun
            container.build()
            # Get language before launching container
            language = holohub_cli_util.normalize_language(
                container.project_metadata.get("metadata", {}).get("language", None)
            )

            # Build command with all necessary arguments
            run_cmd = f"./holohub run {args.project} --language {language} --local"
            if args.verbose:
                run_cmd += " --verbose"
            if args.nsys_profile:
                run_cmd += " --nsys-profile"

            container.run(
                docker_opts="--entrypoint=bash", extra_args=["-c", run_cmd], verbose=args.verbose
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
        EXCLUDE_PATHS = ["applications/holoviz/template"]
        # Known exceptions, such as template files that do not represent a standalone project

        app_paths = (
            HoloHubCLI.HOLOHUB_ROOT / "applications",
            HoloHubCLI.HOLOHUB_ROOT / "benchmarks",
            HoloHubCLI.HOLOHUB_ROOT / "gxf_extensions",
            HoloHubCLI.HOLOHUB_ROOT / "operators",
            HoloHubCLI.HOLOHUB_ROOT / "pkg",
            HoloHubCLI.HOLOHUB_ROOT / "workflows",
        )
        metadata = metadata_util.gather_metadata(app_paths, exclude_paths=EXCLUDE_PATHS)
        grouped_metadata = defaultdict(list)
        for project in metadata:
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

    def handle_lint(self, args: argparse.Namespace) -> None:
        """Handle lint command"""
        if args.install_dependencies:
            self._install_lint_deps(args.dryrun)
            return

        exit_code = 0

        # Change to script directory
        os.chdir(HoloHubCLI.HOLOHUB_ROOT)

        if args.fix:
            # Fix Python
            holohub_cli_util.run_command(
                ["ruff", "check", "--fix", "--ignore", "E712", args.path],
                check=False,
                dry_run=args.dryrun,
            )
            holohub_cli_util.run_command(["isort", args.path], check=False, dry_run=args.dryrun)
            holohub_cli_util.run_command(["black", args.path], check=False, dry_run=args.dryrun)
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
                    "--skip=*.onnx,*.min.js,*.min.js.map,Contrastive_learning_Notebook.ipynb,./data",
                ],
                check=False,
                dry_run=args.dryrun,
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
                    "--recursive",
                    args.path,
                ]
                if args.dryrun:
                    holohub_cli_util.run_command(cmd, dry_run=True)
                    cpp_files = ""
                else:
                    cpp_files = subprocess.check_output(cmd, stderr=subprocess.PIPE, text=True)

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
                ).returncode
                != 0
            ):
                exit_code = 1
            if (
                holohub_cli_util.run_command(
                    ["isort", "-c", args.path], check=False, dry_run=args.dryrun
                ).returncode
                != 0
            ):
                exit_code = 1
            if (
                holohub_cli_util.run_command(
                    ["black", "--check", args.path], check=False, dry_run=args.dryrun
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
                        "--recursive",
                        args.path,
                    ],
                    check=False,
                    dry_run=args.dryrun,
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
                ).returncode
                != 0
            ):
                exit_code = 1

            print(Color.blue("Linting CMake"))
            cmake_files = list(Path(args.path).rglob("CMakeLists.txt"))
            cmake_files.extend(Path(args.path).rglob("*.cmake"))
            if cmake_files:
                if (
                    holohub_cli_util.run_command(
                        [
                            "cmakelint",
                            "--filter=-whitespace/indent,-linelength,-readability/wonkycase,-convention/filename,-package/stdargs",
                            *[str(f) for f in cmake_files],
                        ],
                        check=False,
                        dry_run=args.dryrun,
                    ).returncode
                    != 0
                ):
                    exit_code = 1

        if exit_code == 0 and not args.dryrun:
            print(Color.green("Everything looks good!"))
        sys.exit(exit_code)

    def _install_lint_deps(self, dry_run: bool = False) -> None:
        """Install linting dependencies"""
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
        holohub_cli_util.run_command(
            ["apt", "install", "--no-install-recommends", "-y", "clang-format=1:14.0*"],
            dry_run=dry_run,
        )

    def _install_template_deps(self, dry_run: bool = False) -> None:
        """Install template dependencies"""
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
        # Install system dependencies
        holohub_cli_util.run_command(["sudo", "apt-get", "update"], dry_run=args.dryrun)

        # Install wget if not present
        holohub_cli_util.run_command(
            ["sudo", "apt-get", "install", "-y", "wget"], dry_run=args.dryrun
        )

        # Install xvfb for running tests/examples headless
        holohub_cli_util.run_command(
            ["sudo", "apt-get", "install", "-y", "xvfb"], dry_run=args.dryrun
        )

        # Check and install CMake if needed
        cmake_version = subprocess.check_output(
            ["dpkg", "--status", "cmake"], text=True, stderr=subprocess.DEVNULL
        )
        ubuntu_codename = subprocess.check_output(["cat", "/etc/os-release"], text=True)
        ubuntu_codename = re.search(r"UBUNTU_CODENAME=(\w+)", ubuntu_codename).group(1)

        if not cmake_version or "3.24.0" > cmake_version:
            holohub_cli_util.run_command(
                ["sudo", "apt", "install", "--no-install-recommends", "-y", "gpg"],
                dry_run=args.dryrun,
            )
            holohub_cli_util.run_command(
                [
                    "wget",
                    "-O",
                    "-",
                    "https://apt.kitware.com/keys/kitware-archive-latest.asc",
                    "2>/dev/null",
                    "|",
                    "gpg",
                    "--dearmor",
                    "-",
                    "|",
                    "sudo",
                    "tee",
                    "/usr/share/keyrings/kitware-archive-keyring.gpg",
                    ">/dev/null",
                ],
                dry_run=args.dryrun,
            )
            holohub_cli_util.run_command(
                [
                    "echo",
                    f'"deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ {ubuntu_codename} main"',
                    "|",
                    "sudo",
                    "tee",
                    "/etc/apt/sources.list.d/kitware.list",
                    ">/dev/null",
                ],
                dry_run=args.dryrun,
            )
            holohub_cli_util.run_command(["sudo", "apt-get", "update"], dry_run=args.dryrun)
            holohub_cli_util.run_command(
                [
                    "sudo",
                    "apt",
                    "install",
                    "--no-install-recommends",
                    "-y",
                    "cmake",
                    "cmake-curses-gui",
                ],
                dry_run=args.dryrun,
            )

        # Install Ninja
        holohub_cli_util.run_command(
            ["sudo", "apt", "install", "--no-install-recommends", "-y", "ninja-build"],
            dry_run=args.dryrun,
        )

        # Install Python dev
        python3_dev_version = subprocess.check_output(
            ["dpkg", "--status", "python3-dev"], text=True, stderr=subprocess.DEVNULL
        )
        if not python3_dev_version or "3.9.0" > python3_dev_version:
            holohub_cli_util.run_command(
                [
                    "sudo",
                    "apt",
                    "install",
                    "--no-install-recommends",
                    "-y",
                    "python3",
                    "python3-dev",
                ],
                dry_run=args.dryrun,
            )

        # Install ffmpeg
        holohub_cli_util.run_command(
            ["sudo", "apt", "install", "--no-install-recommends", "-y", "ffmpeg"],
            dry_run=args.dryrun,
        )

        # Install libv4l-dev
        holohub_cli_util.run_command(
            ["sudo", "apt-get", "install", "--no-install-recommends", "-y", "libv4l-dev"],
            dry_run=args.dryrun,
        )

        # Install git if not present
        holohub_cli_util.run_command(
            ["sudo", "apt-get", "install", "--no-install-recommends", "-y", "git"],
            dry_run=args.dryrun,
        )

        # Install unzip if not present
        holohub_cli_util.run_command(
            ["sudo", "apt-get", "install", "--no-install-recommends", "-y", "unzip"],
            dry_run=args.dryrun,
        )

        # Install ngc-cli if not present
        try:
            subprocess.check_output(["ngc", "--version"], stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, FileNotFoundError):
            if platform.machine() == "aarch64":
                holohub_cli_util.run_command(
                    [
                        "wget",
                        "--content-disposition",
                        "https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.64.3/files/ngccli_arm64.zip",
                    ],
                    dry_run=args.dryrun,
                )
                holohub_cli_util.run_command(["unzip", "ngccli_arm64.zip"], dry_run=args.dryrun)
            else:
                holohub_cli_util.run_command(
                    [
                        "wget",
                        "--content-disposition",
                        "https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.64.3/files/ngccli_linux.zip",
                    ],
                    dry_run=args.dryrun,
                )
                holohub_cli_util.run_command(["unzip", "ngccli_linux.zip"], dry_run=args.dryrun)
            holohub_cli_util.run_command(["chmod", "u+x", "ngc-cli/ngc"], dry_run=args.dryrun)
            holohub_cli_util.run_command(
                ["sudo", "ln", "-s", f"{os.getcwd()}/ngc-cli/ngc", "/usr/local/bin/"],
                dry_run=args.dryrun,
            )

        # Install CUDA dependencies
        for version in [
            "12-6",
            "12-5",
            "12-4",
            "12-3",
            "12-2",
            "12-1",
            "12-0",
            "11-8",
            "11-7",
            "11-6",
            "11-4",
        ]:
            try:
                cuda_version = subprocess.check_output(
                    ["dpkg", "--status", f"cuda-cudart-{version}"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                if cuda_version:
                    break
            except subprocess.CalledProcessError:
                continue

        if cuda_version:
            short_cuda_version = cuda_version.split(".")[0]

            # Install cudnn9 first
            holohub_cli_util.install_cuda_dependencies_package(
                "libcudnn9-cuda-12", "9.*", optional=True, dry_run=args.dryrun
            )
            holohub_cli_util.install_cuda_dependencies_package(
                "libcudnn9-dev-cuda-12", "9.*", optional=True, dry_run=args.dryrun
            )

            # Check if cudnn9 is installed, if not install cudnn8
            installed_cudnn9_version = subprocess.check_output(
                ["apt", "list", "--installed", "libcudnn9-cuda-12"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            if not installed_cudnn9_version:
                holohub_cli_util.install_cuda_dependencies_package(
                    "libcudnn8", f"cuda{short_cuda_version}", dry_run=args.dryrun
                )
                holohub_cli_util.install_cuda_dependencies_package(
                    "libcudnn8-dev", f"cuda{short_cuda_version}", dry_run=args.dryrun
                )

            # Install TensorRT dependencies
            installed_libnvinferbin = subprocess.check_output(
                ["dpkg", "--status", "libnvinfer-bin"], text=True, stderr=subprocess.DEVNULL
            )
            # Extract version string using regex
            version_match = re.search(
                r"Version: (\d+\.\d+\.\d+)\.\d+-\d+\+cuda\d+\.\d+", installed_libnvinferbin
            )
            if version_match:
                installed_libnvinferbin_version = version_match.group(1)
            else:
                holohub_cli_util.fatal("Could not determine libnvinfer-bin version")

            holohub_cli_util.install_cuda_dependencies_package(
                "libnvinfer-headers-dev", installed_libnvinferbin_version, dry_run=args.dryrun
            )
            holohub_cli_util.install_cuda_dependencies_package(
                "libnvinfer-dev", installed_libnvinferbin_version, dry_run=args.dryrun
            )
            holohub_cli_util.install_cuda_dependencies_package(
                "libnvinfer-plugin-dev", installed_libnvinferbin_version, dry_run=args.dryrun
            )
            holohub_cli_util.install_cuda_dependencies_package(
                "libnvonnxparsers-dev", installed_libnvinferbin_version, dry_run=args.dryrun
            )

        # Install the autocomplete
        holohub_cli_util.run_command(
            [
                "sudo",
                "cp",
                f"{HoloHubCLI.HOLOHUB_ROOT}/utilities/holohub_autocomplete",
                "/etc/bash_completion.d/",
            ],
            dry_run=args.dryrun,
        )

        print(Color.green("Setup for HoloHub is ready. Happy Holocoding!"))

    def handle_install(self, args: argparse.Namespace) -> None:
        """Handle install command"""
        raise NotImplementedError("Install command not yet implemented")

    def handle_clear_cache(self, args: argparse.Namespace) -> None:
        """Handle clear-cache command"""
        if args.dryrun:
            print(Color.blue("Would clear cache folders:"))
        else:
            print(Color.blue("Clearing cache..."))
        for pattern in ["build", "build-*", "install"]:
            for path in HoloHubCLI.HOLOHUB_ROOT.glob(pattern):
                if path.is_dir():
                    if args.dryrun:
                        print(f"  {Color.yellow('Would remove:')} {path}")
                    else:
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
            "holoscan_version": base_sdk_version,
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
                f"   ./holohub run {context['project_slug']}"
            )

        print(
            Color.green(f"Successfully created new project: {args.project}"),
            f"\nDirectory: {project_dir}\n\n{msg_next}",
        )

    def run(self) -> None:
        """Main entry point for the CLI"""
        args = self.parser.parse_args()
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
