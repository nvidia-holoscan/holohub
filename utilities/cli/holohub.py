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
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

import utilities.cli.util as holohub_cli_util
import utilities.metadata.gather_metadata as metadata_util
from utilities.cli.container import HoloHubContainer, base_sdk_version
from utilities.cli.util import Color, PackageInstallationError, parse_semantic_version

PYTHON_MIN_VERSION = "3.9.0"


class HoloHubCLI:
    """Command-line interface for HoloHub"""

    HOLOHUB_ROOT = Path(__file__).parent.parent.parent
    DEFAULT_BUILD_PARENT_DIR = HOLOHUB_ROOT / "build"
    DEFAULT_DATA_DIR = HOLOHUB_ROOT / "data"
    DEFAULT_SDK_DIR = "/opt/nvidia/holoscan/lib"

    def __init__(self):
        self.script_name = os.environ.get("HOLOHUB_CMD_NAME", "./holohub")
        self.parser = self._create_parser()
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
            choices=["debug", "release", "rel-debug"],
            help="Build type (debug, release, rel-debug)",
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
        setup = subparsers.add_parser("setup", help="Install HoloHub main required packages")
        self.subparsers["setup"] = setup
        setup.add_argument(
            "--dryrun", action="store_true", help="Print commands without executing them"
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
        install.add_argument(
            "--local", action="store_true", help="Install locally instead of in container"
        )
        install.add_argument(
            "--build-type",
            choices=["debug", "release", "rel-debug"],
            help="Build type (debug, release, rel-debug)",
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
        test.add_argument("--cmake-options", help="CMake options")
        test.add_argument("--no-xvfb", action="store_true", help="Do not use xvfb")
        test.add_argument("--ctest-script", help="CTest script")
        test.add_argument(
            "--no-docker-build", action="store_true", help="Skip building the container"
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

    def _make_project_container(
        self, project_name: Optional[str] = None, language: Optional[str] = None
    ) -> HoloHubContainer:
        """Define a project container"""
        if not project_name:
            return HoloHubContainer(project_metadata=None)
        project_data = self.find_project(project_name=project_name, language=language)
        return HoloHubContainer(project_metadata=project_data)

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
            )
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
            docker_opts=args.docker_opts,
            add_volumes=args.add_volume,
            enable_mps=getattr(args, "mps", False),
            extra_args=getattr(args, "_trailing_args", []),  # forward trailing args --
        )

    def handle_test(self, args: argparse.Namespace) -> None:
        """Handle test command"""
        skip_docker_build, _ = holohub_cli_util.check_skip_builds(args)
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
        container.verbose = args.verbose

        if not skip_docker_build:
            container.build(
                docker_file=args.docker_file,
                base_img=args.base_img,
                img=args.img,
                no_cache=args.no_cache,
                build_args=args.build_args,
            )

        xvfb = "" if args.no_xvfb else "xvfb-run -a"

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
        if configure_args:
            cmake_args.extend(configure_args)

        holohub_cli_util.run_command(cmake_args, dry_run=dryrun)

        # Build the project with optional parallel jobs
        build_cmd = ["cmake", "--build", str(build_dir), "--config", build_type]
        if parallel:
            build_cmd.extend(["-j", parallel])
        else:
            build_cmd.append("-j")  # Use default number of jobs

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

        if args.local or os.environ.get("HOLOHUB_BUILD_LOCAL"):
            self.build_project_locally(
                project_name=args.project,
                language=args.language if hasattr(args, "language") else None,
                build_type=args.build_type,
                with_operators=args.with_operators,
                dryrun=args.dryrun,
                pkg_generator=getattr(args, "pkg_generator", "DEB"),
                parallel=getattr(args, "parallel", None),
                benchmark=getattr(args, "benchmark", False),
                configure_args=getattr(args, "configure_args", None),
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
                    build_args=args.build_args,
                )

            # Build command with all necessary arguments
            build_cmd = f"{self.script_name} build {args.project} --local"
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
                    build_cmd += f' --configure-args "{configure_arg}"'

            img = getattr(args, "img", None) or container.image_name
            docker_opts = getattr(args, "docker_opts", "")
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

        if is_local_mode:
            if args.docker_opts:
                holohub_cli_util.fatal(
                    "Container arguments were provided with `--docker-opts` but a non-containerized build was requested."
                )

            if skip_local_build:
                # Skip building, but still need project metadata and build directory
                project_data = self.find_project(
                    project_name=args.project,
                    language=args.language if hasattr(args, "language") else None,
                )
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
                    build_type=args.build_type or "Release",  # Default to Release for run
                    with_operators=args.with_operators,
                    dryrun=args.dryrun,
                    pkg_generator=getattr(args, "pkg_generator", "DEB"),
                    parallel=getattr(args, "parallel", None),
                    configure_args=getattr(args, "configure_args", None),
                )

            language = holohub_cli_util.normalize_language(
                project_data.get("metadata", {}).get("language", None)
            )

            run_config = project_data.get("metadata", {}).get("run", {})
            if not run_config:
                holohub_cli_util.fatal(
                    f"Project '{args.project}' does not have a run configuration"
                )

            path_mapping = holohub_cli_util.build_holohub_path_mapping(
                holohub_root=HoloHubCLI.HOLOHUB_ROOT,
                project_data=project_data,
                build_dir=build_dir,
                data_dir=HoloHubCLI.DEFAULT_DATA_DIR,
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

            if hasattr(args, "run_args") and args.run_args:
                cmd_args = shlex.split(args.run_args)
                if isinstance(cmd, str):  # Ensure cmd is a list of arguments
                    cmd = shlex.split(cmd)
                cmd.extend(cmd_args)

            if language == "cpp":
                if not build_dir.is_dir() and not args.dryrun:
                    holohub_cli_util.fatal(
                        f"The build directory {build_dir} for this application does not exist.\n"
                        f"Did you forget to '{self.script_name} build {args.project}'?"
                    )

            # Handle workdir using the path mapping
            workdir_spec = run_config.get("workdir", "holohub_app_bin")
            if not workdir_spec:
                target_dir = Path(path_mapping.get("holohub_root", "."))
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
                    build_args=args.build_args,
                )
            language = holohub_cli_util.normalize_language(
                container.project_metadata.get("metadata", {}).get("language", None)
            )

            run_cmd = f"{self.script_name} run {args.project} --language {language} --local"
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
                    run_cmd += f' --configure-args "{configure_arg}"'

            img = getattr(args, "img", None) or container.image_name
            docker_opts = getattr(args, "docker_opts", "")
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

    def handle_lint(self, args: argparse.Namespace) -> None:
        """Handle lint command"""
        if args.install_dependencies:
            self._install_lint_deps(args.dryrun)
            return

        exit_code = 0

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
        holohub_cli_util.run_command(
            ["apt", "install", "--no-install-recommends", "-y", "clang-format=1:14.0*"],
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
        # Install system dependencies
        holohub_cli_util.run_command(["apt-get", "update"], dry_run=args.dryrun)

        # Install wget if not present
        holohub_cli_util.run_command(["apt-get", "install", "-y", "wget"], dry_run=args.dryrun)

        # Install xvfb for running tests/examples headless
        holohub_cli_util.run_command(["apt-get", "install", "-y", "xvfb"], dry_run=args.dryrun)

        # Check and install CMake if needed
        try:
            cmake_version = subprocess.run(
                ["dpkg", "--status", "cmake"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout
            version_match = re.search(r"Version: ([^-\s]+)", cmake_version)
            cmake_version = version_match.group(1) if version_match else ""
        except subprocess.CalledProcessError:
            cmake_version = ""

        ubuntu_codename = subprocess.check_output(["cat", "/etc/os-release"], text=True)
        ubuntu_codename = re.search(r"UBUNTU_CODENAME=(\w+)", ubuntu_codename).group(1)

        if not cmake_version or "3.24.0" > cmake_version:
            holohub_cli_util.run_command(
                ["apt", "install", "--no-install-recommends", "-y", "gpg"],
                dry_run=args.dryrun,
            )
            holohub_cli_util.run_command(
                "wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | "
                "gpg --dearmor - | "
                "tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null",
                shell=True,
                check=False,
                dry_run=args.dryrun,
            )
            holohub_cli_util.run_command(
                f'echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ {ubuntu_codename} main" | '
                "tee /etc/apt/sources.list.d/kitware.list >/dev/null",
                shell=True,
                dry_run=args.dryrun,
            )
            holohub_cli_util.run_command(["apt-get", "update"], dry_run=args.dryrun)
            holohub_cli_util.run_command(
                [
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
            ["apt", "install", "--no-install-recommends", "-y", "ninja-build"],
            dry_run=args.dryrun,
        )

        # Install Python dev
        try:
            python3_version = sys.version_info
            python3_dev_output = subprocess.run(
                ["dpkg", "--list", f"python3.{python3_version.minor}-dev"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout

            # Look for the specific version first, then fall back to python3-dev
            version_match = re.search(
                rf"python3\.{python3_version.minor}-dev\s+(\d+\.\d+\.\d+)", python3_dev_output
            )
            if not version_match:
                # Fall back to checking python3-dev
                python3_dev_output = subprocess.run(
                    ["dpkg", "--list", "python3-dev"],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout
                version_match = re.search(r"python3-dev\s+(\d+\.\d+\.\d+)", python3_dev_output)

            python3_dev_version = version_match.group(1) if version_match else ""
        except subprocess.CalledProcessError:
            python3_dev_version = ""

        if not python3_dev_version or parse_semantic_version(
            python3_dev_version
        ) < parse_semantic_version(PYTHON_MIN_VERSION):
            holohub_cli_util.run_command(
                [
                    "apt",
                    "install",
                    "--no-install-recommends",
                    "-y",
                    f"python3.{python3_version.minor}-dev",
                ],
                dry_run=args.dryrun,
            )

        # Install ffmpeg
        holohub_cli_util.run_command(
            ["apt", "install", "--no-install-recommends", "-y", "ffmpeg"],
            dry_run=args.dryrun,
        )

        # Install libv4l-dev
        holohub_cli_util.run_command(
            ["apt-get", "install", "--no-install-recommends", "-y", "libv4l-dev"],
            dry_run=args.dryrun,
        )

        # Install git if not present
        holohub_cli_util.run_command(
            ["apt-get", "install", "--no-install-recommends", "-y", "git"],
            dry_run=args.dryrun,
        )

        # Install unzip if not present
        holohub_cli_util.run_command(
            ["apt-get", "install", "--no-install-recommends", "-y", "unzip"],
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
                        "-O",
                        "ngccli_arm64.zip",
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
                        "-O",
                        "ngccli_linux.zip",
                    ],
                    dry_run=args.dryrun,
                )
                holohub_cli_util.run_command(["unzip", "ngccli_linux.zip"], dry_run=args.dryrun)
            holohub_cli_util.run_command(["chmod", "u+x", "ngc-cli/ngc"], dry_run=args.dryrun)
            holohub_cli_util.run_command(
                ["ln", "-s", f"{os.getcwd()}/ngc-cli/ngc", "/usr/local/bin/"],
                dry_run=args.dryrun,
            )

        # Install CUDA dependencies
        try:
            # Search for dpkg table entry for CUDA Runtime package
            # ii  cuda-cudart-12-3   12.3.101-1   amd64   CUDA Runtime native Libraries
            dpkg_output = subprocess.check_output(
                ["dpkg", "-l"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            cudart_entry = re.search(r"cuda-cudart-[0-9]+\-[0-9]+.*\n", dpkg_output)
            if cudart_entry:
                cuda_runtime_version = re.search(r"[0-9]+\.[0-9]+\.[0-9]+", cudart_entry.group(0))
                cuda_runtime_version = (
                    cuda_runtime_version.group(0) if cuda_runtime_version else None
                )
            else:
                cuda_runtime_version = None
        except subprocess.CalledProcessError:
            cuda_runtime_version = None

        if cuda_runtime_version:
            self._setup_cuda_packages(cuda_runtime_version.split(".")[0], args.dryrun)
        else:
            holohub_cli_util.info(
                "CUDA Runtime package not found, skipping CUDA package installation"
            )

        # Install the autocomplete
        holohub_cli_util.run_command(
            [
                "cp",
                f"{HoloHubCLI.HOLOHUB_ROOT}/utilities/holohub_autocomplete",
                "/etc/bash_completion.d/",
            ],
            dry_run=args.dryrun,
        )

        print(Color.green("Setup for HoloHub is ready. Happy Holocoding!"))

    def _setup_cuda_packages(self, cuda_major_version: str, dryrun: bool = False) -> None:
        """Find and install CUDA packages for Holoscan SDK development"""

        # Attempt to install cudnn9
        # cuDNN version example: libcudnn9-cuda-12/unknown,now 9.10.2.21-1 amd64
        CUDNN_9_PATTERN = r"9\.[0-9]+\.[0-9]+\.[0-9]+\-[0-9]+"
        try:
            installed_cudnn9_version = holohub_cli_util.install_cuda_dependencies_package(
                package_name="libcudnn9-cuda-12",
                version_pattern=CUDNN_9_PATTERN,
                dry_run=dryrun,
            )
            holohub_cli_util.install_cuda_dependencies_package(
                package_name="libcudnn9-dev-cuda-12",
                version_pattern=re.escape(installed_cudnn9_version),
                dry_run=dryrun,
            )
        except PackageInstallationError as e:
            holohub_cli_util.info(f"cuDNN 9.x installation failed, falling back to cuDNN 8.x: {e}")
            try:
                # Fall back to cudnn8
                # cuDNN version example: libcudnn8/unknown 8.9.7.29-1+cuda12.2 amd64
                CUDNN_8_PATTERN = (
                    rf"8\.[0-9]+\.[0-9]+\.[0-9]+\-[0-9]\+cuda{cuda_major_version}\.[0-9]+"
                )
                installed_cudnn8_version = holohub_cli_util.install_cuda_dependencies_package(
                    package_name="libcudnn8",
                    version_pattern=CUDNN_8_PATTERN,
                    dry_run=dryrun,
                )
                holohub_cli_util.install_cuda_dependencies_package(
                    package_name="libcudnn8-dev",
                    version_pattern=re.escape(installed_cudnn8_version),
                    dry_run=dryrun,
                )
            except PackageInstallationError as e:
                holohub_cli_util.info(f"cuDNN 8.x installation failed: {e}.")
                holohub_cli_util.info("cuDNN packages may need to be installed manually.")

        # Install TensorRT dependencies
        # TensorRT version example: libnvinfer-bin/unknown,now 10.12.0.36-1+cuda12.9 amd64
        NVINFER_PATTERN = rf"\d+\.[0-9]+\.[0-9]+\.[0-9]+-[0-9]\+cuda{cuda_major_version}\.[0-9]+"
        try:
            installed_libnvinferbin_version = holohub_cli_util.install_cuda_dependencies_package(
                package_name="libnvinfer-bin",
                version_pattern=NVINFER_PATTERN,
                dry_run=dryrun,
            )
            libnvinfer_pattern = re.escape(installed_libnvinferbin_version)

            for trt_package_name in [
                "libnvinfer-headers-dev",
                "libnvinfer-dev",
                "libnvinfer-plugin-dev",
                "libnvonnxparsers-dev",
            ]:
                holohub_cli_util.install_cuda_dependencies_package(
                    package_name=trt_package_name,
                    version_pattern=libnvinfer_pattern,
                    dry_run=dryrun,
                )
        except PackageInstallationError as e:
            holohub_cli_util.info(f"TensorRT installation failed: {e}")
            holohub_cli_util.info(
                "Continuing with setup - TensorRT packages may need to be installed manually"
            )

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
        if args.local or os.environ.get("HOLOHUB_BUILD_LOCAL"):
            # Build and install locally
            build_dir, project_data = self.build_project_locally(
                project_name=args.project,
                language=getattr(args, "language", None),
                build_type=args.build_type,
                with_operators=getattr(args, "with_operators", None),
                dryrun=args.dryrun,
                parallel=getattr(args, "parallel", None),
                configure_args=getattr(args, "configure_args", None),
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
                project_name=args.project, language=getattr(args, "language", None)
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
                )

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
                    install_cmd += f' --configure-args "{configure_arg}"'

            img = getattr(args, "img", None) or container.image_name
            docker_opts = getattr(args, "docker_opts", "")
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
            )
        else:
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

    def run(self) -> None:
        """Main entry point for the CLI"""

        trailing_docker_args = []  # Handle " -- " separator for run-container command forwarding
        cmd_args = sys.argv[1:]  # Skip script name, return a copy of the args
        if len(cmd_args) >= 2 and cmd_args[0] == "run-container" and "--" in cmd_args:
            sep = cmd_args.index("--")
            cmd_args, trailing_docker_args = cmd_args[:sep], cmd_args[sep + 1 :]

        try:
            args = self.parser.parse_args(cmd_args)
            if trailing_docker_args:
                args._trailing_args = trailing_docker_args  # " -- " used for run-container command
        except SystemExit as e:
            if len(cmd_args) > 0 and e.code != 0:  # exit code is 0 => help was successfully shown
                potential_command = cmd_args[0]
                if potential_command in self.subparsers:
                    # Show help for the specific subcommand
                    print(
                        f"\nError parsing arguments for '{potential_command}' command.\n",
                        file=sys.stderr,
                    )
                    self.subparsers[potential_command].print_help()
                    sys.exit(e.code if e.code is not None else 1)
                elif not potential_command.startswith("-"):
                    # Suggest similar commands using existing utility
                    suggestions = self._suggest_command(potential_command)
                    if suggestions:
                        print("\nDid you mean:", file=sys.stderr)
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
