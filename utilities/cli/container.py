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

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from .util import (
    check_nvidia_ctk,
    fatal,
    get_compute_capacity,
    get_group_id,
    get_host_gpu,
    normalize_language,
    run_command,
)

base_sdk_version = "3.2.0"


class HoloHubContainer:
    """
    Describes the container environment for a HoloHub project.

    This class is responsible for common container operations and environment configuration,
    which may differ across different projects.

    Default attributes may be overridden by a project-specific implementation.
    """

    CONTAINER_PREFIX = "holohub"
    HOLOHUB_ROOT = Path(__file__).parent.parent.parent
    DOCKER_CONFIG_PATH = Path(__file__).parent / "docker_config.json"

    @classmethod
    def load_docker_config(cls) -> dict:
        """Load docker configuration from JSON file"""
        try:
            with open(cls.DOCKER_CONFIG_PATH, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Docker config file not found at {cls.DOCKER_CONFIG_PATH}")
            return {"modes": {}}
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in docker config file: {e}")
            return {"modes": {}}

    @classmethod
    def get_config_args(cls, mode: str, section: str = "run") -> List[str]:
        """Get docker arguments from config for a specific mode and section"""
        import shlex

        config_mode = cls.load_docker_config().get("modes", {}).get(mode, {})
        args = config_mode.get(section, {}).get("docker_args", [])
        result = []
        for arg in args:
            if arg.startswith("-") and " " in arg:
                # Use shlex for proper quote handling and argument splitting
                result.extend(shlex.split(arg))
            else:
                result.append(arg)
        return result

    @classmethod
    def get_conditional_args(cls, condition: str) -> List[str]:
        """Get conditional docker arguments for a specific condition"""
        config = cls.load_docker_config()
        conditional_args = (
            config.get("modes", {}).get("conditional", {}).get("run", {}).get("docker_args", {})
        )
        return conditional_args.get(condition, [])

    @classmethod
    def default_base_image(cls) -> str:
        return f"nvcr.io/nvidia/clara-holoscan/holoscan:v{base_sdk_version}-{get_host_gpu()}"

    @classmethod
    def default_image(cls) -> str:
        return f"{cls.CONTAINER_PREFIX}:ngc-v{base_sdk_version}-{get_host_gpu()}"

    @staticmethod
    def default_dockerfile() -> Path:
        return HoloHubContainer.HOLOHUB_ROOT / "Dockerfile"

    @staticmethod
    def device_args() -> List[str]:
        """Get docker run arguments for mounting devices in the container"""
        options = []

        # Add device mounts
        device_paths = [
            "/usr/lib/libvideomasterhd.so",
            "/opt/deltacast/videomaster/Include",
            "/opt/yuan/qcap/include",
            "/opt/yuan/qcap/lib",
            "/usr/lib/aarch64-linux-gnu/tegra",
        ]

        for path in device_paths:
            if os.path.exists(path):
                if os.path.isfile(path):
                    options.append(f"--volume={path}:{path}")
                else:
                    options.append(f"--volume={path}:{path}")
        return options

    @staticmethod
    def group_args() -> List[str]:
        """Get docker run arguments for adding groups to the container"""
        options = []
        for group in ["video", "render", "docker"]:
            gid = get_group_id(group)
            if gid is not None:
                options.append(f"--group-add={gid}")
        return options

    def get_conditional_options(
        self, use_tini: bool = False, persistent: bool = False
    ) -> List[str]:
        """Get conditional docker options"""
        options = []

        # Handle tini and persistence
        if use_tini:
            options.extend(self.get_conditional_args("tini"))
        if not persistent:
            options.extend(self.get_conditional_args("non_persistent"))

        return options

    @property
    def image_name(self) -> str:
        if self.dockerfile_path != HoloHubContainer.default_dockerfile():
            return f"{self.CONTAINER_PREFIX}:{self.project_metadata.get('project_name', '')}"
        return HoloHubContainer.default_image()

    @property
    def dockerfile_path(self) -> Path:
        """
        Get Dockerfile path for the project according to the search strategy:
        1. As specified in metadata.json
        2. <app_source>/Dockerfile
        3. <app_source>/<language>/Dockerfile
        4. holohub/Dockerfile
        """
        if not self.project_metadata:
            return HoloHubContainer.default_dockerfile()

        if self.project_metadata.get("metadata", {}).get("dockerfile"):
            dockerfile = self.project_metadata["metadata"]["dockerfile"]
            dockerfile = dockerfile.replace(
                "<holohub_app_source>", str(self.project_metadata["source_folder"])
            )
            dockerfile = dockerfile.replace("<holohub_root>", str(HoloHubContainer.HOLOHUB_ROOT))

            # If the Dockerfile path is not absolute, make it absolute
            if not str(dockerfile).startswith(str(HoloHubContainer.HOLOHUB_ROOT)):
                dockerfile = str(HoloHubContainer.HOLOHUB_ROOT / dockerfile)

            return Path(dockerfile)

        source_folder = self.project_metadata.get("source_folder", "")
        if source_folder:
            dockerfile_path = self.project_metadata.get("source_folder", "") / "Dockerfile"
            if (source_folder / "Dockerfile").exists():
                return dockerfile_path

            language = normalize_language(
                self.project_metadata.get("metadata", {}).get("language", "")
            )
            if (source_folder / language / "Dockerfile").exists():
                return source_folder / language / "Dockerfile"

        return HoloHubContainer.HOLOHUB_ROOT / "Dockerfile"

    def __init__(self, project_metadata: Optional[dict[str, any]]):
        if not project_metadata:
            print("No project provided, proceeding with default container")
        elif not isinstance(project_metadata, dict):
            print("No project provided, proceeding with default container")

        # Environment defaults
        self.holoscan_py_exe = os.environ.get("HOLOSCAN_PY_EXE", "python3")
        self.holoscan_docker_exe = os.environ.get("HOLOSCAN_DOCKER_EXE", "docker")

        self.project_metadata = project_metadata

        self.holoscan_sdk_version = "sdk-latest"
        self.holohub_container_base_name = "holohub"

        self.dryrun = False

    def build(
        self,
        docker_file: Optional[str] = None,
        base_img: Optional[str] = None,
        img: Optional[str] = None,
        no_cache: bool = False,
        build_args: Optional[str] = None,
    ) -> None:
        """Build the container image"""

        # Get Dockerfile path
        docker_file_path = docker_file or self.dockerfile_path
        base_img = base_img or self.default_base_image()
        img = img or self.image_name
        gpu_type = get_host_gpu()
        compute_capacity = get_compute_capacity()

        # Check if buildx exists
        if not self.dryrun:
            try:
                run_command(["docker", "buildx", "version"], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                fatal(
                    "docker buildx plugin is missing. Please install docker-buildx-plugin:\n"
                    "https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository"
                )

        # Set DOCKER_BUILDKIT environment variable
        os.environ["DOCKER_BUILDKIT"] = "1"

        cmd = ["docker", "build"]

        # Add static build args from config
        cmd.extend(self.get_config_args("default", "build"))

        # Add dynamic build args
        cmd.extend(
            [
                "--build-arg",
                f"BASE_IMAGE={base_img}",
                "--build-arg",
                f"GPU_TYPE={gpu_type}",
                "--build-arg",
                f"COMPUTE_CAPACITY={compute_capacity}",
            ]
        )

        if no_cache:
            cmd.append("--no-cache")

        if build_args:
            cmd.extend(build_args.split())

        cmd.extend(["-f", str(docker_file_path), "-t", img, str(HoloHubContainer.HOLOHUB_ROOT)])

        run_command(cmd, dry_run=self.dryrun)

    def run(
        self,
        img: Optional[str] = None,
        local_sdk_root: Optional[Path] = None,
        enable_x11: bool = True,
        use_tini: bool = False,
        persistent: bool = False,
        nsys_profile: bool = False,
        nsys_location: str = "",
        as_root: bool = False,
        docker_opts: str = "",
        docker_args: List[str] = None,
        add_volumes: List[str] = None,
        verbose: bool = False,
        extra_args: List[str] = None,
    ) -> None:
        """Launch the container"""

        if not self.dryrun:
            check_nvidia_ctk()

        img = img or self.image_name
        add_volumes = add_volumes or []
        extra_args = extra_args or []
        docker_args = docker_args or []

        # Build docker command
        cmd = [self.holoscan_docker_exe, "run"]

        # Add static run args from config
        cmd.extend(self.get_config_args("default"))

        # Add TTY if available
        if sys.stdout.isatty():
            cmd.extend(self.get_conditional_args("tty_interactive"))

        # User permissions
        if not as_root:
            cmd.extend(["-u", f"{os.getuid()}:{os.getgid()}"])

        # Workspace mounting
        cmd.extend(
            [
                "-v",
                f"{HoloHubContainer.HOLOHUB_ROOT}:/workspace/holohub",
                "-w",
                "/workspace/holohub",
            ]
        )

        # Additional volumes
        for volume in add_volumes:
            base = os.path.basename(volume)
            cmd.extend(["-v", f"{volume}:/workspace/volumes/{base}"])

        # Add conditional options
        cmd.extend(self.get_conditional_options(use_tini, persistent))

        # Add UCX options
        cmd.extend(self.get_config_args("ucx"))

        # Add display server options
        cmd.extend(self.get_display_options(enable_x11))

        # Add nsys options
        cmd.extend(self.get_nsys_options(nsys_profile, nsys_location))

        # Add local SDK options if provided
        if local_sdk_root:
            cmd.extend(self.get_local_sdk_options(local_sdk_root))
            cmd.extend(["-e", "PYTHONPATH=/workspace/holoscan-sdk/build/python/lib"])

        # Add docker options if provided
        if docker_opts:
            cmd.extend(docker_opts.split())

        # Add docker arguments list if provided (preferred method)
        if docker_args:
            cmd.extend(docker_args)

        # Add the image name
        cmd.append(img)

        # Add any extra arguments
        cmd.extend(extra_args)

        if verbose:
            cmd_list = [f'"{arg}"' if " " in str(arg) else str(arg) for arg in cmd]
            print(f"Launch command: {' '.join(cmd_list)}")

        run_command(cmd, dry_run=self.dryrun)

    def get_display_options(self, enable_x11: bool) -> List[str]:
        """Get display-related options"""
        options = []
        if "XDG_SESSION_TYPE" in os.environ:
            options.extend(["-e", "XDG_SESSION_TYPE"])
            if os.environ["XDG_SESSION_TYPE"] == "wayland":
                options.extend(["-e", "WAYLAND_DISPLAY"])

        if "XDG_RUNTIME_DIR" in os.environ:
            options.extend(["-e", "XDG_RUNTIME_DIR"])
            if os.path.isdir(os.environ["XDG_RUNTIME_DIR"]):
                options.extend(
                    ["-v", f"{os.environ['XDG_RUNTIME_DIR']}:{os.environ['XDG_RUNTIME_DIR']}"]
                )

        if enable_x11:
            options.extend(self.get_config_args("display"))
        return options

    def get_nsys_options(self, nsys_profile: bool, nsys_location: str) -> List[str]:
        """Get nsys-related options"""
        options = []
        if nsys_profile:
            options.extend(["--nvtx", "--capture-range", "0"])

        if nsys_location:
            options.extend(["--output", nsys_location])

        return options

    def get_local_sdk_options(self, local_sdk_root: Path) -> List[str]:
        """Get Holoscan SDK-related options"""
        return [
            "-v",
            f"{local_sdk_root}:/workspace/holoscan-sdk",
            "-e",
            "HOLOSCAN_LIB_PATH=/workspace/holoscan-sdk/build/lib",
            "-e",
            "HOLOSCAN_SAMPLE_DATA_PATH=/workspace/holoscan-sdk/data",
            "-e",
            "HOLOSCAN_TESTS_DATA_PATH=/workspace/holoscan-sdk/tests/data",
        ]
