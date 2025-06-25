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
import glob
import os
import shlex
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from .util import (
    build_holohub_path_mapping,
    check_nvidia_ctk,
    docker_args_to_devcontainer_format,
    fatal,
    get_compute_capacity,
    get_group_id,
    get_host_gpu,
    get_image_pythonpath,
    normalize_language,
    replace_placeholders,
    run_command,
)

base_sdk_version = "3.3.0"


class HoloHubContainer:
    """
    Describes the container environment for a HoloHub project.

    This class is responsible for common container operations and environment configuration,
    which may differ across different projects.

    Default attributes may be overridden by a project-specific implementation.
    """

    CONTAINER_PREFIX = "holohub"
    HOLOHUB_ROOT = Path(__file__).parent.parent.parent

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
    def get_build_argparse() -> argparse.ArgumentParser:
        """Get argument parser for container build options"""
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--base-img", help="(Build container) Fully qualified base image name")
        parser.add_argument("--docker-file", help="(Build container) Path to Dockerfile to use")
        parser.add_argument(
            "--img", help="(Build container) Specify fully qualified container name"
        )
        parser.add_argument(
            "--no-cache",
            action="store_true",
            help="(Build container) Do not use cache when building the image",
        )
        parser.add_argument(
            "--build-args",
            help="(Build container) Extra arguments to docker build command, "
            "example: `--build-args '--network=host --build-arg \"CUSTOM=value with spaces\"'`",
        )
        return parser

    @staticmethod
    def get_run_argparse() -> argparse.ArgumentParser:
        """Get argument parser for container run options"""
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--docker-opts",
            default="",
            help="Additional options to the Docker run command, "
            "example: `--docker-opts='--entrypoint=bash'` or `--docker-opts '-e DISPLAY=:1'`",
        )
        parser.add_argument(
            "--ssh-x11",
            action="store_true",
            help="Enable X11 forwarding of graphical HoloHub applications over SSH",
        )
        parser.add_argument(
            "--nsys-profile",
            action="store_true",
            help="Support Nsight Systems profiling in container",
        )
        parser.add_argument(
            "--local-sdk-root",
            help="Path to Holoscan SDK used for building local Holoscan SDK container",
        )
        parser.add_argument("--init", action="store_true", help="Support tini entry point")
        parser.add_argument(
            "--persistent", action="store_true", help="Does not delete container after it is run"
        )
        parser.add_argument(
            "--add-volume",
            action="append",
            help="Mount additional volume to `/workspace/volumes`, example: `--add-volume /tmp`",
        )
        parser.add_argument(
            "--as-root", action="store_true", help="Run the container with root permissions"
        )
        parser.add_argument(
            "--nsys-location",
            help="Specify location of the Nsight Systems installation on the host "
            "(e.g., /opt/nvidia/nsight-systems/2024.1.1/)",
        )
        parser.add_argument(
            "--mps",
            action="store_true",
            help="If CUDA MPS is enabled on the host, mount MPS host directories into the container",
        )
        parser.add_argument(
            "--enable-x11",
            action="store_true",
            default=True,
            help="Enable X11 forwarding (default: True)",
        )
        return parser

    @staticmethod
    def ucx_args() -> List[str]:
        """UCX-related docker run arguments"""
        return [
            "--ipc=host",
            "--cap-add=CAP_SYS_PTRACE",
            "--ulimit=memlock=-1",
            "--ulimit=stack=67108864",
        ]

    @staticmethod
    def get_device_mounts() -> List[str]:
        """Get docker run arguments for mounting specialized hardware devices and libraries"""
        options = []

        for video_dev in glob.glob("/dev/video[0-9]*"):
            options.extend(["--device", video_dev])

        for capture_dev in glob.glob("/dev/capture-vi-channel[0-9]*"):
            options.extend(["--device", capture_dev])

        for video_dev in glob.glob("/dev/ajantv2[0-9]*"):
            options.extend(["--device", f"{video_dev}:{video_dev}"])

        # Deltacast capture boards and Videomaster SDK
        for i in range(4):
            # Deltacast SDI capture board
            delta_sdi = f"/dev/delta-x380{i}"
            if os.path.exists(delta_sdi):
                options.extend(["--device", f"{delta_sdi}:{delta_sdi}"])

            # Deltacast HDMI capture board
            delta_hdmi = f"/dev/delta-x350{i}"
            if os.path.exists(delta_hdmi):
                options.extend(["--device", f"{delta_hdmi}:{delta_hdmi}"])

        # Find and mount all audio devices
        if os.path.isdir("/dev/snd"):
            # Only mount specific audio device patterns, exclude directories
            audio_patterns = [
                "/dev/snd/control*",
                "/dev/snd/pcm*",
                "/dev/snd/timer",
                "/dev/snd/seq",
                "/dev/snd/midi*",
            ]
            for pattern in audio_patterns:
                for audio_dev in glob.glob(pattern):
                    try:
                        # Check if it's a character device using stat module
                        if stat.S_ISCHR(os.stat(audio_dev).st_mode):
                            options.extend(["--device", audio_dev])
                    except OSError:
                        continue

        # Mount ALSA configuration
        if os.path.exists("/etc/asound.conf"):
            options.extend(
                ["--mount", "source=/etc/asound.conf,target=/etc/asound.conf,readonly,type=bind"]
            )

        # Mount ConnectX device nodes
        if os.path.exists("/dev/infiniband/rdma_cm"):
            options.extend(["--device", "/dev/infiniband/rdma_cm"])

        for uverbs_dev in glob.glob("/dev/infiniband/uverbs[0-9]*"):
            options.extend(["--device", uverbs_dev])

        conditional_mounts = [
            "/usr/local/cmake/VideoMasterHDConfigVersion.cmake",
            "/usr/local/cmake/VideoMasterHDConfig.cmake",
            "/usr/lib/libvideomasterhd.so",
            "/usr/lib/libvideomasterhd_audio.so",
            "/usr/lib/libvideomasterhd_vbi.so",
            "/usr/lib/libvideomasterhd_vbidata.so",
            "/usr/include/videomaster",
            "/opt/yuan/qcap/include",
            "/opt/yuan/qcap/lib",
            "/usr/lib/aarch64-linux-gnu/tegra",
        ]

        for path in conditional_mounts:
            if os.path.exists(path):
                options.extend(["-v", f"{path}:{path}"])

        if os.path.exists("/dev/nvgpu/igpu0/nvsched"):
            options.extend(["--device", "/dev/nvgpu/igpu0/nvsched"])
        if os.path.exists("/dev/nvhost-ctrl-nvdec"):
            options.extend(["--device", "/dev/nvhost-ctrl-nvdec"])
        if os.path.exists("/dev/nvhost-ctxsw-gpu"):
            options.extend(["--device", "/dev/nvhost-ctxsw-gpu"])
        if os.path.exists("/dev/nvhost-nvsched-gpu"):
            options.extend(["--device", "/dev/nvhost-nvsched-gpu"])
        if os.path.exists("/dev/nvhost-sched-gpu"):
            options.extend(["--device", "/dev/nvhost-sched-gpu"])
        if os.path.exists("/dev/nvidia0"):
            options.extend(["--device", "/dev/nvidia0"])
        if os.path.exists("/dev/nvidia-modeset"):
            options.extend(["--device", "/dev/nvidia-modeset"])
        if os.path.exists("/usr/share/nvidia/nvoptix.bin"):
            options.extend(["-v", "/usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro"])
        return options

    @staticmethod
    def group_args() -> List[str]:
        """Get docker run arguments for adding groups to the container"""
        options = []
        for group in ["video", "render", "docker", "audio"]:
            gid = get_group_id(group)
            if gid is None:
                continue
            options.extend(["--group-add", str(gid)])
        return options

    def get_conditional_options(
        self, use_tini: bool = False, persistent: bool = False
    ) -> List[str]:
        options = []
        if use_tini:
            options.append("--init")
        if not persistent:
            options.append("--rm")
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
            # Build path mapping for this project
            path_mapping = build_holohub_path_mapping(
                holohub_root=HoloHubContainer.HOLOHUB_ROOT,
                project_data=self.project_metadata,
            )

            dockerfile = replace_placeholders(
                self.project_metadata["metadata"]["dockerfile"], path_mapping
            )

            # If the Dockerfile path is not absolute, make it absolute
            if not str(dockerfile).startswith(str(HoloHubContainer.HOLOHUB_ROOT)):
                dockerfile = str(HoloHubContainer.HOLOHUB_ROOT / dockerfile)

            return Path(dockerfile)

        source_folder = self.project_metadata.get("source_folder", "")
        if source_folder:
            dockerfile_path = source_folder / "Dockerfile"
            if dockerfile_path.exists():
                return dockerfile_path

            language = normalize_language(
                self.project_metadata.get("metadata", {}).get("language", "")
            )
            dockerfile_path = source_folder / language / "Dockerfile"
            if dockerfile_path.exists():
                return dockerfile_path

        return HoloHubContainer.default_dockerfile()

    def __init__(self, project_metadata: Optional[dict[str, any]]):
        if not project_metadata:
            print("No project provided, proceeding with default container")
        elif not isinstance(project_metadata, dict):
            print("No project provided, proceeding with default container")

        # Environment defaults
        self.holoscan_py_exe = os.environ.get("HOLOSCAN_PY_EXE", "python3")
        self.holoscan_docker_exe = os.environ.get("HOLOSCAN_DOCKER_EXE", "docker")
        self.holoscan_sdk_version = os.environ.get("HOLOSCAN_SDK_VERSION", "sdk-latest")
        self.holohub_container_base_name = os.environ.get("HOLOHUB_CONTAINER_BASE_NAME", "holohub")

        self.project_metadata = project_metadata

        self.dryrun = False
        self.verbose = False

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

        cmd = [
            "docker",
            "build",
            "--build-arg",
            "BUILDKIT_INLINE_CACHE=1",
            "--build-arg",
            f"BASE_IMAGE={base_img}",
            "--build-arg",
            f"GPU_TYPE={gpu_type}",
            "--build-arg",
            f"COMPUTE_CAPACITY={compute_capacity}",
            "--network=host",
        ]

        if no_cache:
            cmd.append("--no-cache")

        if build_args:
            cmd.extend(shlex.split(build_args))

        cmd.extend(["-f", str(docker_file_path), "-t", img, str(HoloHubContainer.HOLOHUB_ROOT)])

        run_command(cmd, dry_run=self.dryrun)

    def run(
        self,
        img: Optional[str] = None,
        local_sdk_root: Optional[Path] = None,
        enable_x11: bool = True,
        ssh_x11: bool = False,
        use_tini: bool = False,
        persistent: bool = False,
        nsys_profile: bool = False,
        nsys_location: str = "",
        as_root: bool = False,
        docker_opts: str = "",
        add_volumes: List[str] = None,
        enable_mps: bool = False,
        extra_args: List[str] = None,
    ) -> None:
        """Launch the container"""

        if not self.dryrun:
            check_nvidia_ctk()

        img = img or self.image_name
        add_volumes = add_volumes or []
        extra_args = extra_args or []

        cmd = [self.holoscan_docker_exe, "run"]

        cmd.extend(self.get_basic_args())
        cmd.extend(self.get_security_args(as_root))
        cmd.extend(self.get_volume_args(add_volumes, enable_mps))
        cmd.extend(self.get_gpu_runtime_args())
        cmd.extend(self.get_environment_args())

        cmd.extend(self.get_conditional_options(use_tini, persistent))
        cmd.extend(self.ucx_args())
        cmd.extend(self.get_device_mounts())
        cmd.extend(self.group_args())
        cmd.extend(self.get_display_options(enable_x11, ssh_x11))
        cmd.extend(self.get_nsys_options(nsys_profile, nsys_location))
        cmd.extend(self.get_pythonpath_options(local_sdk_root, img))

        if local_sdk_root:
            cmd.extend(self.get_local_sdk_options(local_sdk_root))

        if docker_opts:
            cmd.extend(shlex.split(docker_opts))

        cmd.append(img)
        cmd.extend(extra_args)

        if self.verbose:
            cmd_list = [f'"{arg}"' if " " in str(arg) else str(arg) for arg in cmd]
            print(f"Launch command: {' '.join(cmd_list)}")

        run_command(cmd, dry_run=self.dryrun)

    def get_basic_args(self) -> List[str]:
        """Basic container runtime arguments"""
        args = ["--net", "host", "--interactive"]
        if sys.stdout.isatty():
            args.append("--tty")
        return args

    def get_security_args(self, as_root: bool) -> List[str]:
        """User and security arguments"""
        args = []

        if not as_root:
            args.extend(["-u", f"{os.getuid()}:{os.getgid()}"])

        args.extend(["-v", "/etc/group:/etc/group:ro", "-v", "/etc/passwd:/etc/passwd:ro"])

        return args

    def get_volume_args(self, add_volumes: List[str], enable_mps: bool) -> List[str]:
        """Volume mounting arguments"""
        args = []

        args.extend(
            [
                "-v",
                f"{HoloHubContainer.HOLOHUB_ROOT}:/workspace/holohub",
                "-w",
                "/workspace/holohub",
            ]
        )

        for volume in add_volumes:
            base = os.path.basename(volume)
            args.extend(["-v", f"{volume}:/workspace/volumes/{base}"])

        if enable_mps:
            if os.path.isdir("/tmp/nvidia-mps") and os.path.isdir("/tmp/nvidia-log"):
                args.extend(
                    [
                        "-v",
                        "/tmp/nvidia-mps:/tmp/nvidia-mps",
                        "-v",
                        "/tmp/nvidia-log:/tmp/nvidia-log",
                    ]
                )
            else:
                print("Warning: MPS directories not found. MPS may not be enabled on the host.")

        return args

    def get_nvidia_runtime_args(self) -> List[str]:
        return [
            "--runtime",
            "nvidia",
            "--gpus",
            "all",
        ]

    def get_device_cgroup_args(self) -> List[str]:
        return [
            "--device-cgroup-rule",
            "c 81:* rmw",  # /dev/video*
            "--device-cgroup-rule",
            "c 189:* rmw",  # /dev/bus/usb/*
        ]

    def get_gpu_runtime_args(self) -> List[str]:
        args = []
        args.extend(self.get_nvidia_runtime_args())
        args.extend(
            [
                "--cap-add",
                "CAP_SYS_PTRACE",
                "--ipc=host",
                "-v",
                "/dev:/dev",
            ]
        )
        args.extend(self.get_device_cgroup_args())
        return args

    def get_environment_args(self) -> List[str]:
        """Environment variable arguments"""
        return [
            "-e",
            "NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display",
            "-e",
            "HOME=/workspace/holohub",
            "-e",
            "CUPY_CACHE_DIR=/workspace/holohub/.cupy/kernel_cache",
            "-e",
            "HOLOHUB_BUILD_LOCAL=1",
        ]

    def enable_x11_access(self) -> None:
        if (
            "DISPLAY" in os.environ
            and shutil.which("xhost")
            and os.environ.get("XDG_SESSION_TYPE", "x11") in ["x11", "tty", ""]
        ):
            run_command(["xhost", "+local:docker"], check=False, dry_run=self.dryrun)

    def get_display_options(self, enable_x11: bool, ssh_x11: bool) -> List[str]:
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

        # Handle X11 forwarding
        if enable_x11 or ssh_x11:
            # Enable X11 access for Docker containers
            self.enable_x11_access()
            options.extend(["-v", "/tmp/.X11-unix:/tmp/.X11-unix", "-e", "DISPLAY"])

        # Handle SSH X11 forwarding
        if ssh_x11:
            xauth_file = "/tmp/.docker.xauth"
            # xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
            # chmod 777 $XAUTH
            options.extend(["-v", f"{xauth_file}:{xauth_file}", "-e", f"XAUTHORITY={xauth_file}"])

        return options

    def get_nsys_options(self, nsys_profile: bool, nsys_location: str) -> List[str]:
        """Get nsys-related options"""
        options = []
        if nsys_profile:
            options.extend(["--cap-add=SYS_ADMIN"])
        if nsys_location:
            options.extend(["-v", f"{nsys_location}:/opt/nvidia/nsys-host"])
        return options

    def get_pythonpath_options(
        self, local_sdk_root: Optional[Path], img: Optional[str] = None
    ) -> List[str]:
        """Get PYTHONPATH configuration"""
        benchmarking_path = "/workspace/holohub/benchmarks/holoscan_flow_benchmarking"
        if local_sdk_root:
            sdk_paths = f"/workspace/holoscan-sdk/build/python/lib:{benchmarking_path}"
        else:
            sdk_paths = f"/opt/nvidia/holoscan/python/lib:{benchmarking_path}"
        all_paths = []
        if img:
            image_pythonpath = get_image_pythonpath(img, self.dryrun)
            if image_pythonpath:
                all_paths.extend([p for p in image_pythonpath.split(":") if p])
        all_paths.extend([p for p in sdk_paths.split(":") if p and p not in all_paths])
        pythonpath = ":".join(all_paths)
        return ["-e", f"PYTHONPATH={pythonpath}"]

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

    def get_devcontainer_args(self, docker_opts: str = "") -> str:
        """Get all devcontainer-formatted arguments as JSON array string"""
        docker_args = []
        docker_args.extend(self.get_device_mounts())
        docker_args.extend(self.group_args())
        docker_args.extend(self.ucx_args())
        docker_args.extend(self.get_device_cgroup_args())
        docker_args.extend(self.get_nvidia_runtime_args())
        if docker_opts:
            docker_args.extend(shlex.split(docker_opts))
        project_name = self.project_metadata.get("project_name") if self.project_metadata else None
        hostname = f"holohub-{project_name}" if project_name else "holohub"
        docker_args.extend(["--hostname", hostname])

        devcontainer_options = docker_args_to_devcontainer_format(docker_args)
        return ",\n        ".join(f'"{opt}"' for opt in devcontainer_options)
