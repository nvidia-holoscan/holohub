#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add the utilities directory to the Python path
sys.path.append(str(Path(os.getcwd()) / "utilities"))

from utilities.cli.container import HoloHubContainer, _ContainerTerminationSignal
from utilities.cli.util import get_cli_arg_value, get_cuda_tag, get_default_cuda_version


class TestHoloHubContainer(unittest.TestCase):
    def setUp(self):
        # make a temporary directory for the test
        self.test_dir = tempfile.mkdtemp()
        self.test_dir_path = Path(self.test_dir)
        self.dockerfile_path = self.test_dir_path / "Dockerfile"

        # create a Dockerfile in the test directory
        self.dockerfile_path.write_text(
            "FROM nvcr.io/nvidia/clara-holoscan/holoscan:4.1.0-cuda12-dgpu"
        )

        # create project metadata
        self.project_metadata = {
            "project_name": "test_project",
            "source_folder": self.test_dir_path,
            "metadata": {"language": "cpp", "dockerfile": "<holohub_app_source>/Dockerfile"},
        }
        self.container = HoloHubContainer(project_metadata=self.project_metadata)

    def tearDown(self):
        """Clean up temporary directory after each test"""
        # Check if test_dir_path exists and remove it
        if hasattr(self, "test_dir_path") and self.test_dir_path.exists():
            try:
                shutil.rmtree(self.test_dir_path)
            except FileNotFoundError:
                # Directory was already removed, ignore
                pass
            except Exception as e:
                # Log but don't fail the test cleanup
                print(f"Warning: Failed to clean up {self.test_dir_path}: {e}")

    def test_default_base_image(self):
        """Test that default base image is correctly formatted"""
        base_image = HoloHubContainer.default_base_image()
        self.assertTrue(base_image.startswith("nvcr.io/nvidia/clara-holoscan/holoscan:"))

    def test_default_image(self):
        """Test that default image name is correctly formatted"""
        image_name = HoloHubContainer.default_image()
        self.assertTrue(image_name.startswith("holohub:ngc-"))

    def test_dockerfile_path(self):
        """Test that Dockerfile path is correctly determined"""
        dockerfile_path = self.container.dockerfile_path
        self.assertEqual(str(dockerfile_path), str(self.dockerfile_path))

    def test_image_name(self):
        """Test that image name is correctly determined"""
        image_name = self.container.image_name
        self.assertEqual(image_name, "holohub:test_project")

    @patch("subprocess.run")
    def test_build(self, mock_run):
        """Test container build command"""
        self.container.build()
        self.assertGreater(mock_run.call_count, 0)
        cmd = mock_run.call_args[0][0]
        self.assertTrue("docker" in cmd)
        self.assertTrue("build" in cmd)
        self.assertTrue(
            str(self.container.dockerfile_path) in " ".join(cmd),
            f"Dockerfile path {self.container.dockerfile_path} not found in command: {cmd}",
        )
        self.assertTrue(
            self.container.image_name in " ".join(cmd),
            f"Image name {self.container.image_name} not found in command: {cmd}",
        )

    @patch("utilities.cli.container.get_current_branch_slug", return_value="main-branch")
    @patch("utilities.cli.container.get_git_short_sha", return_value="abcdef123456")
    def test_image_names_contains_branch_sha_legacy(self, mock_sha, mock_branch):
        """Test that image_names returns branch, sha, and legacy tags in order"""
        names = self.container.image_names
        mock_sha.assert_called_once()
        mock_branch.assert_called_once()
        self.assertGreaterEqual(len(names), 3)
        self.assertEqual(names[0], "holohub-test_project:main-branch")
        self.assertEqual(names[1], "holohub-test_project:abcdef123456")
        self.assertEqual(names[2], "holohub:test_project")

    @patch("utilities.cli.container.get_current_branch_slug", return_value="dev-branch")
    @patch("utilities.cli.container.get_git_short_sha", return_value="feedfacebabe")
    @patch("subprocess.run")
    def test_build_applies_all_default_tags(self, mock_run, mock_sha, mock_branch):
        """When --img is omitted, build should tag with sha, branch, and legacy tags."""
        self.container.build()
        mock_sha.assert_called()
        mock_branch.assert_called()
        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(str(x) for x in cmd)
        self.assertIn("-t holohub-test_project:feedfacebabe", cmd_str)
        self.assertIn("-t holohub-test_project:dev-branch", cmd_str)
        self.assertIn("-t holohub:test_project", cmd_str)

    @patch("subprocess.run")
    def test_build_with_explicit_img_uses_only_that_tag(self, mock_run):
        """When --img is provided, only that tag should be applied."""
        self.container.build(img="custom:tag")
        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(str(x) for x in cmd)
        self.assertIn("-t custom:tag", cmd_str)
        self.assertNotIn("-t holohub-test_project:", cmd_str)
        self.assertNotIn("-t holohub:test_project", cmd_str)

    @patch("utilities.cli.container.check_nvidia_ctk")
    @patch("utilities.cli.container.get_image_pythonpath")
    @patch("subprocess.run")
    @patch("subprocess.check_output")
    def test_run(
        self, mock_check_output, mock_run, mock_get_image_pythonpath, mock_check_nvidia_ctk
    ):
        """Test container run command"""
        mock_check_nvidia_ctk.return_value = None
        mock_check_output.return_value = ""
        mock_get_image_pythonpath.return_value = ""
        self.container.run()
        self.assertGreater(mock_run.call_count, 0)  # xhost maybe called when env DISPLAY is set
        docker_run_call = None
        for call in mock_run.call_args_list:
            cmd = call[0][0]
            if isinstance(cmd, list) and len(cmd) > 0 and cmd[0] == "docker" and "run" in cmd:
                docker_run_call = cmd
                break
        self.assertIsNotNone(docker_run_call, "Docker run command not found in mock calls")
        self.assertTrue("docker" in docker_run_call)
        self.assertTrue("run" in docker_run_call)
        self.assertIn("--cidfile", docker_run_call)
        self.assertTrue("--runtime" in docker_run_call and "nvidia" in docker_run_call)
        self.assertIn(self.container.image_names[0], docker_run_call)

    @patch.dict(os.environ, {}, clear=True)
    @patch("utilities.cli.container.check_nvidia_ctk")
    @patch("utilities.cli.container.get_image_pythonpath")
    @patch("utilities.cli.container.os.kill")
    @patch("utilities.cli.container.signal.signal")
    @patch("utilities.cli.container.subprocess.run")
    @patch("utilities.cli.container.run_command")
    @patch("subprocess.check_output")
    def test_run_stops_cidfile_container_on_signal(
        self,
        mock_check_output,
        mock_run_command,
        mock_subprocess_run,
        _mock_signal,
        mock_kill,
        mock_get_image_pythonpath,
        mock_check_nvidia_ctk,
    ):
        """Signal cleanup should stop the exact container ID Docker wrote."""
        mock_check_nvidia_ctk.return_value = None
        mock_check_output.return_value = ""
        mock_get_image_pythonpath.return_value = ""

        def interrupt_after_cidfile(cmd, **_kwargs):
            if "--cidfile" not in cmd:
                return None
            cidfile = Path(cmd[cmd.index("--cidfile") + 1])
            cidfile.write_text("container123\n")
            raise _ContainerTerminationSignal(15)

        mock_run_command.side_effect = interrupt_after_cidfile

        with self.assertRaises(SystemExit) as raised:
            self.container.run()

        self.assertEqual(raised.exception.code, 128 + 15)
        mock_subprocess_run.assert_called_once()
        self.assertEqual(
            mock_subprocess_run.call_args[0][0],
            ["docker", "stop", "--time", "10", "container123"],
        )
        mock_kill.assert_called_once_with(os.getpid(), 15)

    @patch.dict(os.environ, {}, clear=True)
    @patch("utilities.cli.container.check_nvidia_ctk")
    @patch("utilities.cli.container.get_image_pythonpath")
    @patch("utilities.cli.container.os.kill")
    @patch("utilities.cli.container.signal.signal")
    @patch("utilities.cli.container.subprocess.run")
    @patch("utilities.cli.container.run_command")
    @patch("subprocess.check_output")
    def test_run_stops_user_cidfile_container_on_signal(
        self,
        mock_check_output,
        mock_run_command,
        mock_subprocess_run,
        _mock_signal,
        mock_kill,
        mock_get_image_pythonpath,
        mock_check_nvidia_ctk,
    ):
        """Signal cleanup should follow a caller-provided Docker --cidfile and not
        inject our own (Docker rejects duplicate --cidfile flags)."""
        mock_check_nvidia_ctk.return_value = None
        mock_check_output.return_value = ""
        mock_get_image_pythonpath.return_value = ""
        user_cidfile = self.test_dir_path / "user.cid"

        def interrupt_after_cidfile(cmd, **_kwargs):
            self.assertEqual(cmd.count("--cidfile"), 1)
            self.assertEqual(get_cli_arg_value(cmd, "--cidfile"), str(user_cidfile))
            user_cidfile.write_text("user-container\n")
            raise _ContainerTerminationSignal(15)

        mock_run_command.side_effect = interrupt_after_cidfile

        with self.assertRaises(SystemExit) as raised:
            self.container.run(docker_opts=f"--cidfile {user_cidfile}")

        self.assertEqual(raised.exception.code, 128 + 15)
        self.assertEqual(
            mock_subprocess_run.call_args[0][0],
            ["docker", "stop", "--time", "10", "user-container"],
        )
        self.assertTrue(user_cidfile.exists())
        mock_kill.assert_called_once_with(os.getpid(), 15)

    @patch.dict(os.environ, {}, clear=True)
    @patch("utilities.cli.container.check_nvidia_ctk")
    @patch("utilities.cli.container.get_image_pythonpath")
    @patch("utilities.cli.container.os.kill")
    @patch("utilities.cli.container.signal.signal")
    @patch("utilities.cli.container.subprocess.run")
    @patch("utilities.cli.container.run_command")
    @patch("subprocess.check_output")
    def test_run_warns_when_cidfile_empty_on_signal(
        self,
        mock_check_output,
        mock_run_command,
        mock_subprocess_run,
        _mock_signal,
        mock_kill,
        mock_get_image_pythonpath,
        mock_check_nvidia_ctk,
    ):
        """If Docker hasn't written the cidfile yet, no docker stop is issued."""
        mock_check_nvidia_ctk.return_value = None
        mock_check_output.return_value = ""
        mock_get_image_pythonpath.return_value = ""

        mock_run_command.side_effect = _ContainerTerminationSignal(2)

        with self.assertRaises(SystemExit) as raised:
            self.container.run()

        self.assertEqual(raised.exception.code, 128 + 2)
        mock_subprocess_run.assert_not_called()
        mock_kill.assert_called_once_with(os.getpid(), 2)

    @patch.dict(os.environ, {}, clear=True)
    def test_get_display_options_without_display_returns_empty(self):
        """No display environment should skip display forwarding."""
        self.assertEqual(self.container.get_display_options(True, False), [])

    @patch.dict(os.environ, {"DISPLAY": ":0"}, clear=True)
    def test_get_display_options_native_x11_uses_socket_and_xauth(self):
        """Native X11 should mount the UNIX socket and a generated Xauthority file."""

        def fake_run_command(cmd, **kwargs):
            if cmd[:2] == ["xauth", "nlist"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="0100 0000 0000\n")
            return subprocess.CompletedProcess(cmd, 0)

        with (
            patch("utilities.cli.container.Path.is_dir", return_value=True),
            patch("utilities.cli.container.shutil.which", return_value="/usr/bin/xauth"),
            patch("utilities.cli.container.run_command", side_effect=fake_run_command) as mock_run,
        ):
            options = self.container.get_display_options(True, False)

        self.assertIn("-v", options)
        self.assertIn("/tmp/.X11-unix:/tmp/.X11-unix:ro", options)
        self.assertIn("-e", options)
        self.assertIn("DISPLAY", options)
        self.assertTrue(any(opt.endswith(":ro") and ".docker.xauth-" in opt for opt in options))
        self.assertTrue(any(opt.startswith("XAUTHORITY=") for opt in options))
        self.assertEqual(mock_run.call_count, 2)
        self.assertTrue(mock_run.call_args_list[1].kwargs["input"].startswith("ffff"))

        xauth_mount = next(opt for opt in options if ".docker.xauth-" in opt)
        xauth_path = Path(xauth_mount.split(":", 1)[0])
        self.assertTrue(xauth_path.exists())
        self.container._cleanup_display_temp_files()
        self.assertFalse(xauth_path.exists())

    @patch.dict(os.environ, {"DISPLAY": "localhost:10.0"}, clear=True)
    def test_get_display_options_tcp_x11_skips_unix_socket(self):
        """SSH-forwarded X11 uses TCP and should not mount /tmp/.X11-unix."""
        with (
            patch("utilities.cli.container.Path.is_dir", return_value=True),
            patch("utilities.cli.container.shutil.which", return_value=None),
        ):
            options = self.container.get_display_options(True, True)

        self.assertIn("DISPLAY", options)
        self.assertNotIn("/tmp/.X11-unix:/tmp/.X11-unix:ro", options)
        self.assertFalse(any(opt.startswith("XAUTHORITY=") for opt in options))

    @patch.dict(
        os.environ,
        {
            "WAYLAND_DISPLAY": "wayland-0",
            "XDG_RUNTIME_DIR": "/run/user/1000",
            "XDG_SESSION_TYPE": "wayland",
        },
        clear=True,
    )
    def test_get_display_options_wayland_forwards_runtime_dir(self):
        """Wayland should forward XDG_RUNTIME_DIR (mounted) and WAYLAND_DISPLAY."""
        with patch("utilities.cli.container.Path.is_dir", return_value=True):
            options = self.container.get_display_options(True, False)

        self.assertIn("XDG_SESSION_TYPE", options)
        self.assertIn("WAYLAND_DISPLAY", options)
        self.assertIn("XDG_RUNTIME_DIR", options)
        self.assertIn("/run/user/1000:/run/user/1000", options)
        self.assertNotIn("DISPLAY", options)

    @patch("subprocess.CompletedProcess")
    def test_dry_run(self, mock_completed_process):
        """Test container dry run command"""
        self.container.dryrun = True
        self.container.run()
        cmd = mock_completed_process.call_args[0][0]
        self.assertIn(self.container.image_names[0], cmd)
        self.assertIn("--cidfile", cmd)
        self.assertIn("c 81:* rmw", cmd)
        self.container.dryrun = False

    def test_get_label_args_without_project(self):
        """A container with no project still gets the holohub.cli marker label."""
        container = HoloHubContainer(project_metadata=None)
        self.assertEqual(container.get_label_args(), ["--label", "holohub.cli=true"])

    def test_get_label_args_with_project(self):
        """A project-bound container labels the app for filtering by name."""
        self.assertEqual(
            self.container.get_label_args(),
            ["--label", "holohub.cli=true", "--label", "holohub.app=test_project"],
        )

    def test_get_label_args_with_mode(self):
        """The optional mode label is appended only when a mode name is supplied."""
        self.assertEqual(
            self.container.get_label_args(mode_name="standalone"),
            [
                "--label",
                "holohub.cli=true",
                "--label",
                "holohub.app=test_project",
                "--label",
                "holohub.mode=standalone",
            ],
        )

    @patch("subprocess.CompletedProcess")
    def test_run_emits_holohub_labels(self, mock_completed_process):
        """`docker run` should be stamped with holohub.cli/app/mode labels."""
        self.container.dryrun = True
        self.container.run(mode_name="standalone")
        cmd_str = mock_completed_process.call_args[0][0]
        self.assertIn("--label holohub.cli=true", cmd_str)
        self.assertIn("--label holohub.app=test_project", cmd_str)
        self.assertIn("--label holohub.mode=standalone", cmd_str)
        # Labels must precede the image positional — Docker stops parsing options at it.
        self.assertLess(
            cmd_str.rfind("--label "),
            cmd_str.rfind(self.container.image_names[0]),
        )
        self.container.dryrun = False

    @patch("utilities.cli.container.get_image_pythonpath")
    @patch.dict(os.environ, {}, clear=True)
    def test_get_pythonpath_options_sdk_only(self, mock_get_image_pythonpath):
        """Test PYTHONPATH with only SDK paths"""
        mock_get_image_pythonpath.return_value = ""
        result = self.container.get_pythonpath_options(local_sdk_root=None, img="test_img")
        expected_pythonpath = "/opt/nvidia/holoscan/python/lib:/workspace/holohub/benchmarks/holoscan_flow_benchmarking"
        self.assertEqual(result, ["-e", f"PYTHONPATH={expected_pythonpath}"])
        mock_get_image_pythonpath.assert_called_once_with("test_img", False)

    @patch("utilities.cli.container.get_image_pythonpath")
    @patch.dict(os.environ, {}, clear=True)
    def test_get_pythonpath_options_with_image_env(self, mock_get_image_pythonpath):
        """Test PYTHONPATH with SDK paths and Docker image environment"""
        mock_get_image_pythonpath.return_value = "/docker/lib1:/docker/lib2"
        result = self.container.get_pythonpath_options(local_sdk_root=None, img="test_img")
        expected_pythonpath = "/docker/lib1:/docker/lib2:/opt/nvidia/holoscan/python/lib:/workspace/holohub/benchmarks/holoscan_flow_benchmarking"
        self.assertEqual(result, ["-e", f"PYTHONPATH={expected_pythonpath}"])
        self.container.dryrun = False

    @patch("utilities.cli.container.find_hsdk_build_rel_dir", return_value="install-x86_64-dgpu")
    @patch("utilities.cli.container.get_image_pythonpath")
    @patch.dict(os.environ, {}, clear=True)
    def test_get_pythonpath_options_local_sdk_root_string(
        self, mock_get_image_pythonpath, mock_find_dir
    ):
        """Test PYTHONPATH with local_sdk_root passed as string (not Path)"""
        mock_get_image_pythonpath.return_value = ""
        result = self.container.get_pythonpath_options(
            local_sdk_root="/home/user/holoscan-sdk", img="test_img"
        )
        expected_pythonpath = (
            "/workspace/holoscan-sdk/install-x86_64-dgpu/python/lib"
            ":/workspace/holohub/benchmarks/holoscan_flow_benchmarking"
        )
        self.assertEqual(result, ["-e", f"PYTHONPATH={expected_pythonpath}"])

    @patch("utilities.cli.container.find_hsdk_build_rel_dir", return_value="install-x86_64-dgpu")
    @patch("utilities.cli.container.get_image_pythonpath")
    @patch.dict(os.environ, {}, clear=True)
    def test_get_pythonpath_options_local_sdk_precedes_image_env(
        self, mock_get_image_pythonpath, mock_find_dir
    ):
        """When local_sdk_root is set, local SDK paths must come before image paths"""
        mock_get_image_pythonpath.return_value = (
            "/opt/nvidia/holoscan/python/lib:/opt/nvidia/hololink/python"
        )
        result = self.container.get_pythonpath_options(
            local_sdk_root=Path("/home/user/holoscan-sdk"), img="test_img"
        )
        pythonpath = result[1].split("=", 1)[1]
        paths = pythonpath.split(":")
        local_sdk_path = "/workspace/holoscan-sdk/install-x86_64-dgpu/python/lib"
        image_path = "/opt/nvidia/holoscan/python/lib"
        self.assertIn(local_sdk_path, paths)
        self.assertIn(image_path, paths)
        self.assertLess(
            paths.index(local_sdk_path),
            paths.index(image_path),
            "Local SDK path must appear before image path in PYTHONPATH",
        )

    @patch("utilities.cli.util.get_host_gpu")
    def test_get_cuda_tag_sdk(self, mock_get_host_gpu):
        """Test CUDA tag with different SDK versions"""
        mock_get_host_gpu.return_value = "dgpu"

        # Test SDK 3.6.0 (old format) - returns gpu_type only
        self.assertEqual(get_cuda_tag("12", "3.6.0"), "dgpu")
        self.assertEqual(get_cuda_tag("13", "3.6.0"), "dgpu")

        # Test SDK > 3.6.1 (new format) - returns cuda{version}-{gpu_type}
        self.assertEqual(get_cuda_tag("12", "3.7.0"), "cuda12-dgpu")
        self.assertEqual(get_cuda_tag("13", "3.7.0"), "cuda13")
        self.assertEqual(get_cuda_tag("11", "3.7.0"), "cuda11-dgpu")

        # Test with iGPU
        mock_get_host_gpu.return_value = "igpu"
        self.assertEqual(get_cuda_tag("12", "3.7.0"), "cuda12-igpu")
        self.assertEqual(get_cuda_tag("12", "3.6.0"), "igpu")

    @patch.dict(os.environ, {"NGC_CLI_API_KEY": "token"}, clear=True)
    def test_get_ngc_options_defaults_org_when_missing(self):
        """Default NGC org when API key set and org missing."""
        options = self.container.get_ngc_options()
        self.assertListEqual(["-e", "NGC_CLI_API_KEY", "-e", "NGC_CLI_ORG=nvidia"], options)

    @patch.dict(
        os.environ,
        {"NGC_CLI_API_KEY": "token", "NGC_CLI_ORG": "custom-org"},
        clear=True,
    )
    def test_get_ngc_options_prefers_user_org(self):
        """Prefer user org when API key and org are set."""
        options = self.container.get_ngc_options()
        self.assertListEqual(["-e", "NGC_CLI_API_KEY", "-e", "NGC_CLI_ORG"], options)

    @patch.dict(os.environ, {}, clear=True)
    def test_get_ngc_options_no_default_without_api_key(self):
        """No default org when API key is missing."""
        options = self.container.get_ngc_options()
        self.assertListEqual([], options)

    @patch("utilities.cli.util.run_info_command")
    @patch("utilities.cli.util.shutil.which")
    def test_get_default_cuda_version(self, mock_which, mock_run_info_command):
        """Test default CUDA version detection based on NVIDIA driver version"""
        # nvidia-smi not available -> default to 13
        mock_which.return_value = None
        self.assertEqual(get_default_cuda_version(), "13")
        # nvidia-smi available but driver version unknown -> default to 13
        mock_which.return_value = "/usr/bin/nvidia-smi"
        mock_run_info_command.return_value = None
        self.assertEqual(get_default_cuda_version(), "13")
        # Driver version < 580 -> CUDA 12
        mock_run_info_command.return_value = "550.54.14"
        self.assertEqual(get_default_cuda_version(), "12")
        # Driver version >= 580 -> CUDA 13
        mock_run_info_command.return_value = "580.1"
        self.assertEqual(get_default_cuda_version(), "13")
        # Unparsable driver version -> default to 13
        mock_run_info_command.return_value = "not.a.version"
        self.assertEqual(get_default_cuda_version(), "13")


if __name__ == "__main__":
    unittest.main()
