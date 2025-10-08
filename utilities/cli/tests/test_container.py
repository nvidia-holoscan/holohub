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

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Add the utilities directory to the Python path
sys.path.append(str(Path(os.getcwd()) / "utilities"))

from utilities.cli.container import HoloHubContainer
from utilities.cli.util import get_cuda_tag, get_default_cuda_version


class TestHoloHubContainer(unittest.TestCase):
    def setUp(self):
        self.project_metadata = {
            "project_name": "test_project",
            "source_folder": Path("/test/path"),
            "metadata": {"language": "cpp", "dockerfile": "<holohub_app_source>/Dockerfile"},
        }
        self.container = HoloHubContainer(project_metadata=self.project_metadata)

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
        self.assertEqual(str(dockerfile_path), "/test/path/Dockerfile")

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
        self.assertTrue("--runtime" in docker_run_call and "nvidia" in docker_run_call)
        self.assertTrue(self.container.image_name in docker_run_call)

    @patch("subprocess.CompletedProcess")
    def test_dry_run(self, mock_completed_process):
        """Test container dry run command"""
        self.container.dryrun = True
        self.container.run()
        cmd = mock_completed_process.call_args[0][0]
        self.assertTrue(self.container.image_name in cmd)
        self.assertIn("c 81:* rmw", cmd)
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
