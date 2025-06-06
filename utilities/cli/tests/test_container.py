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
    @patch("subprocess.run")
    @patch("subprocess.check_output")
    def test_run(self, mock_check_output, mock_run, mock_check_nvidia_ctk):
        """Test container run command"""
        mock_check_nvidia_ctk.return_value = None
        mock_check_output.return_value = ""
        self.container.run()
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertTrue("docker" in cmd)
        self.assertTrue("run" in cmd)
        self.assertTrue("--runtime=nvidia" in cmd)
        self.assertTrue(self.container.image_name in cmd)

    @patch("subprocess.CompletedProcess")
    def test_dry_run(self, mock_completed_process):
        """Test container dry run command"""
        self.container.dryrun = True
        self.container.run()
        cmd = mock_completed_process.call_args[0][0]
        self.assertTrue(self.container.image_name in cmd)
        self.assertIn("c 81:* rmw", cmd)
        self.container.dryrun = False


if __name__ == "__main__":
    unittest.main()
