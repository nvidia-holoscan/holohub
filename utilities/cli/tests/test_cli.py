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
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the utilities directory to the Python path
sys.path.append(str(Path(os.getcwd()) / "utilities"))

from utilities.cli.holohub import HoloHubCLI


class TestHoloHubCLI(unittest.TestCase):
    def setUp(self):
        self.cli = HoloHubCLI()
        # Mock project data for testing
        self.mock_project_data = {
            "project_name": "test_project",
            "source_folder": Path(os.getcwd()) / "applications" / "test_project",
            "metadata": {
                "language": "cpp",
                "dockerfile": "<holohub_app_source>/Dockerfile",
                "run": {"command": "./test_project"},
            },
        }
        # Mock project data with some similar names
        self.cli.projects = [
            {"project_name": "hello_world", "metadata": {"language": "cpp"}},
            {"project_name": "hello_world_python", "metadata": {"language": "python"}},
            {"project_name": "hello_world_cpp", "source_folder": "applications/hello_world_cpp"},
            {
                "project_name": "hello_world_advanced",
                "metadata": {"language": "cpp"},
                "source_folder": "applications/hello_world_advanced",
            },
        ]

    def test_parser_creation(self):
        """Test that the argument parser is created with all required commands"""
        parser = self.cli.parser
        self.assertIsNotNone(parser)

        # Check that all main commands are present
        subparsers = [action for action in parser._actions if action.dest == "command"]
        self.assertEqual(len(subparsers), 1)
        # Check for some key commands
        cmds = "build-container run-container build run list lint setup install"
        for cmd in cmds.split():
            self.assertIn(cmd, subparsers[0].choices)

    @patch("utilities.cli.holohub.HoloHubCLI._find_project")
    @patch("utilities.cli.holohub.HoloHubContainer")
    def test_build_container_command(self, mock_container_class, mock_find_project):
        """Test the build-container command parsing"""
        mock_find_project.return_value = self.mock_project_data
        mock_container = MagicMock()
        mock_container_class.return_value = mock_container

        cmd = "build-container test_project --base_img test_image --verbose --no-cache"
        args = self.cli.parser.parse_args(cmd.split())
        self.assertEqual(args.command, "build-container")
        self.assertEqual(args.project, "test_project")
        self.assertEqual(args.base_img, "test_image")
        self.assertTrue(args.verbose)
        self.assertTrue(args.no_cache)

        # Call the function to verify it's called with correct args
        args.func(args)
        mock_find_project.assert_called_once_with(project_name="test_project", language=None)
        mock_container_class.assert_called_once()
        mock_container.build.assert_called_once_with(
            docker_file=None, base_img="test_image", img=None, no_cache=True, build_args=None
        )

    @patch("utilities.cli.holohub.HoloHubCLI._find_project")
    @patch("utilities.cli.holohub.HoloHubContainer")
    @patch("utilities.cli.util.run_command")
    @patch("shutil.which")
    def test_run_command(
        self, mock_which, mock_run_command, mock_container_class, mock_find_project
    ):
        """Test the run command parsing"""
        mock_find_project.return_value = self.mock_project_data
        mock_container = MagicMock()
        mock_container_class.return_value = mock_container
        mock_run_command.return_value = None
        mock_which.return_value = "/usr/local/bin/nsys"  # Mock nsys being available

        cmd = "run test_project --local --verbose --nsys-profile"
        args = self.cli.parser.parse_args(cmd.split())
        self.assertEqual(args.command, "run")
        self.assertEqual(args.project, "test_project")
        self.assertTrue(args.local)
        self.assertTrue(args.verbose)
        self.assertTrue(args.nsys_profile)

        # Call the function to verify it's called with correct args
        args.func(args)
        mock_find_project.assert_called_once_with(project_name="test_project", language=None)

    def test_list_command(self):
        """Test the list command parsing"""
        args = self.cli.parser.parse_args(["list"])
        self.assertEqual(args.command, "list")
        self.assertEqual(len(vars(args)), 2)  # command and func

    def test_did_you_mean_suggestion(self):
        """Test the 'did you mean' suggestion functionality"""

        # Test with a misspelled project name
        with self.assertRaises(SystemExit):
            with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
                self.cli._find_project("hello_wrld")
                stderr_output = mock_stderr.getvalue()
                self.assertIn("Project 'hello_wrld' not found.", stderr_output)
                self.assertIn(
                    "Did you mean: 'hello_world' (source: applications/hello_world)", stderr_output
                )

        # Test with a completely different name
        with self.assertRaises(SystemExit):
            with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
                self.cli._find_project("nonexistent")
                stderr_output = mock_stderr.getvalue()
                self.assertIn("Project 'nonexistent' not found.", stderr_output)
                self.assertIn(
                    "Did you mean: 'hello_world' (source: applications/hello_world)", stderr_output
                )

    @patch("utilities.cli.util.run_command")
    def test_lint_command(self, mock_run_command):
        """Test the lint command parsing"""
        mock_run_command.return_value = None

        args = self.cli.parser.parse_args("lint test/path --fix --install-dependencies".split())
        self.assertEqual(args.command, "lint")
        self.assertEqual(args.path, "test/path")
        self.assertTrue(args.fix)
        self.assertTrue(args.install_dependencies)

        # Call the function to verify it's called with correct args
        args.func(args)
        # Verify that run_command was called at least once
        mock_run_command.assert_called()


if __name__ == "__main__":
    unittest.main()
