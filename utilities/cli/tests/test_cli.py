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
    @patch("utilities.cli.util.run_command")
    @patch("shutil.which")
    @patch("pathlib.Path.is_dir")
    @patch("os.chdir")
    def test_container_commands(
        self,
        mock_chdir,
        mock_is_dir,
        mock_which,
        mock_run_command,
        mock_container_class,
        mock_find_project,
    ):
        """Test both build-container and run commands"""
        # Setup mocks
        mock_find_project.return_value = self.mock_project_data
        mock_container = MagicMock()
        mock_container_class.return_value = mock_container
        mock_is_dir.return_value = True

        # Test build-container command
        build_args = self.cli.parser.parse_args(
            "build-container test_project --base_img test_image --no-cache".split()
        )
        build_args.func(build_args)
        mock_container.build.assert_called_with(
            docker_file=None, base_img="test_image", img=None, no_cache=True, build_args=None
        )

        # Test run command
        run_args = self.cli.parser.parse_args("run test_project --local".split())
        run_args.func(run_args)
        mock_find_project.assert_called_with(project_name="test_project", language=None)

    def test_list_command(self):
        """Test the list command parsing"""
        args = self.cli.parser.parse_args(["list"])
        self.assertEqual(args.command, "list")
        self.assertEqual(len(vars(args)), 2)  # command and func

    def test_did_you_mean_suggestion(self):
        """Test the 'did you mean' suggestion functionality"""
        # Test with a misspelled project name
        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            with self.assertRaises(SystemExit):
                self.cli._find_project("hello_wrld")
                stderr_output = mock_stderr.getvalue()
                self.assertIn("Project 'hello_wrld' not found.", stderr_output)
                self.assertIn(
                    "Did you mean: 'hello_world' (source: applications/hello_world)", stderr_output
                )

        # Test with a completely different name
        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            with self.assertRaises(SystemExit):
                self.cli._find_project("nonexistent")
                stderr_output = mock_stderr.getvalue()
                self.assertIn("Project 'nonexistent' not found.", stderr_output)
                self.assertNotIn("Did you mean", stderr_output)

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

    def test_lint_fix_command(self):
        args = self.cli.parser.parse_args("lint --fix".split())
        try:
            args.func(args)
        except SystemExit as e:
            self.assertEqual(e.code, 0)

    @patch("utilities.cli.holohub.HoloHubCLI._install_template_deps")
    @patch("cookiecutter.main.cookiecutter")
    @patch("pathlib.Path.mkdir")
    @patch("shutil.copy2")
    @patch("utilities.cli.holohub.Path")
    def test_handle_create(
        self, mock_path, mock_copy2, mock_mkdir, mock_cookiecutter, mock_install_template_deps
    ):
        """Test the create command functionality"""
        args = self.cli.parser.parse_args("create new_app --language python".split())
        args.func(args)

        mock_cookiecutter.assert_called_once()
        _, kwargs = mock_cookiecutter.call_args
        self.assertEqual(kwargs["extra_context"]["project_name"], "new_app")
        self.assertEqual(kwargs["extra_context"]["language"], "python")

        mock_mkdir.assert_called_once()
        mock_copy2.assert_called_once()
        self.assertTrue(mock_copy2.call_args[0][0].endswith("example.py"))
        self.assertTrue(mock_copy2.call_args[0][1].endswith("new_app.py"))

    @patch("utilities.cli.holohub.HoloHubCLI._install_template_deps")
    @patch("cookiecutter.main.cookiecutter")
    @patch("utilities.cli.holohub.Path")
    def test_handle_create_dryrun(self, mock_path, mock_cookiecutter, mock_install_template_deps):
        """Test the create command with dryrun flag"""
        args = self.cli.parser.parse_args("create new_app --dryrun".split())
        args.func(args)
        mock_cookiecutter.assert_not_called()
        mock_install_template_deps.assert_not_called()

    @patch("utilities.cli.holohub.HoloHubCLI._install_template_deps")
    @patch("cookiecutter.main.cookiecutter")
    @patch("pathlib.Path.mkdir")
    @patch("shutil.copy2")
    @patch("utilities.cli.holohub.Path")
    def test_handle_create_with_context(
        self, mock_path, mock_copy2, mock_mkdir, mock_cookiecutter, mock_install_template_deps
    ):
        """Test the create command with additional context variables"""
        args = self.cli.parser.parse_args(
            "create new_app --context version=1.0.0 --context description=Test-project".split()
        )
        args.func(args)
        mock_cookiecutter.assert_called_once()
        _, kwargs = mock_cookiecutter.call_args
        self.assertEqual(kwargs["extra_context"]["version"], "1.0.0")
        self.assertEqual(kwargs["extra_context"]["description"], "Test-project")

        # Test with invalid context variable format
        with self.assertRaises(SystemExit):
            args = self.cli.parser.parse_args("create new_app --context invalid_format".split())
            args.func(args)
        # Test with multiple context variables
        args = self.cli.parser.parse_args(
            "create new_app --context key1=value1 --context key2=value2".split()
        )
        args.func(args)
        mock_cookiecutter.assert_called()
        _, kwargs = mock_cookiecutter.call_args
        self.assertEqual(kwargs["extra_context"]["key1"], "value1")
        self.assertEqual(kwargs["extra_context"]["key2"], "value2")


if __name__ == "__main__":
    unittest.main()
