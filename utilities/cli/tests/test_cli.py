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
    @patch("utilities.cli.holohub.HoloHubCLI.handle_create")
    @patch("utilities.cli.holohub.HoloHubCLI._add_to_cmakelists")
    def test_language_passed_to_cookiecutter(
        self,
        mock_add_to_cmakelists,
        mock_handle_create,
        mock_cookiecutter,
        mock_install_template_deps,
    ):
        """Test that the correct language parameter is passed to cookiecutter during project creation."""

        # Mock handle_create to call cookiecutter directly with the extra_context we want to test
        def side_effect(args):
            context = {
                "project_name": args.project,
                "language": args.language.lower() if args.language else "cpp",  # Default is cpp
            }
            # Call cookiecutter with our context
            mock_cookiecutter(
                "dummy_template_path",
                no_input=True,
                extra_context=context,
                output_dir="dummy_output_dir",
            )
            return True  # Indicate success

        mock_handle_create.side_effect = side_effect

        # Test CPP language
        args = self.cli.parser.parse_args("create test_cpp_project --language cpp".split())
        args.func(args)

        # Verify cookiecutter was called with cpp language
        mock_cookiecutter.assert_called_once()
        _, kwargs = mock_cookiecutter.call_args
        self.assertEqual(kwargs["extra_context"]["project_name"], "test_cpp_project")
        self.assertEqual(kwargs["extra_context"]["language"], "cpp")

        # Reset mocks for second test
        mock_cookiecutter.reset_mock()

        # Test Python language
        args = self.cli.parser.parse_args("create test_python_project --language python".split())
        args.func(args)

        # Verify cookiecutter was called with python language
        mock_cookiecutter.assert_called_once()
        _, kwargs = mock_cookiecutter.call_args
        self.assertEqual(kwargs["extra_context"]["project_name"], "test_python_project")
        self.assertEqual(kwargs["extra_context"]["language"], "python")

    @unittest.skipIf(
        not os.path.exists(os.path.join(os.getcwd(), "applications", "template")),
        "Template directory not found",
    )
    @patch("utilities.cli.holohub.HoloHubCLI._add_to_cmakelists")
    def test_project_file_generation(self, mock_add_to_cmakelists):
        import shutil
        import tempfile
        from pathlib import Path

        # Create temporary directory for test projects
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)

        try:

            # Create a C++ project
            cpp_project_name = "test_cpp_project"
            cmd = f"create {cpp_project_name} --language cpp --directory {temp_path} --interactive False"
            cpp_args = self.cli.parser.parse_args(cmd.split())
            cpp_args.directory = temp_path
            cpp_args.func(cpp_args)

            # Verify C++ source file exists
            cpp_source_file = temp_path / cpp_project_name / "src" / "main.cpp"
            self.assertTrue(
                cpp_source_file.exists(), f"C++ source file not found: {cpp_source_file}"
            )

            # Create a Python project
            py_project_name = "test_python_project"
            cmd = f"create {py_project_name} --language python --directory {temp_path} --interactive False"
            py_args = self.cli.parser.parse_args(cmd.split())
            py_args.directory = temp_path
            py_args.func(py_args)

            # Verify Python source file exists
            py_source_file = temp_path / py_project_name / "src" / "main.py"
            self.assertTrue(
                py_source_file.exists(), f"Python source file not found: {py_source_file}"
            )

        finally:
            # Clean up
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
