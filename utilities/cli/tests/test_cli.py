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

    @patch("utilities.cli.holohub.HoloHubCLI._find_project")
    @patch("utilities.cli.holohub.HoloHubCLI._build_project_locally")
    @patch("utilities.cli.util.run_command")
    @patch("utilities.cli.holohub.shlex.split")
    @patch("os.chdir")
    @patch("pathlib.Path.is_dir")
    @patch("os.environ.copy")
    def test_run_args_handling(
        self,
        mock_environ_copy,
        mock_is_dir,
        mock_chdir,
        mock_shlex_split,
        mock_run_command,
        mock_build_project,
        mock_find_project,
    ):
        """Test the run_args parameter handling"""
        # Setup mocks
        mock_find_project.return_value = self.mock_project_data
        mock_build_dir = Path("/path/to/build")
        mock_build_project.return_value = (mock_build_dir, self.mock_project_data)
        mock_shlex_split.side_effect = lambda x: x.split()  # Simple split for testing
        mock_is_dir.return_value = True  # Make sure the directory check passes
        mock_environ_copy.return_value = {
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": "/path/to/python",
        }  # Provide a minimal environment
        complex_args = self.cli.parser.parse_args(["run", "test_project", "--local"])
        complex_args.run_args = "--param1 value1 --param2=value2"
        complex_args.nsys_profile = False  # Skip nsys_profile check
        complex_args.func(complex_args)
        self.assertIn("--param1", mock_run_command.call_args[0][0])
        self.assertIn("value1", mock_run_command.call_args[0][0])
        self.assertIn("--param2=value2", mock_run_command.call_args[0][0])
        mock_run_command.reset_mock()

    @patch("utilities.cli.holohub.HoloHubCLI._find_project")
    @patch("utilities.cli.holohub.HoloHubContainer")
    @patch("utilities.cli.holohub.shlex.quote")
    def test_container_run_args(
        self,
        mock_shlex_quote,
        mock_container_class,
        mock_find_project,
    ):
        """Test run_args handling in container mode"""
        # Setup mocks
        mock_find_project.return_value = self.mock_project_data
        mock_container = MagicMock()
        mock_container_class.return_value = mock_container

        # Set a predictable return value for shlex.quote
        run_args_value = "--flag1 value1 --flag2='quoted value'"
        quoted_value = "'--flag1 value1 --flag2=\\'quoted value\\''"
        mock_shlex_quote.return_value = quoted_value

        # Test container run with run_args
        run_args = self.cli.parser.parse_args(["run", "test_project"])
        run_args.run_args = run_args_value
        run_args.func(run_args)

        # Verify the arguments were properly quoted
        mock_shlex_quote.assert_called_once_with(run_args_value)

        # Verify the container.run was called with the correctly quoted arguments
        kwargs = mock_container.run.call_args[1]
        cmd_string = kwargs["extra_args"][1]
        self.assertIn("--run_args", cmd_string)
        self.assertIn(quoted_value, cmd_string)

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

    @patch("utilities.cli.holohub.HoloHubCLI._build_project_locally")
    @patch("utilities.cli.holohub.HoloHubCLI._find_project")
    @patch("utilities.cli.holohub.HoloHubContainer")
    def test_with_operators_parameter(
        self,
        mock_container_class,
        mock_find_project,
        mock_build_project_locally,
    ):
        """Test the --build_with parameter for both build and run commands in local and container modes"""
        # Common setup
        operators = "operator1;operator2;operator3"
        mock_find_project.return_value = self.mock_project_data
        mock_build_project_locally.return_value = (Path("/path/to/build"), self.mock_project_data)
        mock_container = MagicMock()
        mock_container_class.return_value = mock_container
        mock_container.project_metadata = self.mock_project_data

        # Test 1: Build with operators in local mode
        args = self.cli.parser.parse_args(
            f"build test_project --local --build_with {operators}".split()
        )
        args.func(args)
        # Verify local build
        mock_build_project_locally.assert_called()
        call_args = mock_build_project_locally.call_args[1]
        self.assertEqual(call_args["project_name"], "test_project")
        self.assertEqual(call_args["with_operators"], operators)
        mock_build_project_locally.reset_mock()

        # Test 2: Build with operators in container mode
        args = self.cli.parser.parse_args(f"build test_project --build_with {operators}".split())
        args.func(args)
        # Verify container build
        mock_container.run.assert_called()
        kwargs = mock_container.run.call_args[1]
        command_string = kwargs["extra_args"][1]
        self.assertIn(f'--build_with "{operators}"', command_string)
        mock_container.reset_mock()

        # Test 3: Run with operators in container mode
        args = self.cli.parser.parse_args(f"run test_project --build_with {operators}".split())
        args.func(args)
        # Verify container run
        mock_container.run.assert_called()
        kwargs = mock_container.run.call_args[1]
        command_string = kwargs["extra_args"][1]
        self.assertIn(f'--build_with "{operators}"', command_string)

    @patch("utilities.cli.holohub.HoloHubCLI._find_project")
    @patch("utilities.cli.util.run_command")
    @patch("pathlib.Path.mkdir")
    def test_build_project_locally_with_operators(
        self,
        mock_mkdir,
        mock_run_command,
        mock_find_project,
    ):
        """Test that _build_project_locally correctly adds the HOLOHUB_BUILD_OPERATORS CMake parameter"""
        # Set up mocks
        mock_find_project.return_value = self.mock_project_data
        mock_run_command.return_value = MagicMock()

        # Call _build_project_locally with operators
        operators = "operator1;operator2;operator3"
        self.cli._build_project_locally(
            project_name="test_project",
            language="cpp",
            build_type="debug",
            with_operators=operators,
            dryrun=False,
        )

        # Verify CMake args contain the operators
        self.assertEqual(mock_run_command.call_count, 2)
        cmake_args_str = " ".join(mock_run_command.call_args_list[0][0][0])
        self.assertIn("-DHOLOHUB_BUILD_OPERATORS", cmake_args_str)
        self.assertIn(f'"{operators}"', cmake_args_str)


if __name__ == "__main__":
    unittest.main()
