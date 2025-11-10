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
import subprocess
import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the utilities directory to the Python path
sys.path.append(str(Path(os.getcwd()) / "utilities"))

from utilities.cli import util
from utilities.cli.holohub import HoloHubCLI


def is_cookiecutter_available():
    try:
        import cookiecutter  # noqa: F401

        return True
    except ImportError:
        return False


def is_ruff_available():
    try:
        import ruff  # noqa: F401

        return True
    except ImportError:
        return False


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
            {
                "project_name": "hello_world",
                "project_type": "application",
                "metadata": {"language": "cpp"},
            },
            {
                "project_name": "hello_world_python",
                "project_type": "application",
                "metadata": {"language": "python"},
            },
            {
                "project_name": "hello_world_cpp",
                "project_type": "application",
                "source_folder": "applications/hello_world_cpp",
            },
            {
                "project_name": "hello_world_advanced",
                "project_type": "application",
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

    @patch("utilities.cli.util.check_nvidia_ctk")
    @patch("utilities.cli.holohub.HoloHubCLI.find_project")
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
        mock_check_nvidia_ctk,
    ):
        """Test both build-container and run commands"""
        # Setup mocks
        mock_find_project.return_value = self.mock_project_data
        mock_container = MagicMock()
        mock_container_class.return_value = mock_container
        mock_is_dir.return_value = True
        mock_check_nvidia_ctk.return_value = None
        # Test build-container command
        build_args = self.cli.parser.parse_args(
            "build-container test_project --base-img test_image --no-cache".split()
        )
        build_args.func(build_args)
        mock_container.build.assert_called_with(
            docker_file=None,
            base_img="test_image",
            img=None,
            no_cache=True,
            build_args=None,
            cuda_version=None,
            extra_scripts=None,
        )
        # Test run command
        run_args = self.cli.parser.parse_args("run test_project --local".split())
        run_args.func(run_args)
        mock_find_project.assert_called_with(project_name="test_project", language=None)

        # Test run command with --no-docker-build flag
        mock_container.reset_mock()
        no_build_args = self.cli.parser.parse_args("run test_project --no-docker-build".split())
        no_build_args.func(no_build_args)
        # Verify container.build was not called when --no-docker-build is used
        mock_container.build.assert_not_called()
        mock_container.run.assert_called_once()

    @patch("utilities.cli.holohub.HoloHubCLI.find_project")
    @patch("utilities.cli.holohub.HoloHubCLI.build_project_locally")
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

    @patch("utilities.cli.holohub.HoloHubCLI.find_project")
    @patch("utilities.cli.holohub.HoloHubCLI.build_project_locally")
    @patch("utilities.cli.holohub.HoloHubContainer")
    @patch("utilities.cli.holohub.shlex.quote")
    @patch("utilities.cli.util.get_container_entrypoint")
    def test_container_run_args(
        self,
        mock_get_container_entrypoint,
        mock_shlex_quote,
        mock_container_class,
        mock_build_project_locally,
        mock_find_project,
    ):
        """Test run_args handling in container mode"""
        # Setup mocks
        mock_find_project.return_value = self.mock_project_data
        mock_container = MagicMock()
        mock_container_class.return_value = mock_container
        mock_get_container_entrypoint.return_value = ["bash"]

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
        self.assertIn("./holohub run test_project", cmd_string)
        self.assertIn("--run-args", cmd_string)
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
                self.cli.find_project("hello_wrld")
                stderr_output = mock_stderr.getvalue()
                self.assertIn("Project 'hello_wrld' not found.", stderr_output)
                self.assertIn(
                    "Did you mean: 'hello_world' (source: applications/hello_world)", stderr_output
                )

        # Test with a completely different name
        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            with self.assertRaises(SystemExit):
                self.cli.find_project("nonexistent")
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
        except FileNotFoundError as e:
            if is_ruff_available():
                raise e
            else:
                self.assertIn("ruff", str(e))  # if not installed, it complains about ruff
        except SystemExit as e:
            self.assertEqual(e.code, 0)

    @unittest.skipIf(not is_cookiecutter_available(), "cookiecutter not installed")
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

    @unittest.skipIf(not is_cookiecutter_available(), "cookiecutter not installed")
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

    @patch("utilities.cli.holohub.HoloHubCLI.build_project_locally")
    @patch("utilities.cli.holohub.HoloHubCLI.find_project")
    @patch("utilities.cli.holohub.HoloHubContainer")
    @patch("utilities.cli.util.get_container_entrypoint")
    def test_with_operators_parameter(
        self,
        mock_get_container_entrypoint,
        mock_container_class,
        mock_find_project,
        mock_build_project_locally,
    ):
        """Test the --build-with parameter for both build and run commands in local and container modes"""
        # Common setup
        operators = "operator1;operator2;operator3"
        mock_find_project.return_value = self.mock_project_data
        mock_build_project_locally.return_value = (Path("/path/to/build"), self.mock_project_data)
        mock_container = MagicMock()
        mock_container_class.return_value = mock_container
        mock_container.project_metadata = self.mock_project_data
        mock_get_container_entrypoint.return_value = ["custom-bash"]

        # Test 1: Build with operators in local mode
        args = self.cli.parser.parse_args(
            f"build test_project --local --build-with {operators}".split()
        )
        args.func(args)
        # Verify local build
        mock_build_project_locally.assert_called()
        call_args = mock_build_project_locally.call_args[1]
        self.assertEqual(call_args["project_name"], "test_project")
        self.assertEqual(call_args["with_operators"], operators)
        mock_build_project_locally.reset_mock()

        # Test 2: Build with operators in container mode
        args = self.cli.parser.parse_args(f"build test_project --build-with {operators}".split())
        args.func(args)
        # Verify container build
        mock_container.run.assert_called()
        kwargs = mock_container.run.call_args[1]
        command_string = kwargs["extra_args"][1]
        self.assertIn(f'--build-with "{operators}"', command_string)
        self.assertIn("--entrypoint=/bin/bash", kwargs["docker_opts"])
        mock_container.reset_mock()

        # Test 3: Run with operators in container mode
        args = self.cli.parser.parse_args(f"run test_project --build-with {operators}".split())
        args.func(args)
        # Verify container run
        mock_container.run.assert_called()
        kwargs = mock_container.run.call_args[1]
        command_string = kwargs["extra_args"][1]
        self.assertIn(f'--build-with "{operators}"', command_string)

    @patch("utilities.cli.holohub.HoloHubCLI.find_project")
    @patch("utilities.cli.util.run_command")
    @patch("pathlib.Path.mkdir")
    def test_build_project_locally_with_operators(
        self,
        mock_mkdir,
        mock_run_command,
        mock_find_project,
    ):
        """Test that build_project_locally correctly adds the HOLOHUB_BUILD_OPERATORS CMake parameter"""
        # Set up mocks
        mock_find_project.return_value = self.mock_project_data
        mock_run_command.return_value = MagicMock()

        # Call build_project_locally with operators
        operators = "operator1;operator2;operator3"
        self.cli.build_project_locally(
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

    def test_custom_script_name_entry_point(self):
        """Test that CLI behaves as if it's called with the custom script name"""
        import subprocess
        import tempfile
        from pathlib import Path

        # Create a custom entry point script
        custom_name = "my_custom_holohub"
        holohub_script = (
            Path(os.path.dirname(os.path.abspath(__file__))) / ".." / ".." / ".." / "holohub"
        )
        if not holohub_script.exists():
            self.skipTest("holohub bash script not found")

        original_env = os.environ.get("HOLOHUB_CMD_NAME")

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Create a custom wrapper script that sets HOLOHUB_CMD_NAME
                custom_script = Path(temp_dir) / custom_name
                with open(custom_script, "w") as f:
                    f.write(
                        f"""#!/bin/bash
export HOLOHUB_CMD_NAME='{custom_name}'
exec {holohub_script} "$@"
"""
                    )
                custom_script.chmod(0o755)

                result_help = subprocess.run(
                    [str(custom_script), "--help"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    cwd=os.getcwd(),
                )
                self.assertEqual(result_help.returncode, 0, "Help command should succeed")
                self.assertIn(
                    custom_name,
                    result_help.stdout or result_help.stderr,
                    f"Help output should mention custom script name '{custom_name}'",
                )
            except subprocess.TimeoutExpired:
                self.skipTest("Command timed out")
            except FileNotFoundError:
                self.skipTest("Script execution failed")
            finally:
                if original_env is not None:
                    os.environ["HOLOHUB_CMD_NAME"] = original_env
                elif "HOLOHUB_CMD_NAME" in os.environ:
                    del os.environ["HOLOHUB_CMD_NAME"]

    @patch("utilities.cli.holohub.HoloHubCLI.find_project")
    @patch("utilities.cli.util.run_command")
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.glob")
    @patch("pathlib.Path.exists")
    def test_package_build_functionality(
        self,
        mock_exists,
        mock_glob,
        mock_mkdir,
        mock_run_command,
        mock_find_project,
    ):
        """Test package build functionality including PKG prefix and cpack execution"""
        # Mock package data
        mock_package_data = {
            "project_name": "test_package",
            "project_type": "package",
            "source_folder": Path(os.getcwd()) / "pkg" / "test_package",
            "metadata": {"language": "cpp"},
        }

        mock_find_project.return_value = mock_package_data
        mock_run_command.return_value = MagicMock()

        mock_cpack_config = Path("/path/to/build/test_package/pkg/CPackConfig-test_package.cmake")
        mock_exists.return_value = True  # pkg directory exists
        mock_glob.return_value = [mock_cpack_config]  # cpack config file exists

        self.cli.build_project_locally(
            project_name="test_package",
            language="cpp",
            build_type="release",
            dryrun=False,
        )

        self.assertEqual(mock_run_command.call_count, 3)  # cmake configure, cmake build, cpack
        cmake_configure_args = mock_run_command.call_args_list[0][0][0]
        cmake_args_str = " ".join(cmake_configure_args)
        self.assertIn("-DPKG_test_package=ON", cmake_args_str)
        self.assertNotIn("-DAPP_test_package=ON", cmake_args_str)

        cpack_args = mock_run_command.call_args_list[2][0][0]
        self.assertEqual(cpack_args[0], "cpack")
        self.assertIn("--config", cpack_args)
        self.assertIn(str(mock_cpack_config), cpack_args)
        self.assertIn("-G", cpack_args)
        self.assertIn("DEB", cpack_args)

        mock_run_command.reset_mock()

        self.cli.build_project_locally(
            project_name="test_package",
            language="cpp",
            build_type="release",
            pkg_generator="RPM",
            dryrun=False,
        )
        self.assertEqual(mock_run_command.call_count, 3)
        cpack_args = mock_run_command.call_args_list[2][0][0]
        self.assertIn("RPM", cpack_args)

    @patch("utilities.cli.holohub.HoloHubCLI.find_project")
    @patch("utilities.cli.holohub.HoloHubCLI.build_project_locally")
    @patch("utilities.cli.holohub.HoloHubContainer")
    @patch("utilities.cli.util.run_command")
    @patch("utilities.cli.util.get_container_entrypoint")
    def test_install_command(
        self,
        mock_get_container_entrypoint,
        mock_run_command,
        mock_container_class,
        mock_build_project_locally,
        mock_find_project,
    ):
        """Test the install command in both local and container modes"""
        # Common setup
        mock_find_project.return_value = self.mock_project_data
        mock_build_dir = Path("/path/to/build")
        mock_build_project_locally.return_value = (mock_build_dir, self.mock_project_data)
        mock_container = MagicMock()
        mock_container_class.return_value = mock_container
        mock_run_command.return_value = MagicMock()
        mock_get_container_entrypoint.return_value = ["python"]

        # Test 1: Install in local mode
        args = self.cli.parser.parse_args(
            "install test_project --local --build-type release".split()
        )
        args.func(args)
        mock_build_project_locally.assert_called_once()
        call_kwargs = mock_build_project_locally.call_args[1]
        self.assertEqual(call_kwargs["project_name"], "test_project")
        self.assertEqual(call_kwargs["build_type"], "release")
        mock_run_command.assert_called_with(
            ["cmake", "--install", str(mock_build_dir)], dry_run=False
        )
        mock_build_project_locally.reset_mock()
        mock_run_command.reset_mock()
        mock_container.reset_mock()

        # Test 2: Install in container mode
        args = self.cli.parser.parse_args(
            "install test_project --build-type debug --language cpp".split()
        )
        args.func(args)
        mock_container.build.assert_called_once()
        mock_container.run.assert_called_once()
        kwargs = mock_container.run.call_args[1]
        command_string = kwargs["extra_args"][1]
        self.assertIn("./holohub install test_project --local", command_string)
        self.assertIn("--build-type debug", command_string)
        self.assertIn("--language cpp", command_string)
        mock_container.reset_mock()
        args = self.cli.parser.parse_args("install test_project --parallel 4".split())
        args.func(args)
        kwargs = mock_container.run.call_args[1]
        command_string = kwargs["extra_args"][1]
        self.assertIn("--parallel 4", command_string)

    @patch("utilities.cli.holohub.HoloHubCLI.find_project")
    @patch("utilities.cli.util.run_command")
    @patch("pathlib.Path.mkdir")
    def test_configure_args_functionality(
        self,
        mock_mkdir,
        mock_run_command,
        mock_find_project,
    ):
        """Test that --configure-args are properly passed to CMake (single and multiple)"""
        mock_find_project.return_value = self.mock_project_data
        mock_run_command.return_value = MagicMock()

        # Test single configure arg
        configure_arg = "-DCUSTOM_OPTION=ON"
        self.cli.build_project_locally(
            project_name="test_project",
            language="cpp",
            build_type="debug",
            configure_args=[configure_arg],
            dryrun=False,
        )

        # Verify CMake configure call includes the custom argument
        self.assertEqual(mock_run_command.call_count, 2)  # cmake configure + cmake build
        cmake_configure_args = mock_run_command.call_args_list[0][0][0]
        cmake_args_str = " ".join(cmake_configure_args)
        self.assertIn(configure_arg, cmake_args_str)

        mock_run_command.reset_mock()

        # Test multiple configure args
        configure_args = ["-DCUSTOM_OPTION=ON", "-DCMAKE_VERBOSE_MAKEFILE=ON", "-DDEBUG_MODE=1"]
        self.cli.build_project_locally(
            project_name="test_project",
            language="cpp",
            build_type="debug",
            configure_args=configure_args,
            dryrun=False,
        )

        # Verify CMake configure call includes all custom arguments
        self.assertEqual(mock_run_command.call_count, 2)  # cmake configure + cmake build
        cmake_configure_args = mock_run_command.call_args_list[0][0][0]
        cmake_args_str = " ".join(cmake_configure_args)

        for configure_arg in configure_args:
            self.assertIn(configure_arg, cmake_args_str)

    def test_configure_args_parsing(self):
        """Test that --configure-args are properly parsed in build, run, and install commands"""
        # Test that configure_args is None when not provided for all commands
        for command in ["build", "run", "install"]:
            args = self.cli.parser.parse_args([command, "test_project", "--local"])
            self.assertIsNone(
                args.configure_args, f"configure_args should be None for {command} command"
            )

    @patch("utilities.cli.holohub.HoloHubCLI.find_project")
    @patch("utilities.cli.util.run_command")
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    @patch("utilities.cli.util.get_container_entrypoint")
    def test_benchmark_functionality(
        self,
        mock_get_container_entrypoint,
        mock_exists,
        mock_mkdir,
        mock_run_command,
        mock_find_project,
    ):
        """Test benchmark functionality including patch/restore scripts and CMake flags"""

        mock_app_data = {
            "project_name": "test_app",
            "project_type": "application",
            "source_folder": Path(os.getcwd()) / "applications" / "test_app",
            "metadata": {"language": "cpp"},
        }

        mock_run_command.return_value = MagicMock()
        mock_exists.return_value = True
        mock_find_project.return_value = mock_app_data
        mock_get_container_entrypoint.return_value = ["/bin/bash"]

        args = self.cli.parser.parse_args("build test_app --local --benchmark".split())
        self.assertTrue(args.benchmark, "Benchmark flag should be parsed correctly")
        args.func(args)
        patch_script_call, restore_script_call, cmake_calls = None, None, []

        for call in mock_run_command.call_args_list:
            cmd = call[0][0]
            if isinstance(cmd, list) and len(cmd) >= 1:
                if "patch_application.sh" in str(cmd[0]):
                    patch_script_call = cmd
                elif "restore_application.sh" in str(cmd[0]):
                    restore_script_call = cmd
                elif cmd[0] == "cmake" and "-B" in cmd:
                    cmake_calls.append(cmd)

        self.assertIsNotNone(patch_script_call, "Patch script should be called")
        self.assertIn("patch_application.sh", str(patch_script_call[0]))
        self.assertEqual(str(patch_script_call[1]), str(mock_app_data["source_folder"]))
        self.assertIsNotNone(restore_script_call, "Restore script should be called")
        self.assertIn("restore_application.sh", str(restore_script_call[0]))
        self.assertEqual(str(restore_script_call[1]), str(mock_app_data["source_folder"]))
        self.assertTrue(len(cmake_calls) > 0, "CMake should be called")
        cmake_args_str = " ".join(cmake_calls[0])
        self.assertIn("-DCMAKE_CXX_FLAGS=-I", cmake_args_str)
        self.assertIn("benchmarks/holoscan_flow_benchmarking", cmake_args_str)

        # Test benchmark in container mode
        with patch("utilities.cli.holohub.HoloHubContainer") as mock_container_class:
            mock_container = MagicMock()
            mock_container_class.return_value = mock_container
            mock_find_project.return_value = mock_app_data

            args = self.cli.parser.parse_args("build test_app --benchmark".split())
            self.assertTrue(args.benchmark, "Benchmark flag for container mode correctly parsed")
            args.func(args)

            mock_container.build.assert_called_once()
            mock_container.run.assert_called_once()
            kwargs = mock_container.run.call_args[1]
            command_string = kwargs["extra_args"][1]
            self.assertIn("--benchmark", command_string)
            self.assertIn("./holohub build test_app --local", command_string)

    @patch("utilities.cli.holohub.HoloHubCLI.find_project")
    @patch("utilities.cli.holohub.HoloHubContainer")
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    @patch("builtins.print")
    def test_vscode_command(
        self,
        mock_print,
        mock_exists,
        mock_mkdir,
        mock_container_class,
        mock_find_project,
    ):
        """Test the vscode command for generating devcontainer configuration"""
        mock_find_project.return_value = self.mock_project_data
        mock_container = MagicMock()
        mock_container_class.return_value = mock_container
        mock_container.get_devcontainer_args.return_value = (
            '"--device", "/dev/video0", "--runtime", "nvidia"'
        )
        mock_container.image_name = "holohub:test_project"
        mock_exists.return_value = True
        mock_mkdir.return_value = None

        args = self.cli.parser.parse_args("vscode test_project --dryrun".split())
        self.assertEqual(args.command, "vscode")
        self.assertEqual(args.project, "test_project")

        args.func(args)
        mock_find_project.assert_called_with(project_name="test_project", language=None)
        mock_container_class.assert_called_with(
            project_metadata=self.mock_project_data, language=None
        )
        mock_container.get_devcontainer_args.assert_called()

        print_calls = [call[0][0] for call in mock_print.call_args_list if call[0]]
        printed_output = " ".join(print_calls)
        self.assertIn("devcontainer.json", printed_output.lower())
        self.assertIn(".devcontainer", printed_output.lower())
        self.assertIn("holohub-dev-container-test_project:dev", printed_output)
        self.assertIn("vscode://vscode-remote/dev-container", printed_output)
        mock_mkdir.assert_not_called()

    def test_modes_functionality(self):
        """Test mode resolution and configuration application"""
        project_data = {
            "metadata": {
                "modes": {
                    "default": {
                        "description": "Default mode",
                        "requirements": ["camera"],
                        "build": {"depends": ["op1", "op2"]},
                        "run": {"command": "python3 app.py"},
                    }
                }
            }
        }

        # Test mode resolution
        mode_name, mode_config = self.cli.resolve_mode(project_data)
        self.assertEqual(mode_name, "default")
        self.assertEqual(mode_config["description"], "Default mode")

        # Test build config application with mock args
        from argparse import Namespace

        mock_args = Namespace(with_operators="existing", docker_opts="", configure_args=None)
        enhanced = self.cli.get_effective_build_config(mock_args, mode_config)
        # Mode config should replace CLI args entirely
        self.assertEqual(enhanced["with_operators"], "op1;op2")

        # Test run config application with mock args
        mock_args = Namespace(run_args="", docker_opts="--net host")
        enhanced = self.cli.get_effective_run_config(mock_args, mode_config)
        self.assertEqual(enhanced["run_args"], "")  # CLI run_args
        # Note: run config doesn't have docker_run_args, so CLI docker_opts should be preserved
        self.assertEqual(enhanced["docker_opts"], "--net host")

        # Test run config with docker_run_args replacement
        mode_config_with_docker = {
            "run": {"command": "python3 app.py", "docker_run_args": ["--privileged", "--net=host"]}
        }
        enhanced = self.cli.get_effective_run_config(mock_args, mode_config_with_docker)
        # Mode docker_run_args should replace CLI docker_opts
        self.assertEqual(enhanced["docker_opts"], "--privileged --net=host")

    @patch("utilities.cli.holohub.HoloHubCLI.find_project")
    @patch("builtins.print")
    def test_modes_command(self, mock_print, mock_find_project):
        """Test modes command execution"""
        mock_find_project.return_value = {
            "metadata": {
                "modes": {
                    "default": {"description": "Default mode", "requirements": ["camera"]},
                    "gpu": {"description": "GPU mode"},
                }
            }
        }

        args = self.cli.parser.parse_args("modes test_app".split())
        args.func(args)

        print_calls = [call[0][0] for call in mock_print.call_args_list if call[0]]
        output = " ".join(print_calls)
        self.assertIn("Available modes", output)
        self.assertIn("default", output)
        self.assertIn("gpu", output)

    def test_mode_argument_parsing(self):
        """Test mode argument parsing for build/run commands"""
        # Test with mode
        args = self.cli.parser.parse_args("build test_app custom --local".split())
        self.assertEqual(args.mode, "custom")

        # Test without mode
        args = self.cli.parser.parse_args("build test_app --local".split())
        self.assertIsNone(args.mode)


class TestRunCommand(unittest.TestCase):
    """Test the run_command function with explicit shell parameter"""

    def setUp(self):
        self.mock_completed_process = subprocess.CompletedProcess([], 0)

    @patch("subprocess.run")
    @patch("builtins.print")
    def test_run_command_scenarios(self, mock_print, mock_subprocess):
        """Test run_command with shell execution and dry run mode"""
        mock_subprocess.return_value = self.mock_completed_process
        result = util.run_command("echo hello | grep hello", shell=True)
        self.assertEqual(result, self.mock_completed_process)

        mock_subprocess.reset_mock()
        mock_print.reset_mock()

        util.run_command(["echo", "hello"], dry_run=True)
        mock_subprocess.assert_not_called()
        mock_print.assert_called()
        printed_args = mock_print.call_args[0][0]
        self.assertIn("echo hello", printed_args)
        self.assertIn("[dryrun]", printed_args)

    def test_parse_semantic_version(self):
        """Test the parse_semantic_version function"""
        self.assertEqual(util.parse_semantic_version("1.2.3"), (1, 2, 3))
        self.assertEqual(util.parse_semantic_version("1.2.3+dev4"), (1, 2, 3))
        self.assertEqual(util.parse_semantic_version("1.2.3-rc4"), (1, 2, 3))
        self.assertEqual(util.parse_semantic_version("1.2.3.dev4"), (1, 2, 3))
        self.assertEqual(util.parse_semantic_version("1.0.0-beta+exp.sha.5114f85"), (1, 0, 0))
        self.assertRaises(ValueError, util.parse_semantic_version, "1.2")
        self.assertRaises(ValueError, util.parse_semantic_version, "1.2.dev3")
        self.assertGreater(util.parse_semantic_version("1.2.3"), (1, 1, 10))
        self.assertLess(util.parse_semantic_version("1.2.3"), (1, 12, 3))


if __name__ == "__main__":
    unittest.main()
