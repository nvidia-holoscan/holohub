#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(os.getcwd()) / "utilities"))

from utilities.cli.system_check import (
    CheckResult,
    check_cli,
    check_container,
    check_cuda,
    check_devices,
    check_disk,
    check_display,
    check_docker,
    check_gpu,
    check_python,
    format_results,
    format_results_json,
    run_all_checks,
)


class TestCheckResult(unittest.TestCase):
    """Test the CheckResult dataclass"""

    def test_check_result_creation(self):
        result = CheckResult(status="OK", name="Test", message="all good")
        self.assertEqual(result.status, "OK")
        self.assertEqual(result.name, "Test")
        self.assertEqual(result.message, "all good")
        self.assertIsNone(result.fix_suggestion)

    def test_check_result_with_fix(self):
        result = CheckResult(status="FAIL", name="Test", message="bad", fix_suggestion="fix it")
        self.assertEqual(result.fix_suggestion, "fix it")


class TestIndividualChecks(unittest.TestCase):
    """Test individual check functions return valid CheckResult on any system"""

    def _assert_valid_result(self, result):
        self.assertIsInstance(result, CheckResult)
        self.assertIn(result.status, ("OK", "WARN", "FAIL", "SKIP"))
        self.assertTrue(len(result.name) > 0)
        self.assertTrue(len(result.message) > 0)

    @patch("utilities.cli.system_check.get_gpu_name", return_value="NVIDIA Test GPU")
    @patch("utilities.cli.system_check.run_info_command")
    def test_check_gpu_with_gpu(self, mock_run_info, mock_gpu_name):
        mock_run_info.return_value = "530.0"
        result = check_gpu()
        self._assert_valid_result(result)
        self.assertEqual(result.status, "OK")
        self.assertIn("NVIDIA Test GPU", result.message)

    @patch("utilities.cli.system_check.get_gpu_name", return_value=None)
    def test_check_gpu_without_gpu(self, mock_gpu_name):
        result = check_gpu()
        self._assert_valid_result(result)
        self.assertEqual(result.status, "FAIL")

    def test_check_python(self):
        """Python check should always pass since we require >= 3.10 to run"""
        result = check_python()
        self._assert_valid_result(result)
        self.assertEqual(result.status, "OK")
        self.assertIn("3.10", result.message)

    def test_check_disk(self):
        """Disk check should work on any system"""
        result = check_disk()
        self._assert_valid_result(result)
        self.assertIn(result.status, ("OK", "WARN", "FAIL"))
        self.assertIn("GB", result.message)

    def test_check_cli(self):
        result = check_cli()
        self._assert_valid_result(result)
        self.assertEqual(result.status, "OK")
        self.assertIn("holohub", result.message)

    def test_check_container(self):
        result = check_container()
        self._assert_valid_result(result)
        self.assertEqual(result.status, "OK")

    @patch("utilities.cli.system_check.get_default_cuda_version", return_value="12")
    @patch("shutil.which", return_value=None)
    def test_check_cuda_without_nvcc(self, mock_which, mock_cuda):
        result = check_cuda()
        self._assert_valid_result(result)
        self.assertEqual(result.status, "WARN")

    @patch("shutil.which", return_value="/usr/bin/docker")
    @patch("utilities.cli.system_check.run_info_command")
    def test_check_docker_installed(self, mock_run_info, mock_which):
        mock_run_info.return_value = "Docker version 27.0.0"
        result = check_docker()
        self._assert_valid_result(result)

    @patch("shutil.which", return_value=None)
    def test_check_docker_not_installed(self, mock_which):
        result = check_docker()
        self._assert_valid_result(result)
        self.assertEqual(result.status, "WARN")
        self.assertIn("not installed", result.message)

    def test_check_devices(self):
        """Devices check should return OK or SKIP depending on hardware"""
        result = check_devices()
        self._assert_valid_result(result)
        self.assertIn(result.status, ("OK", "SKIP"))

    @patch.dict(os.environ, {"DISPLAY": ":0"})
    @patch("os.path.exists", return_value=True)
    def test_check_display_with_display(self, mock_exists):
        result = check_display()
        self._assert_valid_result(result)
        self.assertEqual(result.status, "OK")

    @patch.dict(os.environ, {}, clear=True)
    @patch("utilities.cli.system_check.is_running_in_docker", return_value=False)
    def test_check_display_without_display(self, mock_docker):
        # Ensure DISPLAY is not set
        os.environ.pop("DISPLAY", None)
        result = check_display()
        self._assert_valid_result(result)
        self.assertEqual(result.status, "WARN")


class TestRunAllChecks(unittest.TestCase):
    """Test the run_all_checks orchestrator"""

    def test_run_all_checks_returns_list(self):
        results = run_all_checks()
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIsInstance(r, CheckResult)

    def test_run_all_checks_covers_expected_names(self):
        results = run_all_checks()
        names = {r.name for r in results}
        # These should always be present regardless of hardware
        for expected in ("Python", "Disk", "CLI", "Container", "Display"):
            self.assertIn(expected, names, f"Missing check: {expected}")


class TestFormatResults(unittest.TestCase):
    """Test output formatting"""

    def test_format_results_contains_all_checks(self):
        results = [
            CheckResult(status="OK", name="GPU", message="test gpu"),
            CheckResult(status="WARN", name="Display", message="no display"),
            CheckResult(status="SKIP", name="Devices", message="none"),
        ]
        output = format_results(results, elapsed=0.5)
        self.assertIn("System Info", output)
        self.assertIn("0.5s", output)
        self.assertIn("GPU", output)
        self.assertIn("Display", output)
        self.assertIn("Devices", output)

    def test_format_results_shows_fix_for_fail(self):
        results = [
            CheckResult(status="FAIL", name="Test", message="bad", fix_suggestion="do this"),
        ]
        output = format_results(results, elapsed=0.1)
        self.assertIn("do this", output)

    def test_format_results_json_is_valid(self):
        results = [
            CheckResult(status="OK", name="Python", message="3.12"),
            CheckResult(status="WARN", name="Display", message="not set"),
        ]
        output = format_results_json(results, elapsed=0.1)
        data = json.loads(output)
        self.assertIn("checks", data)
        self.assertIn("summary", data)
        self.assertEqual(data["summary"]["ok"], 1)
        self.assertEqual(data["summary"]["warn"], 1)
        self.assertAlmostEqual(data["elapsed_seconds"], 0.1, places=1)


if __name__ == "__main__":
    unittest.main()
