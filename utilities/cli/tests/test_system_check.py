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
import unittest.mock
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
        self.assertGreater(len(result.name), 0)
        self.assertGreater(len(result.message), 0)

    @patch("utilities.cli.system_check.get_gpu_name", return_value="NVIDIA Test GPU")
    @patch("utilities.cli.system_check.run_info_command")
    def test_check_gpu_single(self, mock_run_info, mock_gpu_name):
        mock_run_info.return_value = "0, NVIDIA Test GPU, 530.0, 8.9, 24576 MiB"
        result = check_gpu()
        self._assert_valid_result(result)
        self.assertEqual(result.status, "OK")
        self.assertIn("NVIDIA Test GPU", result.message)

    @patch(
        "utilities.cli.system_check.get_gpu_name",
        return_value="NVIDIA RTX PRO 6000\nNVIDIA IGX iGPU",
    )
    @patch("utilities.cli.system_check.run_info_command")
    def test_check_gpu_multi(self, mock_run_info, mock_gpu_name):
        mock_run_info.return_value = (
            "0, NVIDIA RTX PRO 6000, 580.0, 12.0, 97249 MiB\n"
            "1, NVIDIA IGX iGPU, 580.0, 8.7, 62798 MiB"
        )
        result = check_gpu()
        self._assert_valid_result(result)
        self.assertEqual(result.status, "OK")
        self.assertIn("2 GPUs", result.message)
        self.assertIn("RTX PRO 6000", result.details)
        self.assertIn("IGX iGPU", result.details)

    @patch("utilities.cli.system_check.get_gpu_name", return_value=None)
    def test_check_gpu_without_gpu(self, mock_gpu_name):
        result = check_gpu()
        self._assert_valid_result(result)
        self.assertEqual(result.status, "FAIL")

    @patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"})
    @patch("utilities.cli.system_check.get_gpu_name", return_value="NVIDIA RTX PRO 6000")
    @patch("utilities.cli.system_check.run_info_command")
    def test_check_gpu_cuda_visible_devices(self, mock_run_info, mock_gpu_name):
        mock_run_info.return_value = "0, NVIDIA RTX PRO 6000, 580.0, 12.0, 97249 MiB"
        result = check_gpu()
        self._assert_valid_result(result)
        self.assertIn("CUDA_VISIBLE_DEVICES=0", result.message)

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

    @patch("utilities.cli.system_check.get_cuda_runtime_version", return_value=None)
    @patch("utilities.cli.system_check._get_driver_cuda_version", return_value="12")
    @patch("shutil.which", return_value=None)
    def test_check_cuda_without_nvcc_no_runtime(self, mock_which, mock_cuda, mock_runtime):
        result = check_cuda()
        self._assert_valid_result(result)
        self.assertEqual(result.status, "WARN")

    @patch("utilities.cli.system_check.get_cuda_runtime_version", return_value="12.6.77")
    @patch("utilities.cli.system_check._get_driver_cuda_version", return_value="12")
    @patch("shutil.which", return_value=None)
    def test_check_cuda_without_nvcc_with_runtime(self, mock_which, mock_cuda, mock_runtime):
        result = check_cuda()
        self._assert_valid_result(result)
        self.assertEqual(result.status, "OK")
        self.assertIn("runtime 12.6.77", result.message)

    @patch("utilities.cli.system_check.subprocess.run")
    @patch("utilities.cli.system_check.run_info_command")
    @patch("shutil.which")
    def test_check_docker_with_ctk(self, mock_which, mock_run_info, mock_subproc):
        def which_side_effect(cmd):
            return {
                "docker": "/usr/bin/docker",
                "nvidia-ctk": "/usr/bin/nvidia-ctk",
            }.get(cmd)

        def run_info_side_effect(cmd, **kwargs):
            if cmd == ["docker", "--version"]:
                return "Docker version 27.0.0, build abcdef0"
            if cmd == ["nvidia-ctk", "--version"]:
                return "NVIDIA Container Toolkit CLI version 1.17.3"
            return None

        mock_which.side_effect = which_side_effect
        mock_run_info.side_effect = run_info_side_effect
        mock_subproc.return_value = unittest.mock.Mock(returncode=0)

        result = check_docker()
        self._assert_valid_result(result)
        self.assertEqual(result.status, "OK")
        self.assertIn("27.0.0", result.message)
        self.assertIn("nvidia-ctk 1.17.3", result.message)

    @patch("utilities.cli.system_check.subprocess.run")
    @patch("utilities.cli.system_check.run_info_command")
    @patch("shutil.which")
    def test_check_docker_without_ctk(self, mock_which, mock_run_info, mock_subproc):
        def which_side_effect(cmd):
            return "/usr/bin/docker" if cmd == "docker" else None

        def run_info_side_effect(cmd, **kwargs):
            if cmd == ["docker", "--version"]:
                return "Docker version 27.0.0, build abcdef0"
            return None

        mock_which.side_effect = which_side_effect
        mock_run_info.side_effect = run_info_side_effect
        mock_subproc.return_value = unittest.mock.Mock(returncode=0)

        result = check_docker()
        self._assert_valid_result(result)
        self.assertEqual(result.status, "WARN")
        self.assertIn("27.0.0", result.message)
        self.assertNotIn("nvidia-ctk", result.message)
        self.assertIn("NVIDIA Container Toolkit", result.fix_suggestion)

    @patch("utilities.cli.system_check.is_running_in_docker", return_value=False)
    @patch("shutil.which", return_value=None)
    def test_check_docker_not_installed(self, mock_which, mock_docker):
        result = check_docker()
        self._assert_valid_result(result)
        self.assertEqual(result.status, "WARN")
        self.assertIn("not installed", result.message)

    @patch("utilities.cli.system_check.is_running_in_docker", return_value=True)
    @patch("shutil.which", return_value=None)
    def test_check_docker_not_installed_in_container(self, mock_which, mock_docker):
        result = check_docker()
        self._assert_valid_result(result)
        self.assertEqual(result.status, "SKIP")
        self.assertIn("inside container", result.message)

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
        os.environ.pop("DISPLAY", None)
        result = check_display()
        self._assert_valid_result(result)
        self.assertEqual(result.status, "WARN")


class TestRunAllChecks(unittest.TestCase):
    """Test the run_all_checks orchestrator with mocked check functions"""

    _mock_results = [
        CheckResult(status="OK", name="GPU", message="mock gpu"),
        CheckResult(status="OK", name="CUDA", message="mock cuda"),
        CheckResult(status="OK", name="Docker", message="mock docker"),
        CheckResult(status="OK", name="Holoscan", message="mock sdk"),
        CheckResult(status="OK", name="Disk", message="mock disk"),
        CheckResult(status="OK", name="CLI", message="mock cli"),
        CheckResult(status="OK", name="Container", message="mock container"),
        CheckResult(status="WARN", name="Display", message="mock display"),
        CheckResult(status="SKIP", name="Devices", message="mock devices"),
    ]

    def _patch_checks(self):
        """Patch all individual check functions to return mock results"""
        check_names = [
            "check_gpu",
            "check_cuda",
            "check_docker",
            "check_holoscan",
            "check_disk",
            "check_cli",
            "check_container",
            "check_display",
            "check_devices",
        ]
        patchers = []
        for name, result in zip(check_names, self._mock_results):
            p = patch(f"utilities.cli.system_check.{name}", return_value=result)
            p.start()
            patchers.append(p)
        return patchers

    def test_run_all_checks_returns_list(self):
        patchers = self._patch_checks()
        try:
            results = run_all_checks()
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), 9)
            for r in results:
                self.assertIsInstance(r, CheckResult)
        finally:
            for p in patchers:
                p.stop()

    def test_run_all_checks_covers_expected_names(self):
        patchers = self._patch_checks()
        try:
            results = run_all_checks()
            names = {r.name for r in results}
            for expected in (
                "GPU",
                "CUDA",
                "Docker",
                "Holoscan",
                "Disk",
                "CLI",
                "Container",
                "Display",
                "Devices",
            ):
                self.assertIn(expected, names, f"Missing check: {expected}")
        finally:
            for p in patchers:
                p.stop()

    def test_run_all_checks_handles_exception(self):
        """If a check raises, run_all_checks catches it and returns a FAIL result"""
        with patch("utilities.cli.system_check.check_gpu", side_effect=RuntimeError("boom")), patch(
            "utilities.cli.system_check.check_cuda", return_value=self._mock_results[1]
        ), patch(
            "utilities.cli.system_check.check_docker", return_value=self._mock_results[2]
        ), patch(
            "utilities.cli.system_check.check_holoscan", return_value=self._mock_results[3]
        ), patch(
            "utilities.cli.system_check.check_disk", return_value=self._mock_results[4]
        ), patch(
            "utilities.cli.system_check.check_cli", return_value=self._mock_results[5]
        ), patch(
            "utilities.cli.system_check.check_container", return_value=self._mock_results[6]
        ), patch(
            "utilities.cli.system_check.check_display", return_value=self._mock_results[7]
        ), patch(
            "utilities.cli.system_check.check_devices", return_value=self._mock_results[8]
        ):
            results = run_all_checks()
            self.assertEqual(results[0].status, "FAIL")
            self.assertIn("boom", results[0].message)


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
