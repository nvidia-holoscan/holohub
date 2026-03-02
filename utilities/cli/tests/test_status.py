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
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(os.getcwd()) / "utilities"))

from utilities.cli.status import (
    BuildInfo,
    CacheDir,
    CacheInfo,
    ContainerInfo,
    DeviceInfo,
    PlatformInfo,
    collect_build_info,
    collect_cache_info,
    collect_device_info,
    collect_platform_info,
    format_status,
    format_status_json,
    format_status_short,
)


class TestCollectPlatformInfo(unittest.TestCase):
    """Test platform info collection works on any system"""

    def test_returns_platform_info(self):
        info = collect_platform_info()
        self.assertIsInstance(info, PlatformInfo)
        self.assertIn(info.arch, ("x86_64", "aarch64"))
        self.assertIn(info.gpu_type, ("dgpu", "igpu"))
        self.assertIn(info.cuda_version, ("12", "13"))


class TestCollectBuildInfo(unittest.TestCase):
    """Test build directory scanning"""

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builds = collect_build_info(Path(tmpdir))
            self.assertEqual(builds, [])

    def test_nonexistent_dir(self):
        builds = collect_build_info(Path("/nonexistent/path"))
        self.assertEqual(builds, [])

    def test_detects_configured_build(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            build_dir = Path(tmpdir) / "my_project"
            build_dir.mkdir()
            (build_dir / "CMakeCache.txt").touch()
            builds = collect_build_info(Path(tmpdir))
            self.assertEqual(len(builds), 1)
            self.assertEqual(builds[0].name, "my_project")
            self.assertEqual(builds[0].status, "OK")

    def test_detects_failed_build(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            build_dir = Path(tmpdir) / "bad_project"
            build_dir.mkdir()
            (build_dir / "CMakeCache.txt").touch()
            cmake_files = build_dir / "CMakeFiles"
            cmake_files.mkdir()
            error_log = cmake_files / "CMakeError.log"
            error_log.write_text("some error")
            builds = collect_build_info(Path(tmpdir))
            self.assertEqual(len(builds), 1)
            self.assertEqual(builds[0].status, "FAIL")

    def test_skips_dirs_without_cmake_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "random_dir").mkdir()
            builds = collect_build_info(Path(tmpdir))
            self.assertEqual(builds, [])


class TestCollectCacheInfo(unittest.TestCase):
    """Test cache info collection"""

    def test_empty_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            build_dir = root / "build"
            data_dir = root / "data"
            cache = collect_cache_info(root, build_dir, data_dir)
            self.assertIsInstance(cache, CacheInfo)
            self.assertEqual(cache.build, [])
            self.assertEqual(cache.data, [])
            self.assertEqual(cache.install, [])
            self.assertAlmostEqual(cache.total_mb, 0.0)

    def test_detects_existing_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            build_dir = root / "build"
            build_dir.mkdir()
            # Write a small file so size > 0
            (build_dir / "file.txt").write_text("hello")
            data_dir = root / "data"
            data_dir.mkdir()
            cache = collect_cache_info(root, build_dir, data_dir)
            self.assertGreater(len(cache.build), 0)
            self.assertGreater(len(cache.data), 0)
            self.assertGreater(cache.total_mb, 0)


class TestCollectDeviceInfo(unittest.TestCase):
    """Test device detection returns valid structure"""

    def test_returns_list(self):
        devices = collect_device_info()
        self.assertIsInstance(devices, list)
        for d in devices:
            self.assertIsInstance(d, DeviceInfo)
            self.assertTrue(len(d.category) > 0)
            self.assertIsInstance(d.devices, list)


class TestFormatStatus(unittest.TestCase):
    """Test status formatting"""

    def _make_sample_data(self):
        platform = PlatformInfo(
            arch="x86_64",
            gpu_type="dgpu",
            gpu_name="Test GPU",
            cuda_version="12",
            holoscan_version="3.11.0",
        )
        containers = [
            ContainerInfo(
                name="test",
                image="holohub:test",
                size="1.2GB",
                created="Built 1h ago",
                status="Stopped",
            )
        ]
        builds = [
            BuildInfo(
                name="my_app", status="OK", last_modified="5 min ago", path="/tmp/build/my_app"
            )
        ]
        devices = [DeviceInfo(category="V4L2", devices=["/dev/video0"])]
        cache = CacheInfo(
            build=[CacheDir(path="/tmp/build", size_mb=100.0)],
            data=[CacheDir(path="/tmp/data", size_mb=500.0)],
            install=[],
            total_mb=600.0,
        )
        return platform, containers, builds, devices, cache

    def test_format_status_contains_sections(self):
        platform, containers, builds, devices, cache = self._make_sample_data()
        output = format_status(platform, containers, builds, devices, cache)
        self.assertIn("Platform:", output)
        self.assertIn("Containers:", output)
        self.assertIn("Builds (local):", output)
        self.assertIn("Devices:", output)
        self.assertIn("Cache (clear-cache):", output)
        self.assertIn("holohub:test", output)
        self.assertIn("my_app", output)
        self.assertIn("/dev/video0", output)

    def test_format_status_empty(self):
        platform = PlatformInfo("x86_64", "dgpu", None, "12", "not found")
        output = format_status(platform, [], [], [])
        self.assertIn("(none found)", output)
        self.assertIn("(none detected)", output)

    def test_format_status_short(self):
        platform, containers, builds, devices, cache = self._make_sample_data()
        output = format_status_short(platform, containers, builds, devices, cache)
        # Single line, pipe-separated
        self.assertNotIn("\n", output)
        self.assertIn("|", output)
        self.assertIn("cache:", output)

    def test_format_status_json_is_valid(self):
        platform, containers, builds, devices, cache = self._make_sample_data()
        output = format_status_json(platform, containers, builds, devices, cache)
        data = json.loads(output)
        self.assertIn("platform", data)
        self.assertIn("containers", data)
        self.assertIn("builds", data)
        self.assertIn("devices", data)
        self.assertIn("cache", data)
        self.assertEqual(data["platform"]["arch"], "x86_64")
        self.assertAlmostEqual(data["cache"]["total_mb"], 600.0)


class TestBuildTiming(unittest.TestCase):
    """Test build timing utilities"""

    def test_timing_functions(self):
        from utilities.cli.util import (
            clear_command_timings,
            format_timing_summary,
            get_command_timings,
        )

        clear_command_timings()
        self.assertEqual(get_command_timings(), [])
        self.assertEqual(format_timing_summary(), "")

    @patch("subprocess.run")
    @patch("builtins.print")
    def test_run_command_records_timing(self, mock_print, mock_run):
        import subprocess

        from utilities.cli.util import clear_command_timings, get_command_timings, run_command

        mock_run.return_value = subprocess.CompletedProcess([], 0)
        clear_command_timings()
        run_command(["echo", "test"], check=False)
        timings = get_command_timings()
        self.assertEqual(len(timings), 1)
        self.assertGreaterEqual(timings[0][1], 0)
        clear_command_timings()


if __name__ == "__main__":
    unittest.main()
