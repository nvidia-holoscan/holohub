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
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(os.getcwd()) / "utilities"))

from utilities.cli.status import (
    BuildInfo,
    ContainerInfo,
    FolderInfo,
    GitInfo,
    PlatformInfo,
    _dir_size_mb,
    _format_size,
    _relative_time,
    collect_build_info,
    collect_container_info,
    collect_docker_disk_usage,
    collect_folder_info,
    collect_git_info,
    collect_platform_info,
    format_status,
    format_status_json,
)


class TestRelativeTime(unittest.TestCase):
    def test_just_now(self):
        self.assertEqual(_relative_time(time.time() - 10), "just now")

    def test_minutes(self):
        self.assertIn("m ago", _relative_time(time.time() - 300))

    def test_hours(self):
        self.assertIn("h ago", _relative_time(time.time() - 7200))

    def test_days(self):
        self.assertIn("d ago", _relative_time(time.time() - 172800))


class TestFormatSize(unittest.TestCase):
    def test_megabytes(self):
        self.assertEqual(_format_size(512), "512 MB")

    def test_gigabytes(self):
        self.assertEqual(_format_size(2048), "2.0 GB")


class TestDirSizeMb(unittest.TestCase):
    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(_dir_size_mb(Path(d)), 0.0)

    def test_dir_with_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "test.bin"
            p.write_bytes(b"\x00" * 1024)
            size = _dir_size_mb(Path(d))
            self.assertAlmostEqual(size, 1024 / (1024 * 1024), places=4)

    def test_nonexistent_dir(self):
        self.assertEqual(_dir_size_mb(Path("/nonexistent/path")), 0.0)


class TestCollectPlatformInfo(unittest.TestCase):
    @patch("utilities.cli.status.get_host_arch", return_value="x86_64")
    @patch("utilities.cli.status.get_host_gpu", return_value="dgpu")
    @patch("utilities.cli.status.get_gpu_name", return_value="NVIDIA RTX 4090")
    @patch("utilities.cli.status.get_default_cuda_version", return_value="12.6")
    def test_basic(self, *_mocks):
        with tempfile.TemporaryDirectory() as d:
            version_file = Path(d) / "VERSION"
            version_file.write_text("2.5.0")
            with patch.dict(os.environ, {"HOLOHUB_DEFAULT_HSDK_DIR": d}):
                info = collect_platform_info()
        self.assertEqual(info.arch, "x86_64")
        self.assertEqual(info.gpu_type, "dgpu")
        self.assertEqual(info.gpu_name, "NVIDIA RTX 4090")
        self.assertEqual(info.cuda_version, "12.6")
        self.assertEqual(info.holoscan_version, "2.5.0")


class TestCollectGitInfo(unittest.TestCase):
    @patch("utilities.cli.status.run_info_command")
    def test_normal(self, mock_run):
        mock_run.side_effect = lambda cmd, **kw: {
            "branch": "main",
            "rev-parse": "abc1234",
            "status": " M file.py\n",
        }.get(cmd[3] if len(cmd) > 3 else "", "")
        info = collect_git_info(Path("/tmp"))
        self.assertIsNotNone(info)
        self.assertTrue(info.dirty)
        self.assertEqual(info.modified_count, 1)

    @patch("utilities.cli.status.run_info_command", return_value=None)
    def test_no_git(self, _mock):
        self.assertIsNone(collect_git_info(Path("/tmp")))


class TestCollectContainerInfo(unittest.TestCase):
    @patch("utilities.cli.status.run_info_command")
    def test_with_containers(self, mock_run):
        def side_effect(cmd):
            if "ps" in cmd:
                return "holohub:latest"
            if "images" in cmd:
                return "holohub:latest\t2 hours ago\nother:v1\t1 day ago"
            return None

        mock_run.side_effect = side_effect
        containers = collect_container_info()
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0].status, "Running")

    @patch("utilities.cli.status.run_info_command", return_value=None)
    def test_no_docker(self, _mock):
        self.assertEqual(collect_container_info(), [])


class TestCollectBuildInfo(unittest.TestCase):
    def test_with_cmake_builds(self):
        with tempfile.TemporaryDirectory() as d:
            build = Path(d) / "build-x86_64"
            build.mkdir()
            (build / "CMakeCache.txt").touch()
            builds = collect_build_info(Path(d))
            self.assertEqual(len(builds), 1)
            self.assertEqual(builds[0].name, "build-x86_64")
            self.assertEqual(builds[0].status, "OK")

    def test_with_error_log(self):
        with tempfile.TemporaryDirectory() as d:
            build = Path(d) / "build"
            cmake_files = build / "CMakeFiles"
            cmake_files.mkdir(parents=True)
            (build / "CMakeCache.txt").touch()
            (cmake_files / "CMakeError.log").write_text("error occurred")
            builds = collect_build_info(Path(d))
            self.assertEqual(builds[0].status, "FAIL")

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(collect_build_info(Path(d)), [])

    def test_nonexistent_dir(self):
        self.assertEqual(collect_build_info(Path("/nonexistent")), [])


class TestCollectFolderInfo(unittest.TestCase):
    def test_existing_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "data"
            sub.mkdir()
            (sub / "file.bin").write_bytes(b"\x00" * 2048)
            result = collect_folder_info([sub])
            self.assertEqual(len(result), 1)
            self.assertGreater(result[0].size_mb, 0)

    def test_deduplicates(self):
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "data"
            sub.mkdir()
            result = collect_folder_info([sub, sub])
            self.assertEqual(len(result), 1)


class TestCollectDockerDiskUsage(unittest.TestCase):
    @patch("utilities.cli.status.run_info_command", return_value="Images\t5.2GB\nContainers\t100MB")
    def test_parses_output(self, _mock):
        result = collect_docker_disk_usage()
        self.assertIn("Images: 5.2GB", result)
        self.assertIn("Containers: 100MB", result)

    @patch("utilities.cli.status.run_info_command", return_value=None)
    def test_no_docker(self, _mock):
        self.assertIsNone(collect_docker_disk_usage())


class TestFormatStatus(unittest.TestCase):
    def _make_args(self):
        platform = PlatformInfo("x86_64", "dgpu", "RTX 4090", "12.6", "2.5.0")
        git = GitInfo("main", "abc1234", True, 2)
        containers = [ContainerInfo("holohub:latest", "2h ago", "Running")]
        builds = [BuildInfo("build", "OK", "1h ago")]
        build_folders = [FolderInfo("/tmp/build", 512.0)]
        data_folders = [FolderInfo("/tmp/data", 1024.0)]
        docker_disk = "Images: 5GB, Containers: 100MB"
        return platform, git, containers, builds, build_folders, data_folders, docker_disk

    def test_text_output(self):
        output = format_status(*self._make_args())
        self.assertIn("x86_64", output)
        self.assertIn("RTX 4090", output)
        self.assertIn("main", output)
        self.assertIn("abc1234", output)
        self.assertIn("2 modified", output)
        self.assertIn("holohub:latest", output)
        self.assertIn("Build folders:", output)
        self.assertIn("Data folders:", output)

    def test_text_no_git(self):
        args = list(self._make_args())
        args[1] = None
        output = format_status(*args)
        self.assertIn("x86_64", output)
        self.assertNotIn("Git:", output)

    def test_text_empty_containers(self):
        args = list(self._make_args())
        args[2] = []
        output = format_status(*args)
        self.assertIn("Containers:", output)
        self.assertIn("(none)", output)

    def test_json_output(self):
        output = format_status_json(*self._make_args())
        data = json.loads(output)
        self.assertEqual(data["platform"]["arch"], "x86_64")
        self.assertEqual(data["git"]["branch"], "main")
        self.assertTrue(data["git"]["dirty"])
        self.assertEqual(len(data["containers"]), 1)
        self.assertEqual(len(data["builds"]), 1)
        self.assertIn("docker_disk", data)

    def test_json_no_docker_disk(self):
        args = list(self._make_args())
        args[6] = None
        data = json.loads(format_status_json(*args))
        self.assertNotIn("docker_disk", data)


if __name__ == "__main__":
    unittest.main()
