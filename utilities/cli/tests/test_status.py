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
    FolderInfo,
    GitInfo,
    ImageInfo,
    PlatformInfo,
    collect_build_info,
    collect_docker_disk_usage,
    collect_folder_info,
    collect_git_info,
    collect_image_info,
    collect_platform_info,
    format_status,
    format_status_json,
)
from utilities.cli.util import dir_size_mb, format_size, relative_time


def _status_args():
    return [
        PlatformInfo("x86_64", "dgpu", "RTX 4090", "12.6", "2.5.0"),
        GitInfo("main", "abc1234", True, 2),
        [ImageInfo("holohub:latest", "2h ago", "Running")],
        [BuildInfo("build", "OK", "1h ago")],
        [FolderInfo("/tmp/build", 512.0)],
        [FolderInfo("/tmp/data", 1024.0)],
        "Images: 5GB, Containers: 100MB",
    ]


class TestStatusUtilities(unittest.TestCase):
    def test_relative_time(self):
        for seconds, expected, exact in [
            (10, "just now", True),
            (300, "m ago", False),
            (7200, "h ago", False),
            (172800, "d ago", False),
        ]:
            result = relative_time(time.time() - seconds)
            self.assertEqual(result, expected) if exact else self.assertIn(expected, result)

    def test_format_size(self):
        for size_mb, expected in [(512, "512 MB"), (2048, "2.0 GB")]:
            self.assertEqual(format_size(size_mb), expected)

    def test_dir_size_mb(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertEqual(dir_size_mb(root), 0.0)
            (root / "test.bin").write_bytes(b"\x00" * 1024)
            self.assertAlmostEqual(dir_size_mb(root), 1024 / (1024 * 1024), places=4)
        self.assertEqual(dir_size_mb(Path("/nonexistent/path")), 0.0)


class TestStatusCollectors(unittest.TestCase):
    @patch("utilities.cli.status.get_host_arch", return_value="x86_64")
    @patch("utilities.cli.status.get_host_gpu", return_value="dgpu")
    @patch("utilities.cli.status.get_gpu_name", return_value="NVIDIA RTX 4090")
    @patch("utilities.cli.status.get_default_cuda_version", return_value="12.6")
    def test_collect_platform_info(self, *_mocks):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "VERSION").write_text("2.5.0")
            with patch.dict(os.environ, {"HOLOHUB_DEFAULT_HSDK_DIR": d}):
                info = collect_platform_info()
        self.assertEqual(
            (info.arch, info.gpu_type, info.gpu_name, info.cuda_version, info.holoscan_version),
            ("x86_64", "dgpu", "NVIDIA RTX 4090", "12.6", "2.5.0"),
        )

    @patch("utilities.cli.status.run_info_command")
    def test_collect_git_info(self, mock_run):
        mock_run.side_effect = lambda cmd, **_: {
            "branch": "main",
            "rev-parse": "abc1234",
            "status": " M file.py\n",
        }.get(cmd[3] if len(cmd) > 3 else "", "")
        info = collect_git_info(Path("/tmp"))
        self.assertIsNotNone(info)
        self.assertEqual((info.dirty, info.modified_count), (True, 1))
        mock_run.side_effect = None
        mock_run.return_value = None
        self.assertIsNone(collect_git_info(Path("/tmp")))

    @patch("utilities.cli.status.run_info_command")
    def test_collect_image_info(self, mock_run):
        mock_run.side_effect = lambda cmd: (
            "holohub:latest\tabc123"
            if "ps" in cmd
            else (
                "img001\tholohub:latest\t2 hours ago\nimg002\tother:v1\t1 day ago"
                if "images" in cmd
                else None
            )
        )
        images = collect_image_info()
        self.assertEqual((len(images), images[0].status), (1, "Running"))
        mock_run.return_value = None
        mock_run.side_effect = None
        self.assertEqual(collect_image_info(), [])

    def test_collect_build_info(self):
        def mk_build(root, name="build", configure_ok=True):
            build = root / name
            build.mkdir()
            (build / "CMakeCache.txt").touch()
            if configure_ok:
                (build / "Makefile").touch()
            return build

        # Makefile present → configure succeeded → OK
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            mk_build(root, "build-x86_64")
            builds = collect_build_info(root)
            self.assertEqual(
                (len(builds), builds[0].name, builds[0].status), (1, "build-x86_64", "OK")
            )

        # build.ninja also indicates configure success → OK
        with tempfile.TemporaryDirectory() as d:
            build = Path(d) / "build"
            build.mkdir()
            (build / "CMakeCache.txt").touch()
            (build / "build.ninja").touch()
            self.assertEqual(collect_build_info(Path(d))[0].status, "OK")

        # CMakeCache.txt only, no generator output → configure failed → FAIL
        with tempfile.TemporaryDirectory() as d:
            mk_build(Path(d), configure_ok=False)
            self.assertEqual(collect_build_info(Path(d))[0].status, "FAIL")
        for path in [Path("/nonexistent"), Path(tempfile.mkdtemp())]:
            self.assertEqual(collect_build_info(path), [])

    def test_collect_folder_info(self):
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "data"
            sub.mkdir()
            (sub / "file.bin").write_bytes(b"\x00" * 2048)
            result = collect_folder_info([sub])
            self.assertEqual(len(result), 1)
            self.assertGreater(result[0].size_mb, 0)
            self.assertEqual(len(collect_folder_info([sub, sub])), 1)

    @patch("utilities.cli.status.run_info_command")
    def test_collect_docker_disk_usage(self, mock_run):
        mock_run.return_value = "Images\t5.2GB\nContainers\t100MB"
        result = collect_docker_disk_usage()
        self.assertIn("Images: 5.2GB", result)
        self.assertIn("Containers: 100MB", result)
        mock_run.return_value = None
        self.assertIsNone(collect_docker_disk_usage())


class TestStatusFormatting(unittest.TestCase):
    def test_format_status_text(self):
        args = _status_args()
        output = format_status(*args)
        for expected in [
            "x86_64",
            "RTX 4090",
            "main",
            "abc1234",
            "2 modified",
            "holohub:latest",
            "Build folders:",
            "Data folders:",
        ]:
            self.assertIn(expected, output)

        args[1] = None
        output = format_status(*args)
        self.assertIn("x86_64", output)
        self.assertNotIn("Git:", output)

        args = _status_args()
        args[2] = []
        output = format_status(*args)
        self.assertIn("Images:", output)
        self.assertIn("(none)", output)

    def test_format_status_json(self):
        data = json.loads(format_status_json(*_status_args()))
        self.assertEqual(data["platform"]["arch"], "x86_64")
        self.assertEqual(data["git"]["branch"], "main")
        self.assertTrue(data["git"]["dirty"])
        self.assertEqual(len(data["images"]), 1)
        self.assertEqual(len(data["builds"]), 1)
        self.assertIn("docker_disk", data)

        args = _status_args()
        args[6] = None
        self.assertNotIn("docker_disk", json.loads(format_status_json(*args)))


if __name__ == "__main__":
    unittest.main()
