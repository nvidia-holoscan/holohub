#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pathlib import Path
from unittest.mock import patch

# Add the utilities directory to the Python path.
sys.path.append(str(Path(os.getcwd()) / "utilities"))

from utilities.cli import util  # noqa: E402


class TestUtilGitHelpers(unittest.TestCase):
    def setUp(self) -> None:
        self.orig_cwd = os.getcwd()

    def tearDown(self) -> None:
        try:
            os.chdir(self.orig_cwd)
        except Exception:
            pass

    @patch("utilities.cli.util.run_info_command", return_value="1a2b3c4d5e6f")
    def test_get_git_short_sha(self, mock_run_info_command):
        """Test that get_git_short_sha returns the short SHA when available."""
        sha = util.get_git_short_sha()
        self.assertEqual(sha, "1a2b3c4d5e6f")
        mock_run_info_command.assert_called()

    @patch("utilities.cli.util.run_info_command", return_value=None)
    def test_get_git_short_sha_fallback(self, mock_run_info_command):
        """Test that get_git_short_sha falls back to 'latest' when git info is unavailable."""
        sha = util.get_git_short_sha()
        self.assertEqual(sha, "latest")
        mock_run_info_command.assert_called()

    @patch("utilities.cli.util.run_info_command", return_value="Feature__Thing Name--X")
    def test_get_current_branch_slug_slugifies(self, mock_run_info_command):
        """Test that get_current_branch_slug lowercases and normalizes branch names to slugs."""
        slug = util.get_current_branch_slug()
        self.assertEqual(slug, "feature-thing-name-x")
        mock_run_info_command.assert_called()

    @patch("utilities.cli.util.run_info_command", return_value="HEAD")
    def test_get_current_branch_slug_head_fallback(self, mock_run_info_command):
        """Test that get_current_branch_slug falls back to 'latest' on detached HEAD."""
        slug = util.get_current_branch_slug()
        self.assertEqual(slug, "latest")
        mock_run_info_command.assert_called()


if __name__ == "__main__":
    unittest.main()
