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

"""
Version checking utility for holohub CLI.
"""

import os
import subprocess
import time
from pathlib import Path


def check_for_cli_updates() -> None:
    """
    Check if a newer version of the CLI is available.

    - Checks once per 24 hours (configurable via CLI_CHECK_INTERVAL)
    - Compares local commit hash with remote HEAD or branch
    - Shows update notification if newer version available
    - Fails silently on network issues or missing files

    Environment Variables:
        CLI_SKIP_UPDATE_CHECK: Set to "1" to disable update checks
        CLI_CHECK_INTERVAL: Seconds between checks (default: 86400 = 24 hours)
        CLI_REPO_URL: Custom repository URL (default: https://github.com/nvidia-holoscan/holohub.git)
        HOLOHUB_BRANCH: Branch to track (default: main)
        CLI_PINNED_COMMIT: If set, pin the CLI to this commit hash (short or full).
            - When set, remote update checks are skipped.
            - If the local CLI commit differs from the pinned commit, a warning is shown.
    """
    if os.environ.get("CLI_SKIP_UPDATE_CHECK") == "1":  # Skip if user disabled update checks
        return

    commit_file = Path(__file__).with_name(".cli_commit_hash")
    last_check_file = Path(__file__).with_name(".cli_last_check")

    pinned_commit = os.environ.get("CLI_PINNED_COMMIT", "").strip()

    local_hash = ""
    try:
        local_hash = commit_file.read_text().strip()
    except OSError:
        # If we can't read the local hash, we can't enforce pinning, but we can still
        # optionally perform remote checks later.
        pass

    if pinned_commit:
        # Allow short hash vs full hash comparisons by treating the shorter one as a prefix.
        def _hashes_match(a: str, b: str) -> bool:
            if not a or not b:
                return False
            if len(a) == len(b):
                return a == b
            if len(a) < len(b):
                return b.startswith(a)
            return a.startswith(b)

        if not _hashes_match(local_hash, pinned_commit):
            cmd_name = os.environ.get("HOLOHUB_CMD_NAME", "./holohub")
            print()
            print("════════════════════════════════════════════════════")
            print(f"⚠️  {cmd_name} CLI version does not match pinned commit.")
            if local_hash:
                print(f"Current:  {local_hash[:8]}")
            else:
                print("Current:  <unknown>")
            print(f"Pinned:   {pinned_commit[:8]}")
            print("Please reinstall or checkout holohub at the pinned commit.")
            print("Remote update checks are skipped while CLI_PINNED_COMMIT is set.")
            print("════════════════════════════════════════════════════")

        # When a pin is set we do not perform any remote update checks.
        return

    try:  # Default: check once per 24 hours
        check_interval = int(os.environ.get("CLI_CHECK_INTERVAL", "86400"))
    except ValueError:
        check_interval = 86400

    current_time, last_check = int(time.time()), 0
    try:
        if last_check_file.exists():
            last_check = int(last_check_file.read_text().strip())
    except (OSError, ValueError):
        pass  # Treat unreadable or corrupted timestamp as "never checked"

    if (current_time - last_check) < check_interval:
        return  # Too soon to check again

    try:  # Update last check timestamp atomically
        temp_file = last_check_file.with_suffix(".tmp")
        temp_file.write_text(str(current_time))
        temp_file.replace(last_check_file)
    except OSError:
        return

    repo_url = os.environ.get("CLI_REPO_URL", "https://github.com/nvidia-holoscan/holohub.git")
    branch = os.environ.get("HOLOHUB_BRANCH") or "main"

    try:
        result = subprocess.run(
            ["git", "ls-remote", repo_url, f"refs/heads/{branch}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            remote_hash = result.stdout.split()[0]
        else:
            return  # Git command failed, skip silently
    except (subprocess.TimeoutExpired, FileNotFoundError, IndexError):
        return  # Network issue or git not available, skip silently

    if local_hash and remote_hash and local_hash != remote_hash:
        cmd_name = os.environ.get("HOLOHUB_CMD_NAME", "./holohub")
        print()
        print("════════════════════════════════════════════════════")
        print(f"📦 A new version of {cmd_name} CLI is available!")
        print(f"Current: {local_hash[:8]} | Latest: {remote_hash[:8]}")
        print("To update, run:")
        print(f"  CLI_FORCE_UPDATE=1 {cmd_name} --help")
        print("════════════════════════════════════════════════════")


if __name__ == "__main__":
    try:
        check_for_cli_updates()
    except Exception:
        # Suppress all exceptions to ensure CLI always runs, even if version check fails.
        pass
