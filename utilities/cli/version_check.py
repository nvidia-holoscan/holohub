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

"""
Version check utility for holohub CLI.
"""

import os
from pathlib import Path


def _hashes_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return long_.startswith(short)


def check_for_cli_updates() -> None:
    """
    Validate that the downloaded CLI matches the commit hash (when specified).

    Environment Variables:
        CLI_SKIP_UPDATE_CHECK: Set to "1" to disable the pin check.
        CLI_PINNED_COMMIT: Commit hash to check the CLI (short or full hash).
    """
    if os.environ.get("CLI_SKIP_UPDATE_CHECK") == "1":  # Skip if user disabled checks
        return

    cli_pinned_commit = os.environ.get("CLI_PINNED_COMMIT", "").strip()
    if not cli_pinned_commit:
        return  # No pin requested; nothing to verify.

    commit_file = Path(__file__).with_name(".cli_commit_hash")
    try:
        local_hash = commit_file.read_text().strip()
    except OSError:
        local_hash = ""

    if _hashes_match(local_hash, cli_pinned_commit):
        return

    cmd_name = os.environ.get("HOLOHUB_CMD_NAME", "./holohub")
    print()
    print("════════════════════════════════════════════════════")
    print(f"⚠️  {cmd_name} CLI version does not match CLI_PINNED_COMMIT.")
    print(f"Current:  {local_hash[:8] if local_hash else '<unknown>'} ({commit_file})")
    print(f"Expected: {cli_pinned_commit[:8]}")
    print("To reinstall pinned CLI, run:")
    print(f"  CLI_FORCE_UPDATE=1 {cmd_name} --help")
    print("════════════════════════════════════════════════════")


if __name__ == "__main__":
    try:
        check_for_cli_updates()
    except Exception:
        pass  # Suppress all exceptions to ensure CLI always runs, even if version check fails.
