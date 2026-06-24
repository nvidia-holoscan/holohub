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
"""Validate metadata.json files against holoscan-cli JSON schemas.

Walks the repository, loads every metadata.json that isn't under an
excluded directory, and validates it using holoscan_cli's schema
registry (operator, application, module, package, …).  The correct
schema is chosen automatically from the top-level envelope key.
"""

import json
import os
import sys
from pathlib import Path

_EXCLUDE_DIRS = {"build", ".local", ".cache", "_CPack_Packages"}


def iter_metadata_files(repo_root: Path):
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in _EXCLUDE_DIRS]
        if "metadata.json" in filenames:
            yield Path(dirpath) / "metadata.json"


def main():
    try:
        from holoscan_cli.metadata.metadata_validator import validate_json
    except ImportError:
        print(
            "error: holoscan-cli is not installed in the active Python environment.\n"
            "       Activate the environment that has holoscan-cli before running "
            "pre-commit.",
            file=sys.stderr,
        )
        sys.exit(1)

    repo_root = Path(__file__).resolve().parents[3]
    failed = False

    for path in sorted(iter_metadata_files(repo_root)):
        rel = path.relative_to(repo_root)
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            print(f"{rel}: invalid JSON — {exc}", file=sys.stderr)
            failed = True
            continue

        ok, message = validate_json(data, path.parent)
        if not ok:
            print(f"{rel}: {message}", file=sys.stderr)
            failed = True
        else:
            print(f"{rel}: ok")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
