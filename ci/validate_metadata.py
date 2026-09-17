#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate metadata.json files against Holoscan CLI JSON schemas."""

import json
import os
import sys
from pathlib import Path

_EXCLUDE_DIRS = {".git", "install", ".local", ".cache", "_CPack_Packages"}


def _is_excluded_dir(name: str) -> bool:
    return name in _EXCLUDE_DIRS or name == "build" or name.startswith("build-")


def iter_metadata_files(repo_root: Path):
    """Yield repository metadata files outside generated directories."""
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [
            directory for directory in dirnames if not _is_excluded_dir(directory)
        ]
        if "metadata.json" in filenames:
            yield Path(dirpath) / "metadata.json"


def main():
    """Validate every metadata file, returning a nonzero status on failure."""
    try:
        from holoscan_cli.metadata.metadata_validator import validate_json
    except ImportError as import_error:
        print(
            f"error: {sys.executable} cannot import the Holoscan CLI "
            f"metadata validator: {import_error}",
            file=sys.stderr,
        )
        return 1

    repo_root = Path(__file__).resolve().parents[1]
    failed = False
    for path in sorted(iter_metadata_files(repo_root)):
        relative_path = path.relative_to(repo_root)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            print(f"{relative_path}: invalid JSON — {error}", file=sys.stderr)
            failed = True
            continue

        ok, message = validate_json(data, path.parent)
        if ok:
            print(f"{relative_path}: ok")
        else:
            print(f"{relative_path}: {message}", file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
