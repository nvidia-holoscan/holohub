# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small Git helpers used by repository CI checks."""

import os
import subprocess


def is_file_empty(filename):
    """Return whether a file has no content."""
    return os.stat(filename).st_size == 0


def _git(*arguments):
    return subprocess.check_output(
        ["git", *arguments],
        encoding="utf-8",
    ).rstrip("\n")


def uncommitted_files():
    """Return modified, added, and untracked paths in the worktree."""
    paths = []
    for line in _git("status", "--porcelain", "--untracked-files=all").splitlines():
        status = line[:2]
        if "M" in status or "A" in status or "?" in status:
            paths.append(line[3:])
    return paths


def changed_files_between(base_ref, new_ref):
    """Return paths changed between two Git references."""
    output = _git(
        "--no-pager",
        "diff",
        "--name-only",
        "--ignore-submodules",
        f"{base_ref}..{new_ref}",
    )
    return output.splitlines()


def modified_files(target=None, absolute_path=False):
    """Return paths changed from target, or current worktree changes."""
    paths = changed_files_between(target, "HEAD") if target else uncommitted_files()
    if absolute_path:
        repository = _git("rev-parse", "--show-toplevel")
        return [os.path.join(repository, path) for path in paths]
    return paths
