#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check copyright headers in repository source files."""

import argparse
import datetime
import re
import sys
from pathlib import Path

import gitutils

_FILE_PATTERNS = (
    re.compile(r"\.(cmake|cpp|cu|cuh|h|hpp|sh|pxd|py|pyx|yaml)$"),
    re.compile(r"(^|/)CMakeLists(_standalone)?\.txt$"),
    re.compile(r"(^|/)Dockerfile$"),
    re.compile(r"\.dockerfile$"),
    re.compile(r"(^|/)setup\.cfg$"),
    re.compile(r"(^|/)\.flake8\.cython$"),
    re.compile(r"(^|/)meta\.yaml$"),
)
_COPYRIGHT_PATTERNS = (
    re.compile(
        r"SPDX-FileCopyrightText:\s*Copyright(?: \(c\))?\s*"
        r"(\d{4})(?:-(\d{4}))?",
        re.IGNORECASE,
    ),
    re.compile(
        r"SPDX-FileCopyrightText:\s*(\d{4})(?:-(\d{4}))?",
        re.IGNORECASE,
    ),
    re.compile(
        r"Copyright(?: \(c\))?\s*(\d{4})(?:-(\d{4}))?",
        re.IGNORECASE,
    ),
)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="*")
    parser.add_argument(
        "--git-modified-only",
        nargs="?",
        const="",
        metavar="BASE_REF",
    )
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--exclude-config")
    parser.add_argument("--ignore-year-mismatch", action="store_true")
    parser.add_argument("--update-current-year", action="store_true")
    return parser.parse_args()


def _exclude_patterns(arguments):
    patterns = list(arguments.exclude)
    if arguments.exclude_config:
        config_path = Path(arguments.exclude_config)
        for line in config_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
    return [re.compile(pattern) for pattern in patterns]


def _candidate_files(arguments):
    if arguments.git_modified_only is not None:
        target = arguments.git_modified_only or None
        paths = [Path(path) for path in gitutils.modified_files(target)]
    else:
        paths = []
        for directory in arguments.directories or ["."]:
            paths.extend(path for path in Path(directory).rglob("*") if path.is_file())
    return paths


def _is_checked_file(path, excludes):
    path_string = path.as_posix()
    return (
        path.is_file()
        and not gitutils.is_file_empty(path)
        and any(pattern.search(path_string) for pattern in _FILE_PATTERNS)
        and not any(pattern.search(path_string) for pattern in excludes)
    )


def _copyright_match(content):
    for pattern in _COPYRIGHT_PATTERNS:
        match = pattern.search(content)
        if match:
            return match
    return None


def _update_year(path, content, match, current_year):
    start_year = int(match.group(1))
    end_year = int(match.group(2) or start_year)
    if current_year <= end_year:
        return False
    year_range = f"{start_year}-{current_year}"
    updated = (
        content[: match.start(1)]
        + year_range
        + content[match.end(2) if match.group(2) else match.end(1) :]
    )
    path.write_text(updated, encoding="utf-8")
    return True


def main():
    arguments = _arguments()
    excludes = _exclude_patterns(arguments)
    current_year = datetime.datetime.now().year
    errors = []
    checked = 0

    for path in sorted(set(_candidate_files(arguments))):
        if not _is_checked_file(path, excludes):
            continue
        checked += 1
        content = path.read_text(encoding="utf-8")
        match = _copyright_match(content)
        if not match:
            errors.append(f"{path}: copyright header missing or malformed")
            continue

        start_year = int(match.group(1))
        end_year = int(match.group(2) or start_year)
        if start_year > end_year:
            errors.append(f"{path}: copyright year range is reversed")
        elif (
            not arguments.ignore_year_mismatch
            and not start_year <= current_year <= end_year
        ):
            if arguments.update_current_year and _update_year(
                path,
                content,
                match,
                current_year,
            ):
                print(f"{path}: updated copyright year to {current_year}")
            else:
                errors.append(f"{path}: copyright does not include {current_year}")

    print(f"Checked copyright headers in {checked} files")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Copyright check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
