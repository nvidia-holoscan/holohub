#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Common utilities shared across website scripts."""

import logging
import re
import subprocess
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
COMPONENT_TYPES = ["workflows", "applications", "operators", "tutorials", "benchmarks"]
HOLOHUB_REPO_URL = "https://github.com/nvidia-holoscan/holohub"

# Ranking levels for documentation
RANKING_LEVELS = {
    0: "Level 0 - Core Stable",
    1: "Level 1 - Highly Reliable",
    2: "Level 2 - Trusted",
    3: "Level 3 - Developmental",
    4: "Level 4 - Experimental",
    5: "Level 5 - Obsolete",
}


def get_git_root() -> Path:
    """Get the absolute path to the Git repository root."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
    )
    return Path(result.stdout.strip())


def get_metadata_file_commit_date(metadata_path: Path, git_repo_path: Path) -> datetime:
    """Get the date of the first commit that introduced the metadata file."""
    try:
        rel_file_path = str(metadata_path.relative_to(git_repo_path))
        repo_path = str(git_repo_path)
        # Use --reverse to sort from oldest to newest, and take the first one
        cmd = [
            "git",
            "-C",
            repo_path,
            "log",
            "--follow",
            "--format=%at",
            "--reverse",
            rel_file_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        timestamps = result.stdout.strip().split("\n")

        if timestamps and timestamps[0]:
            # First line contains the oldest commit timestamp
            return datetime.fromtimestamp(int(timestamps[0]))
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.error(f"Error getting creation date for {metadata_path}: {e}")

    # Fallback to file creation time or modification time if creation time not available
    return datetime.fromtimestamp(metadata_path.stat().st_mtime)


def format_date(date_str: str) -> str:
    """Format a date string in YYYY-MM-DD format to Month DD, YYYY format."""
    try:
        year, month, day = date_str.split("-")
        months = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        return f"{months[int(month)-1]} {int(day)}, {year}"
    except (ValueError, IndexError):
        # Return the original string if we can't parse it
        return date_str


def get_last_modified_date(file_path: Path, git_repo_path: Path) -> str:
    """Get the last modified date of a file or directory using git or stat."""
    # Try using git to get the last modified date
    try:
        rel_file_path = str(file_path.relative_to(git_repo_path))
        repo_path = str(git_repo_path)
        cmd = ["git", "-C", repo_path, "log", "-1", "--format=%ad", "--date=short", rel_file_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        git_date = result.stdout.strip()

        if git_date:  # If we got a valid date from git
            return format_date(git_date)
    except (subprocess.CalledProcessError, ValueError):
        # Git command failed or path is not in repo, we'll fall back to stat
        pass

    # Second try: Filesystem stat date
    try:
        cmd = ["stat", "-c", "%y", str(file_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stat_date = result.stdout.split()[0].strip()  # Get just the date portion

        if stat_date:  # If we got a valid date from stat
            return format_date(stat_date)
    except (subprocess.CalledProcessError, ValueError, IndexError):
        logger.error(f"Failed to get modification date for {file_path}")

    # Fallback if both methods fail
    return "Unknown"


def get_file_from_git(file_path: Path, git_ref: str, git_repo_path: Path) -> str:
    """Get file content from a specific git revision."""
    try:
        rel_file_path = file_path.relative_to(git_repo_path)
        cmd = ["git", "-C", str(git_repo_path), "show", f"{git_ref}:{rel_file_path}"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except (subprocess.CalledProcessError, ValueError) as e:
        if isinstance(e, subprocess.CalledProcessError):
            logger.error(f"Git error: {e.stderr}")
        else:
            logger.error(f"Path {file_path} is not within the Git repository")
        raise e


def extract_image_from_readme(readme_content):
    """Extracts the first image from a README file."""
    if not readme_content:
        return None

    # Try HTML image tags
    html_pattern = r'<img\s+[^>]*src=["\'](.*?)["\'][^>]*>'
    html_match = re.search(html_pattern, readme_content, re.IGNORECASE)
    if html_match:
        return html_match.group(1).strip()

    # Try Markdown image syntax
    md_pattern = r"!\[[^\]]*\]\(([^)]+)\)"
    md_match = re.search(md_pattern, readme_content, re.IGNORECASE)
    if md_match:
        return md_match.group(1).strip()

    return None
