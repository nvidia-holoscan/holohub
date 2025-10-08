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

import json
import logging
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import markdown
import requests
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
COMPONENT_TYPES = ["workflows", "applications", "operators", "tutorials", "benchmarks"]
HOLOHUB_REPO_URL = "https://github.com/nvidia-holoscan/holohub"


def get_current_git_ref() -> str:
    """Get the current git branch, tag, or commit hash being built.

    Returns:
        A string representing the current git reference (branch, tag, or commit hash).
        Falls back to 'main' if detection fails.
    """
    # Check common CI environment variables
    if "GITHUB_REF_NAME" in os.environ:
        return os.environ["GITHUB_REF_NAME"]
    if "CI_COMMIT_REF_NAME" in os.environ:  # GitLab
        return os.environ["CI_COMMIT_REF_NAME"]
    try:
        # Get current branch, tag, or commit hash
        for cmd in [
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            ["git", "describe", "--tags", "--exact-match"],
            ["git", "rev-parse", "--short", "HEAD"],
        ]:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip() and result.stdout.strip() != "HEAD":
                return result.stdout.strip()
    except Exception as e:
        logger.warning(f"Failed to determine git reference: {e}")
        return "main"  # Default fallback


# Use a function to get the raw base URL dynamically
def get_raw_base_url(name="") -> str:
    """Get the base URL for raw GitHub content based on current git reference."""
    if not name:
        name = get_current_git_ref()
    return f"https://raw.githubusercontent.com/nvidia-holoscan/holohub/{name}/"


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
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
        )
        return Path(result.stdout.strip())
    except Exception as e:
        logger.error(f"Error getting Git root: {e}")
        return Path(".")


def parse_metadata_file(metadata_path: Path) -> dict:
    """Parse metadata.json file and extract relevant information."""
    with metadata_path.open("r") as f:
        data = json.load(f)
    component_type = list(data.keys())[0]
    metadata = data[component_type]
    return metadata, component_type


def get_metadata_file_commit_date(metadata_path: Path, git_repo_path: Path) -> datetime:
    """Get the creation date of a metadata.json file from git history.
    
    This function determines when an application/component was first created by finding
    the first commit that introduced its metadata.json file.
    
    Uses: git log --follow --format=%at --reverse <file>
    - --follow: Tracks file through renames
    - --format=%at: Returns Unix timestamp
    - --reverse: Oldest commits first (so first entry is the creation date)
    
    Args:
        metadata_path: Path to the metadata.json file
        git_repo_path: Path to the Git repository root
        
    Returns:
        datetime: The date when the metadata.json was first committed (application creation date)
    """
    rel_file_path = str(metadata_path.relative_to(git_repo_path))
    cmd = f"git -C {git_repo_path} log --follow --format=%at --reverse {rel_file_path}".split()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        timestamps = result.stdout.strip().split("\n")
        if timestamps and timestamps[0]:
            return datetime.fromtimestamp(int(timestamps[0]))
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.error(f"Error getting creation date for {metadata_path}: {e}")
    # Fallback to file modification time if git fails
    return datetime.fromtimestamp(metadata_path.stat().st_mtime)


def get_recent_source_code_update_date(metadata_path: Path, git_repo_path: Path):
    """Get the most recent update date for source code files in a component directory.
    
    This function checks for recent modifications to source code files (.py, .cpp, .h, .hpp, .cu, .cuh)
    in the component directory to determine if the component has been recently updated.
    
    Args:
        metadata_path: Path to the metadata.json file
        git_repo_path: Path to the Git repository root
        
    Returns:
        datetime: The date of the most recent source code update, or None if no updates found
    """
    # Get the component directory (parent of metadata.json, or parent's parent if in cpp/python subdirs)
    component_dir = metadata_path.parent
    if component_dir.name in ["cpp", "python"]:
        component_dir = component_dir.parent
    
    # Source code file extensions to check
    source_extensions = ["*.py", "*.cpp", "*.h", "*.hpp", "*.cu", "*.cuh", "*.c", "*.cc", "*.cxx"]
    
    rel_component_dir = str(component_dir.relative_to(git_repo_path))
    
    # Build git command to find the most recent commit affecting source files
    # Format: git log -1 --format=%at -- <pattern1> <pattern2> ...
    patterns = [f"{rel_component_dir}/**/{ext}" for ext in source_extensions]
    
    # Use git log to find the most recent commit that modified source files
    cmd = ["git", "-C", str(git_repo_path), "log", "-1", "--format=%at", "--"]
    cmd.extend(patterns)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        timestamp_str = result.stdout.strip()
        if timestamp_str:
            return datetime.fromtimestamp(int(timestamp_str))
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.debug(f"No recent source code updates found for {component_dir.name}: {e}")
    
    return None


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
    rel_file_path = str(file_path.relative_to(git_repo_path))
    cmd = f"git -C {git_repo_path} log -1 --format=%ad --date=short {rel_file_path}".split()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        git_date = result.stdout.strip()
        if git_date:  # If we got a valid date from git
            return format_date(git_date)
    except (subprocess.CalledProcessError, ValueError):
        pass

    # Second try: Filesystem stat date
    cmd = ["stat", "-c", "%y", str(file_path)]
    try:
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


def get_readme_content(component_type, component_name, href=None):
    """Reads the README.md content from the local filesystem, handling various path variations.

    Args:
        component_type: Type of the component (applications, operators, etc.)
        component_name: Name of the component
        href: Optional full href path from the app card which may provide a more precise location

    Returns:
        Tuple of (readme_content, readme_path) or (None, None) if not found
    """
    try:
        # Get the Git repository root
        git_repo_path = get_git_root()

        # Initialize the list of possible README paths
        readme_paths = []

        # If href is provided, use it as the primary path
        if href and "/" in href:
            href_path = href.strip("/")
            readme_paths.append(git_repo_path / href_path / "README.md")
            href_parent = Path(href_path).parent
            if str(href_parent) != ".":  # Only if parent is not root
                readme_paths.append(git_repo_path / href_parent / "README.md")

        readme_paths.extend(
            [
                git_repo_path / component_type / component_name / "README.md",
                git_repo_path / component_type / component_name / "python" / "README.md",
                git_repo_path / component_type / component_name / "cpp" / "README.md",
            ]
        )

        # Try each possible path
        for path in readme_paths:
            if path.exists():
                logger.info(f"Found README at {path}")
                with open(path, "r", encoding="utf-8") as f:
                    return f.read(), path

        # No README found
        logger.warning(f"No README found for {component_type}/{component_name} (href: {href})")
        return None, None

    except Exception as e:
        logger.warning(f"Error reading README for {component_type}/{component_name}: {e}")
        return None, None


def get_full_image_url(relative_path, readme_path=None):
    """Converts a relative image path to a full GitHub URL using the README path for context.

    Args:
        relative_path: The relative path to the image from the README
        readme_path: Path object pointing to the README file that referenced the image
    """
    if relative_path.startswith("./"):
        relative_path = relative_path[2:]
    if relative_path.startswith(("http://", "https://")):
        return relative_path
    if readme_path:
        readme_dir = readme_path.parent
        image_path = (readme_dir / relative_path).resolve()
        git_root = get_git_root()
        rel_image_path = image_path.relative_to(git_root)

        url = urljoin(get_raw_base_url(), str(rel_image_path))
        try:
            response = requests.head(url, timeout=5)
            if response.status_code == 200:
                return str(url)
            else:
                logger.warning(f"URL {url} returned status code {response.status_code}")
        except requests.RequestException as e:
            logger.warning(f"Error checking URL {url}: {e}")
        return str(urljoin(get_raw_base_url("main"), str(rel_image_path)))

    logger.warning(f"Using direct URL without context: {relative_path}")
    return relative_path


def extract_first_sentences(readme_text, num_sentences=3, max_chars=160):
    """Extract the first few meaningful sentences from README markdown content.

    Args:
        readme_text: The raw markdown content of the README
        num_sentences: Number of sentences to extract
        max_chars: Maximum character length before truncation

    Returns:
        A string with the first few sentences, truncated if necessary
    """
    html = markdown.markdown(readme_text, extensions=["markdown.extensions.fenced_code"])
    soup = BeautifulSoup(html, "html.parser")
    # Remove code blocks
    for code_block in soup.find_all(["pre", "code"]):
        code_block.decompose()
    # Get all paragraphs
    pghs = soup.find_all("p")
    # Skip first paragraph if it's very short (likely badges)
    start_idx = 1 if pghs and len(pghs) > 1 and len(pghs[0].get_text().strip()) < 40 else 0
    text_content = ""
    for p in pghs[start_idx:]:
        p_text = p.get_text().strip()
        if len(p_text) > 15 and not p_text.startswith(("$", "http")):
            text_content += p_text + " "
            if len(text_content) > max_chars * 2:
                break
    # Use nltk if available for better sentence tokenization
    try:
        import nltk.tokenize

        nltk.download("punkt_tab")
        sentences = nltk.tokenize.sent_tokenize(text_content)
    except (ImportError, AttributeError):
        # Fallback to regex-based approach
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text_content)
    result = " ".join(sentences[:num_sentences]).strip()

    result = re.sub(r"\s+", " ", result)
    if len(result) > max_chars:
        result = result[: max_chars - 3].rstrip() + "..."

    return result
