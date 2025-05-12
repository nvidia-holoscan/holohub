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
from typing import Dict, List, Tuple
from urllib.parse import urljoin

import markdown
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

# Category title mapping for proper capitalization
CATEGORY_TITLE_MAPPING = {
    "healthcare ai": "Healthcare AI",
    "computer vision and perception": "Computer Vision",
    "natural language and conversational ai": "NLP & Conversational",
    "networking and distributed computing": "Networking",
    "signal processing": "Signal Processing",
    "tools and other specialized applications": "Specialized Tools",
    "extended reality": "Extended Reality",
    "visualization": "Visualization",
}

CATEGORY_ICONS = {
    "Healthcare AI": "medical_services",
    "Computer Vision": "visibility",
    "NLP & Conversational": "chat",
    "Networking": "hub",
    "Signal Processing": "radar",
    "Specialized Tools": "tune",
    "Extended Reality": "view_in_ar",
    "visualization": "auto_awesome_motion",
}


def format_category_title(title):
    """Format a category title with proper capitalization."""
    lower_title = title.lower()
    if lower_title in CATEGORY_TITLE_MAPPING:
        return CATEGORY_TITLE_MAPPING[lower_title]
    return title.title()


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
    try:
        with metadata_path.open("r") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error parsing metadata file: {metadata_path}")
        raise e
    component_type = list(data.keys())[0]
    metadata = data[component_type]
    return metadata, component_type


def get_metadata_file_commit_date(metadata_path: Path, git_repo_path: Path) -> datetime:
    """Get the date of the first commit that introduced the metadata file."""
    rel_file_path = str(metadata_path.relative_to(git_repo_path))
    cmd = f"git -C {git_repo_path} log --follow --format=%at --reverse {rel_file_path}".split()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        timestamps = result.stdout.strip().split("\n")
        if timestamps and timestamps[0]:
            return datetime.fromtimestamp(int(timestamps[0]))
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.error(f"Error getting creation date for {metadata_path}: {e}")
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

        # url = urljoin(get_raw_base_url(), str(rel_image_path))
        # try:
        #     response = requests.head(url, timeout=5)
        #     if response.status_code == 200:
        #         return str(url)
        #     else:
        #         logger.warning(f"URL {url} returned status code {response.status_code}")
        # except requests.RequestException as e:
        #     logger.warning(f"Error checking URL {url}: {e}")
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
        import nltk

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
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


def find_readme_path(app_dir, git_repo_path):
    """Find the README.md file for an application."""
    readme_paths = [
        app_dir / "README.md",
        app_dir / "python" / "README.md",
        app_dir / "cpp" / "README.md",
    ]
    # Also check parent directories up to a limit
    parent_dir = app_dir
    for _ in range(3):  # Maximum depth to search up
        if parent_dir.name == "applications" or parent_dir == git_repo_path:
            break
        parent_dir = parent_dir.parent
        readme_paths.append(parent_dir / "README.md")

    for path in readme_paths:
        if path.exists():
            return path
    return None


def find_app_pairs(
    git_repo_path: Path, component_types: List[str] = ["applications"]
) -> Dict[str, Tuple[Path, Path]]:
    """
    Find valid application pairs with both metadata.json and README.md files.
    When a metadata or README is missing, try to find its most similar sister.

    Args:
        git_repo_path: Path to the git repository root
        component_types: List of component directories to search (default: ["applications"])

    Returns:
        Dict mapping app_id to tuple of (metadata_path, readme_path)
    """
    # Find all metadata.json files
    metadata_files = {}
    readme_files = {}

    for component_type in component_types:
        component_dir = git_repo_path / component_type
        if not component_dir.exists():
            logger.warning(f"Component directory not found: {component_dir}")
            continue

        # Collect all metadata.json files
        for metadata_path in component_dir.rglob("metadata.json"):
            # Skip specific excluded paths
            keywords = ["data_writer", "operator", "xr_hello_holoscan", "template", "{{"]
            if any(t in str(metadata_path) for t in keywords):
                continue

            # Get app identifier from path
            app_id = get_app_id_from_path(metadata_path, git_repo_path)
            metadata_files[app_id] = metadata_path

        # Collect all README.md files
        for readme_path in component_dir.rglob("README.md"):
            # Skip specific excluded paths
            keywords = ["data_writer", "operator", "xr_hello_holoscan", "template", "{{"]
            if any(t in str(readme_path) for t in keywords):
                continue

            # Get app identifier from path
            app_id = get_app_id_from_path(readme_path, git_repo_path)
            readme_files[app_id] = readme_path

    # Match metadata with readmes
    app_pairs = {}

    # First, handle direct matches (both metadata and readme exist with same app_id)
    exact_matches = set(metadata_files.keys()).intersection(set(readme_files.keys()))
    for app_id in exact_matches:
        app_pairs[app_id] = (metadata_files[app_id], readme_files[app_id])
    logger.info(f"Exact matches (len: {len(exact_matches)}): {exact_matches}")

    # Handle orphaned metadata files (no matching README)
    orphaned_metadata = set(metadata_files.keys()) - exact_matches
    for app_id in orphaned_metadata:
        # Find the closest README file
        closest_readme = find_closest_file(app_id, readme_files)
        if closest_readme:
            # Create a new entry with the metadata and the closest README
            app_pairs[app_id] = (metadata_files[app_id], readme_files[closest_readme])
            logger.info(f"Orphaned metadata paired: {app_id} with README from {closest_readme}")

    return app_pairs


def get_app_id_from_path(file_path: Path, git_repo_path: Path) -> str:
    """
    Generate a unique app identifier from a file path.
    Handles common path patterns for applications.

    Args:
        file_path: Path to the metadata.json or README.md file
        git_repo_path: Path to the git repository root

    Returns:
        String identifier for the application
    """
    rel_path = file_path.relative_to(git_repo_path)
    parts = list(rel_path.parts)

    # Remove the filename
    parts.pop()

    # Generate an app_id that represents the directory structure
    return "/".join(parts)


def find_closest_file(app_id: str, available_files: Dict[str, Path]) -> str:
    """
    Find the closest matching file from the available files.
    Uses path similarity to determine the closest match.

    Args:
        app_id: The app ID to find a match for
        available_files: Dictionary of app_id -> file_path

    Returns:
        Closest matching app_id or None if no match found
    """
    if not available_files:
        return None

    # Split the app_id into parts
    parts = app_id.split("/")

    # Try different matching strategies in order of preference

    # 1. Direct parent directory
    if len(parts) > 1:
        parent_id = "/".join(parts[:-1])
        if parent_id in available_files:
            return parent_id

    # 2. Same parent, different variant (python/cpp)
    if len(parts) > 1:
        parent = parts[:-1]
        siblings = [aid for aid in available_files if aid.startswith("/".join(parent) + "/")]

        if siblings:
            # Return the first sibling (could be improved to find the most similar)
            return siblings[0]

    # 3. Most common path components
    best_match = None
    best_score = -1

    for other_id in available_files:
        other_parts = other_id.split("/")
        # Count common parts from the beginning
        common = 0
        for i in range(min(len(parts), len(other_parts))):
            if parts[i] == other_parts[i]:
                common += 1
            else:
                break

        # Score based on common parts and total length difference
        score = common - 0.1 * abs(len(parts) - len(other_parts))
        if score > best_score:
            best_score = score
            best_match = other_id

    return best_match
