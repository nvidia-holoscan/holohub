#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Generate the featured apps HTML content based on the most recent metadata.json files."""

import json
import os
import sys
from pathlib import Path

# Add the script directory to the path to enable importing common_utils
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(script_dir)) if str(script_dir) not in sys.path else None

# Import after adding script_dir to path
from common_utils import (  # noqa: E402
    COMPONENT_TYPES,
    get_git_root,
    get_metadata_file_commit_date,
    logger,
)
from extract_readme_images import update_featured_apps_html  # noqa: E402

# Constants
OUTPUT_FILE = "docs/_data/featured-apps.html"


def find_most_recent_metadata_files(git_repo_path: Path, count: int = 3) -> list:
    """Find the most recently created/updated metadata.json files.

    Args:
        git_repo_path: Path to the Git repository root
        count: Number of recent metadata files to retrieve

    Returns:
        List of tuples (metadata_path, datetime) for the most recent unique applications
    """
    # Dictionary to track the most recent version of each unique application
    unique_applications = {}

    # Search for metadata.json files in component directories
    for component_type in ["applications"]:
        component_dir = git_repo_path / component_type
        if not component_dir.exists():
            continue

        for metadata_path in component_dir.rglob("metadata.json"):
            # Skip certain components
            if any(
                t in str(metadata_path) for t in ["datawriter", "operators", "xr_hello_holoscan"]
            ):
                continue

            # Determine the unique application name
            app_dir = metadata_path.parent
            # If we're in a language subdirectory (cpp/python), use the parent directory name
            if app_dir.name in ["cpp", "python"]:
                app_dir = app_dir.parent
            app_name = app_dir.name

            # Get the commit date from git
            commit_date = get_metadata_file_commit_date(metadata_path, git_repo_path)

            # Only keep the most recent version of each application
            if (
                app_name not in unique_applications
                or commit_date > unique_applications[app_name][1]
            ):
                unique_applications[app_name] = (metadata_path, commit_date)

    # Convert dictionary to sorted list of the most recent unique applications
    result = sorted(unique_applications.values(), key=lambda x: x[1], reverse=True)
    return result[:count]


def parse_metadata_file(metadata_path: Path) -> dict:
    """Parse metadata.json file and extract relevant information."""
    with metadata_path.open("r") as f:
        data = json.load(f)
    component_type = list(data.keys())[0]
    metadata = data[component_type]
    return metadata


def find_readme_path(metadata_dir: Path, git_repo_path: Path) -> Path:
    """Find the README.md file associated with the component."""
    # First check in the same directory as the metadata file
    readme_path = metadata_dir / "README.md"
    if readme_path.exists():
        return readme_path
    # If not found, search up the directory tree until we reach a component type directory
    parent_dir = metadata_dir
    while parent_dir.name not in COMPONENT_TYPES and parent_dir != git_repo_path.parent:
        readme_path = parent_dir / "README.md"
        if readme_path.exists():
            return readme_path
        parent_dir = parent_dir.parent
    return None


def get_component_path(metadata_path: Path, git_repo_path: Path) -> str:
    """Generate the relative path for the component documentation link."""
    rel_path = metadata_path.parent.relative_to(git_repo_path)

    if rel_path.name in ["cpp", "python"]:
        rel_path = rel_path.parent
    return f"{rel_path}"


def generate_featured_app_card(metadata_path: Path, git_repo_path: Path) -> str:
    """Generate HTML for a featured app card."""
    try:
        metadata = parse_metadata_file(metadata_path)
        name = metadata.get("name", metadata_path.parent.name)
        description = metadata.get("description", "")
        logger.info(f"Generating featured app card for {name}")

        # If no description in metadata, look for first paragraph in README
        if not description:
            readme_path = find_readme_path(metadata_path.parent, git_repo_path)
            if readme_path and readme_path.exists():
                with readme_path.open("r") as f:
                    readme_text = f.read()
                    # Skip header and look for first paragraph
                    content_parts = readme_text.split("\n\n")
                    for part in content_parts[1:]:
                        if part.strip() and not part.startswith(("#", ":", ">")):
                            # Clean up and truncate description
                            description = part.replace("\n", " ").strip()
                            if len(description) > 160:
                                description = description[:157] + "..."
                            break

        component_path = get_component_path(metadata_path, git_repo_path)
        component_type = metadata_path.parent.relative_to(git_repo_path).parts[0]
        app_name = component_path.split("/")[-1]

        # Generate card HTML - placeholder image will be updated by extract_readme_images.py
        card_html = f"""
    <a href="{component_path}" class="featured-app-card">
      <h3>
        {name}
      </h3>
      <div class="app-thumbnail">
        <img data-src="https://github.com/nvidia-holoscan/holohub/blob/main/doc/website/docs/assets/images/{component_type}_default.png?raw=true" alt="{name}" data-app-name="{app_name}" data-component-type="{component_type}" />
      </div>
      <div class="app-description">
        {description}
      </div>
    </a>"""

        return card_html

    except Exception as e:
        logger.error(f"Error generating card for {metadata_path}: {e}")
        return ""


def generate_featured_apps_html(n: int = 3):
    """Generate the featured apps HTML content."""
    try:
        git_repo_path = get_git_root()

        recent_metadata_files = find_most_recent_metadata_files(git_repo_path, n)
        if not recent_metadata_files:
            logger.warning("No metadata files found to feature")
            return
        cards = []
        for metadata_path, _ in recent_metadata_files:
            card_html = generate_featured_app_card(metadata_path, git_repo_path)
            cards.append(card_html)

        cards_html = ''.join(cards)
        # Create the featured apps container HTML
        featured_apps_html = f"""<div class="featured-apps">
  <h2>Featured Components</h2>
  <div class="featured-apps-container" id="featured-apps-container">{cards_html}
  </div>
</div>
"""

        with open(OUTPUT_FILE, "w") as f:
            f.write(featured_apps_html)
        logger.info(f"Generated featured apps HTML with {len(recent_metadata_files)} components")
    except Exception as e:
        logger.error(f"Error generating featured apps HTML: {e}")


if __name__ in {"__main__", "<run_path>"}:
    generate_featured_apps_html(3)
    update_featured_apps_html()
