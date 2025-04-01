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
"""Generate the code reference pages and copy Jupyter notebooks and README files."""

import json
import logging
import re
import subprocess
import traceback
from pathlib import Path

import mkdocs_gen_files

# log stuff
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set the git repository path for Docker environment
GIT_REPO_PATH = "/holohub"

COMPONENT_TYPES = ["workflows", "applications", "operators", "tutorials", "benchmarks"]


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


def get_last_modified_date(file_path: Path) -> str:
    """Get the last modified date of a file or directory using git or stat."""
    rel_file_path = str(file_path.relative_to(GIT_REPO_PATH))

    # First try: Git date
    try:
        cmd = ["git", "-C", GIT_REPO_PATH, "log", "-1", "--format=%ad", "--date=short", rel_file_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        git_date = result.stdout.strip()

        if git_date:  # If we got a valid date from git
            return format_date(git_date)
    except subprocess.CalledProcessError:
        # Git command failed, we'll fall back to stat
        pass

    # Second try: Filesystem stat date
    try:
        cmd = ["stat", "-c", "%y", file_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stat_date = result.stdout.split()[0].strip()  # Get just the date portion

        if stat_date:  # If we got a valid date from stat
            return format_date(stat_date)
    except (subprocess.CalledProcessError, ValueError, IndexError):
        logger.error(f"Failed to get modification date for {rel_file_path}")

    # Fallback if both methods fail
    return "Unknown"


def create_metadata_header(metadata: dict, last_modified: str) -> str:
    """Create the metadata header for the documentation page."""
    authors = metadata["authors"]
    platforms = metadata["platforms"]
    language = metadata["language"] if "language" in metadata else None
    version = metadata["version"]
    min_sdk_version = metadata["holoscan_sdk"]["minimum_required_version"]
    tested_sdk_versions = metadata["holoscan_sdk"]["tested_versions"]
    metric = metadata["ranking"]

    metric_str = "Level 0 - Core Stable"
    if metric == 1:
        metric_str = "Level 1 - Highly Reliable"
    if metric == 2:
        metric_str = "Level 2 - Trusted"
    if metric == 3:
        metric_str = "Level 3 - Developmental"
    if metric == 4:
        metric_str = "Level 4 - Experimental"
    if metric == 5:
        metric_str = "Level 5 - Obsolete"

    header_text = (
        ":octicons-person-24: **Authors:** "
        + ", ".join(f"{author['name']} ({author['affiliation']})" for author in authors)
        + "<br>"
    )

    platforms_str = ", ".join(platforms)
    header_text += f":octicons-device-desktop-24: **Supported platforms:** {platforms_str}<br>"

    if language:
        header_text += f":octicons-code-square-24: **Language:** {language}<br>"

    header_text += f":octicons-clock-24: **Last modified:** {last_modified}<br>"

    # Remove archive-specific handling
    header_text += f":octicons-tag-24: **Latest version:** {version}<br>"

    header_text += f":octicons-stack-24: **Minimum Holoscan SDK version:** {min_sdk_version}<br>"

    tested_vers_str = ", ".join(tested_sdk_versions)
    header_text += f":octicons-beaker-24: **Tested Holoscan SDK versions:** {tested_vers_str}<br>"

    header_text += f":octicons-sparkle-fill-24: **Contribution metric:** {metric_str}<br><br>"

    return header_text


def create_page(
    metadata: dict,
    readme_text: str,
    dest_path: Path,
    last_modified: str,
):
    """Create a documentation page.

    Args:
        metadata: The metadata dictionary
        readme_text: Content of the README file
        dest_path: relative path to the documentation page
        last_modified: Last modified date string
    Returns:
        Generated page content as string
    """

    # Title
    title = metadata["name"]

    # Tags
    tags = metadata["tags"]

    # Frontmatter
    output_text = "---"
    output_text += f'\ntitle: "{title}"'
    output_text += "\ntags:"
    for tag in tags:
        output_text += f"\n - {tag}"
    output_text += "\n---\n"

    # Process README content to fix image paths
    dest_dir = dest_path.parent
    base_url = f"https://github.com/nvidia-holoscan/holohub/blob/main/{str(dest_dir)}"
    # Regular expression pattern to match paths containing .gif, .png, or .jpg
    pattern = r'["(\[][^:")]*\.(?:gif|png|jpg)[")\]]'
    matches = re.findall(pattern, readme_text)
    for match in matches:
        # Find the URL inside [](image)
        parenthensis_match = re.search(r"\((.*?)\)", match)
        if parenthensis_match:
            match = parenthensis_match.group(1)

        match = match.strip('"()[]')
        imgmatch = match
        if match.startswith("."):
            imgmatch = match[1:]
        if imgmatch.startswith("/"):
            imgmatch = imgmatch[1:]
        readme_text = readme_text.replace(match, f"{base_url}/{imgmatch}?raw=true")

    # Get the header metadata
    header_text = create_metadata_header(metadata, last_modified)

    # Find the first header
    pattern = r"^#\s+(.+)"
    match = re.match(pattern, readme_text)

    # Add URL to the title, and list metadata after it
    if match:
        header_title = match.group(1)
        header_title = f"[{header_title}]({base_url})"
        output_text += readme_text.replace(match.group(1), f"{header_title}\n{header_text}", 1)
    else:
        logger.warning(f"No header found in {dest_path}, can't insert metadata header")
        output_text += readme_text

    with mkdocs_gen_files.open(dest_path, "w") as dest_file:
        dest_file.write(output_text)


def parse_metadata_path(metadata_path: Path, components) -> None:
    """Copy README file from a sub-package to the user guide's developer guide directory.

    Args:
        metadata_path: Path to the metadata file
        components: Dictionary tracking unique components

    Returns:
        None
    """
    # Disable application with {{ in the name
    if "{{" in str(metadata_path):
        return

    # Parse the metadata
    with metadata_path.open("r") as metadata_file:
        metadata = json.load(metadata_file)

    project_type = list(metadata.keys())[0]
    metadata = metadata[project_type]

    # Check valid component type
    component_type = f"{project_type}s"
    if component_type not in COMPONENT_TYPES:
        logger.error(f"Skipping {metadata_path}: unknown type '{component_type}'")
        return

    # Dirs & Paths
    metadata_dir = metadata_path.parent
    readme_dir = metadata_dir
    readme_path = readme_dir / "README.md"
    while not readme_path.exists():
        readme_dir = readme_dir.parent
        readme_path = readme_dir / "README.md"
    dest_dir = readme_dir.relative_to(GIT_REPO_PATH)

    # Valid README is not in the component type directory or the git repo root
    if readme_dir.name in COMPONENT_TYPES or readme_dir == git_repo_path:
        logger.error(f"Skipping {metadata_path}: no README found")
        return

    # Track components if in adequate directories
    # Ex: don't track operators under application folders
    if component_type == dest_dir.parts[0]:
        components[component_type].add(dest_dir)
    logger.info(f"Processing: {dest_dir}")
    logger.debug(f"  for metadata_path: {metadata_path}")

    # Prepare suffix with language info if it's a language-specific component
    suffix = ""
    if metadata_dir.name in ["cpp", "python"] and readme_dir == metadata_dir:
        language_agnostic_dir = metadata_dir.parent
        nbr_language_dirs = len(list(language_agnostic_dir.glob("*/metadata.json")))
        if nbr_language_dirs > 1:
            suffix = "C++" if metadata_dir.name == "cpp" else f"{metadata_dir.name.capitalize()}"
            suffix = f" ({suffix})"
    logger.debug(f"suffix: {suffix}")
    title = metadata["name"] if "name" in metadata else metadata_path.name
    title += suffix

    # Process the README content
    readme_text = f"# {title}\n\nNo README available."
    if readme_path.exists():
        with readme_path.open("r") as readme_file:
            readme_text = readme_file.read()

    # Create page content
    dest_file = dest_dir / "README.md"
    create_page(
        metadata, readme_text, dest_file, get_last_modified_date(metadata_path)
    )

    # Add a .nav.yml file to control navigation with mkdocs-awesome-nav
    nav_path = dest_dir / ".nav.yml"
    nav_content = f"title: \"{title}\""

    with mkdocs_gen_files.open(nav_path, "w") as nav_file:
        nav_file.write(nav_content)


def generate_pages() -> None:
    """Generate pages for documentation.

    This function orchestrates the entire process of generating API references,
    copying README files for workflow, applications and operators.

    Returns:
        None
    """

    # Dirs
    src_dir = Path(GIT_REPO_PATH)
    website_src_dir = Path(__file__).parent.parent

    # Initialize map of projects/component per type
    components = {key: set() for key in COMPONENT_TYPES}

    for component_type in COMPONENT_TYPES:
        component_dir = src_dir / component_type
        if not component_dir.exists():
            logger.error(f"Component directory not found: {component_dir}")
            continue

        # Parse the metadata.json files
        for metadata_path in component_dir.rglob("metadata.json"):
            try:
                parse_metadata_path(metadata_path, components)
            except Exception as e:
                logger.error(f"Failed to process {metadata_path}:\n{traceback.format_exc()}")

        # Write navigation file to sort components by title
        nav_path = Path(component_type) / ".nav.yml"
        nav_content = """
sort:
  by: title
"""
        with mkdocs_gen_files.open(nav_path, "w") as nav_file:
            nav_file.write(nav_content)

    logger.debug(f"Components: {components}")

    # Write the home page
    homefile_path = website_src_dir / "docs" / "index.md"
    with homefile_path.open("r") as home_file:
        home_text = home_file.read()

        # Replace the number of components in the home page
        for component_type in COMPONENT_TYPES:
            nbr_components = len(components[component_type])
            home_text = home_text.replace(f"#{component_type}", str(nbr_components))

    with mkdocs_gen_files.open("index.md", "w") as index_file:
        index_file.write(home_text)

    # Write explicit navigation order for the root
    nav_content = """
nav:
- index.md
- workflows
- applications
- operators
- tutorials
- benchmarks
"""
    with mkdocs_gen_files.open(".nav.yml", "w") as nav_file:
        nav_file.write(nav_content)


if __name__ in {"__main__", "<run_path>"}:
    # Check if name is either '__main__', or the equivalent default in `runpy.run_path(...)`, which is '<run_path>'
    generate_pages()
