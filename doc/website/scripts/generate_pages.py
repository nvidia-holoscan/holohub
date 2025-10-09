# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os.path
import re
import subprocess
import sys
import traceback
import urllib.parse
from pathlib import Path

import mkdocs_gen_files

# log stuff
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

COMPONENT_TYPES = ["workflows", "applications", "operators", "tutorials", "benchmarks"]

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


def create_frontmatter(metadata: dict, archive_version: str = None) -> str:
    """Create the frontmatter for the documentation page."""

    # Title
    title = metadata["name"]
    title += f" ({archive_version})" if archive_version else " (latest)"

    # Tags
    tags = metadata["tags"]
    tags_str = ""
    for tag in tags:
        tags_str += f"\n - {tag}"

    return f"""---
title: "{title}"
tags:{tags_str}
---
"""


def create_metadata_header(metadata: dict, last_modified: str, archive_version: str = None, version_selector_html: str = "") -> str:
    """Create the metadata header for the documentation page.

    This function generates a formatted metadata header with icons and labels for display
    on the documentation page. It includes information such as authors, platforms, language,
    version information, and contribution metrics.

    Args:
        metadata (dict): Dictionary containing the application metadata
        last_modified (str): String representing the last modification date
        archive_version (str, optional): Version string for archived documentation. Default: None.
        version_selector_html (str, optional): HTML for version selector dropdown. Default: "".

    Returns:
        str: Formatted HTML-like string containing the metadata header with icons and labels
    """

    # Safely extract metadata to handle missing keys
    authors_str = None
    if authors := metadata.get("authors"):
        authors_str = ", ".join(
            [f'{author.get("name", "")} ({author.get("affiliation", "")})' for author in authors]
        )
    platforms = metadata.get("platforms")
    platforms_str = ", ".join(platforms) if platforms else None
    language = metadata.get("language")
    if isinstance(language, list):
        language = ", ".join(language)
    version = metadata.get("version")
    hsdk_meta = metadata.get("holoscan_sdk")
    min_sdk_version = hsdk_meta.get("minimum_required_version") if hsdk_meta else None
    tested_sdk_versions = hsdk_meta.get("tested_versions") if hsdk_meta else None
    tested_sdk_versions_str = ", ".join(tested_sdk_versions) if tested_sdk_versions else None
    ranking = metadata.get("ranking")
    ranking_str = RANKING_LEVELS.get(ranking)

    # List inputs for creating metadata header lines
    line_str_inputs = [
        ("person", "Authors", authors_str),
        ("device-desktop", "Supported platforms", platforms_str),
    ]

    if language:
        line_str_inputs.append(("code-square", "Language", language))

    line_str_inputs.append(("clock", "Last modified", last_modified))

    # Add version line - either with dropdown selector or plain text
    if version_selector_html:
        # Use the version selector instead of plain text
        line_str_inputs.append(("tag", "Version", version_selector_html))
    elif archive_version:
        line_str_inputs.append(("history", "Archive version", archive_version))
    else:
        line_str_inputs.append(("tag", "Latest version", version))

    line_str_inputs.extend(
        [
            ("stack", "Minimum Holoscan SDK version", min_sdk_version),
            ("beaker", "Tested Holoscan SDK versions", tested_sdk_versions_str),
            ("sparkle-fill", "Contribution metric", ranking_str),
        ]
    )

    # Generate lines strings
    output_lines = []
    for icon, label, value in line_str_inputs:
        if not value:
            logger.warning(f"Skipping metadata line: no value for '{label}'")
            continue

        output_lines.append(f":octicons-{icon}-24: **{label}:** {value}<br>")

    # Join the valid lines and add a line break
    return "".join(output_lines) + "<br>"


def _get_path_relative_to_repo(
    original_path: Path,
    git_repo_path: Path,
    base_dir: Path = None,  # for relative paths
) -> Path | None:
    """Calculates the path relative to the git repo root.

    Handles both absolute (relative to repo root) and relative (relative to base_dir)
    input paths.

    Returns the relative Path object or None if the path is outside the repo.
    """
    if original_path.is_absolute():
        return original_path.relative_to("/")

    if base_dir is None:
        logger.error(f"Path {original_path} is relative but no base directory was provided.")
        return None

    # Resolve relative paths against the base directory
    resolved_absolute = (base_dir / original_path).resolve()
    try:
        # Calculate path relative to repo root
        return resolved_absolute.relative_to(git_repo_path)
    except ValueError:
        logger.error(f"Path {original_path} is not within the git repo {git_repo_path}.")
        return None


def _encode_path_for_url(path_obj: Path) -> str:
    """URL-encode each part of a Path object and join with slashes."""
    return "/".join(urllib.parse.quote(part, safe="") for part in path_obj.parts)


def patch_links(
    text: str,
    relative_dir: Path,  # Relative dir of source/dest (e.g., "tutorials/tut1")
    git_repo_path: Path,  # Absolute path to repo root
    base_url: str,  # Base URL for new external links
) -> str:
    """Patch Markdown links and images in the text.

    - Images: Point to raw GitHub URL.
    - Internal README.md links (within COMPONENT_TYPES): Convert to relative MkDocs paths.
    - Other internal links: Point to blob GitHub URL.
    - External links: Leave untouched.
    """
    readme_dir = (git_repo_path / relative_dir).resolve()
    patched_text = text

    # Regex to find image URLs in both Markdown and HTML
    md_img_regex = re.compile(r"!\[[\s\S]*?\]\((.*?)\)")  # ![alt](url)
    html_img_regex = re.compile(r'<img\s+[^>]*?src=[\'"]([^\'"]*)[\'"]')  # <img src="url"

    for pattern in [md_img_regex, html_img_regex]:
        for match in pattern.finditer(text):
            original_img_str = match.group(1)

            # Skip external URLs
            if original_img_str.startswith(("http://", "https://")):
                continue

            # Get relative path to git repo root
            original_img_path = Path(original_img_str)
            rel_repo_target_path = _get_path_relative_to_repo(
                original_img_path, git_repo_path, readme_dir
            )
            if not rel_repo_target_path:
                continue

            # Create raw GitHub URL
            encoded_path = _encode_path_for_url(rel_repo_target_path)
            new_image_path = f"{base_url}/{encoded_path}?raw=true"
            patched_text = patched_text.replace(original_img_str, new_image_path)

    # Regex to find Markdown links [text](target)
    link_pattern = re.compile(r"\[([\s\S]*?)\]\((.*?)\)")

    for match in link_pattern.finditer(patched_text):  # Use text patched by image loop
        link_text, original_target = match.groups()

        # Skip external and anchor links
        if original_target.strip().startswith(("http://", "https://", "mailto:", "#")):
            continue

        # Separate anchor if present
        original_target_path_str, *anchor_parts = original_target.split("#", 1)
        anchor = f"#{anchor_parts[0]}" if anchor_parts else ""

        # Get relative path to git repo root
        original_target_path = Path(original_target_path_str)
        rel_repo_target_path = _get_path_relative_to_repo(
            original_target_path, git_repo_path, readme_dir
        )
        if not rel_repo_target_path:
            continue

        # Generate new target URL based on type
        # Note: ideally we'd do a first pass of discovery to determine what docs make it in the
        # site instead of this heuristic that depends on logic elsewhere in the codebase, but this
        # is simpler for now.
        is_in_website = (
            rel_repo_target_path.name == "README.md"  # Is a README
            and len(rel_repo_target_path.parts) > 2  # Nested (not root README)
            and rel_repo_target_path.parts[0] in COMPONENT_TYPES  # Top dir is a component type
        )
        if is_in_website:
            # Point relative path within the MkDocs site
            # Note: can't use Path.relative_to() with ../ (ValueError: "not in the subpath")
            new_target = Path(os.path.relpath(rel_repo_target_path, relative_dir)).as_posix()
        else:  # Handle links pointing to GitHub
            # URL encode the path parts first
            encoded_path = _encode_path_for_url(rel_repo_target_path)

            # Construct full URL using the provided base ref URL
            new_target = f"{base_url}/{encoded_path}"

        # Reconstruct the full link
        new_link = f"[{link_text}]({new_target}{anchor})"
        patched_text = patched_text.replace(match.group(0), new_link)

    return patched_text


def extract_markdown_header(md_txt: str) -> tuple[str, str, str] | None:
    """Extract the main header (title) from Markdown text.

    Supports Setext-style (Header\n===) and ATX-style (# Header) headers.
    Finds the first header occurrence in the text.

    Args:
        md_txt: The Markdown content as a string.

    Returns:
        A tuple containing (full_header, header_text, header_symbols)
        if a header is found, otherwise None.
        - full_header: The full matched text of the header.
        - header_text: The text content of the header.
        - header_symbols: The markdown symbols used (e.g., '#', '##', '===', '---').
    """

    # Try ATX first (e.g., # Header, ## Header)
    #   \s*#+\s+ -> group(1) -> '#' symbols, with surrounding whitespaces
    #   .+?      -> group(2) -> text, before end of line $
    atx_match = re.search(r"^(\s*#+\s+)(.+?)$", md_txt, re.MULTILINE)
    if atx_match:
        full_header = atx_match.group(0)
        header_text = atx_match.group(2).strip()
        header_symbols = atx_match.group(1).strip()  # The '#' symbols
        return full_header, header_text, header_symbols

    # If no ATX, try Setext (e.g., Header\n=== or Header\n---)
    #   ^(?! \s*#)  -> don't start line with '#'
    #   .+?         -> group(1) -> text, before new line \n
    #   ={3,}|-{3,} -> group(2) -> underline symbols, before end of line $
    setext_match = re.search(r"^(?! \s*#)(.+?)\n(={3,}|-{3,})\s*$", md_txt, re.MULTILINE)
    if setext_match:
        full_header = setext_match.group(0)
        header_text = setext_match.group(1).strip()
        header_symbols = setext_match.group(2).strip()  # The '===' or '---'
        return full_header, header_text, header_symbols

    return None


def create_version_selector_html(current_version: str, archives: dict, dest_dir: Path, latest_version: str = None) -> tuple[str, str]:
    """Create HTML and JavaScript for version selector dropdown.
    
    Args:
        current_version: The current version being displayed ("latest" or version number)
        archives: Dictionary mapping version names to git references
        dest_dir: Destination directory for the current page (to calculate relative paths)
        latest_version: The version string from metadata.json for the latest version
    
    Returns:
        Tuple of (dropdown_html, script_html) - dropdown for inline use, script for page footer
    """
    # Build the version options
    options = []
    
    # Add "latest" option with version number
    is_latest = current_version == "latest"
    selected_latest = ' selected' if is_latest else ''
    latest_label = f"latest ({latest_version})" if latest_version else "latest"
    options.append(f'<option value="latest"{selected_latest}>{latest_label}</option>')
    
    # Add archived versions (sorted in reverse order)
    for version in sorted(archives.keys(), reverse=True):
        is_selected = current_version == version
        selected_attr = ' selected' if is_selected else ''
        options.append(f'<option value="{version}"{selected_attr}>{version}</option>')
    
    options_html = '\n'.join(options)
    
    dropdown_html = f'<select class="version-selector" id="versionSelector" aria-label="Select version">{options_html}</select>'
    
    script_html = '''
<script>
document.addEventListener('DOMContentLoaded', function() {
    const versionSelector = document.getElementById('versionSelector');
    if (versionSelector) {
        // Get all available versions from the dropdown options
        const availableVersions = Array.from(versionSelector.options)
            .map(opt => opt.value)
            .filter(val => val !== 'latest');
        
        versionSelector.addEventListener('change', function() {
            const selectedValue = this.value;
            let currentPath = window.location.pathname;
            
            // Remove trailing slash for easier processing
            if (currentPath.endsWith('/')) {
                currentPath = currentPath.slice(0, -1);
            }
            
            let pathParts = currentPath.split('/').filter(part => part !== '');
            
            // Check if the last part is a version number
            const lastPart = pathParts[pathParts.length - 1];
            const isOnVersionPage = availableVersions.includes(lastPart);
            
            if (selectedValue === 'latest') {
                // If currently on a version page, remove the version from the path
                if (isOnVersionPage) {
                    pathParts.pop();
                }
            } else {
                // Selecting a specific version
                if (isOnVersionPage) {
                    // Replace the current version with the new one
                    pathParts[pathParts.length - 1] = selectedValue;
                } else {
                    // Append the version to the path
                    pathParts.push(selectedValue);
                }
            }
            
            const newPath = '/' + pathParts.join('/') + '/';
            window.location.href = newPath;
        });
    }
});
</script>
'''
    
    return dropdown_html, script_html


def patch_header(readme_text: str, url: str, metadata_header: str) -> str:
    """Finds the main header in the readme_text, replaces it with a linked
    version, and inserts the metadata_header.

    Args:
        readme_text: The original text of the README.
        url: The URL to link the header title to.
        metadata_header: The formatted metadata block to insert.

    Returns:
        The modified readme_text with the patched header and metadata.
        Returns the original text if no header is found.
    """

    # Extract current header info
    header_info = extract_markdown_header(readme_text)
    if not header_info:
        logger.warning("No markdown header found. Cannot insert metadata header.")
        return readme_text

    full_header, header_text, header_symbols = header_info

    # Create the linked header text
    header_with_url = f"[{header_text}]({url})"

    # Restore the header symbols, force to h1
    if "#" in header_symbols:
        # ATX style: '#' sequence
        new_header = f"# {header_with_url}"
    else:
        # Setext style: '===' or '---'
        new_header = f"{header_with_url}\n==="

    # Append the metadata header
    new_header += f"\n{metadata_header}"

    # Replace the original header
    return readme_text.replace(full_header, new_header, 1)


def create_page(
    metadata: dict,
    readme_text: str,
    dest_path: Path,
    last_modified: str,
    git_repo_path: Path,
    archive: dict = {"version": None, "git_ref": "main"},
    archives: dict = None,
):
    """Create a documentation page, handling both versioned and non-versioned cases.

    Args:
        metadata: The metadata dictionary
        readme_text: Content of the README file
        dest_path: relative path to the documentation page
        last_modified: Last modified date string
        git_repo_path: Path to the Git repository root
        archive: Dictionary of version label and git reference strings
          - if provided, links are versioned accordingly
        archives: Dictionary of all archives (for version selector dropdown)
    Returns:
        Generated page content as string
    """

    # Frontmatter
    archive_version = archive["version"] if archive and "version" in archive else None
    output_text = create_frontmatter(metadata, archive_version)

    # Patch links in the README content
    git_ref = archive["git_ref"] if archive and "git_ref" in archive else "main"
    relative_dir = dest_path.parent
    base_url = f"https://github.com/nvidia-holoscan/holohub/blob/{git_ref}"
    readme_text = patch_links(
        readme_text,
        relative_dir,
        git_repo_path,
        base_url,
    )

    # Generate version selector HTML if archives are present
    version_selector_html = ""
    version_script_html = ""
    if archives:
        current_version = archive_version if archive_version else "latest"
        latest_version = metadata.get("version")
        version_selector_html, version_script_html = create_version_selector_html(current_version, archives, relative_dir, latest_version)

    # Patch the header (finds header, links it, inserts metadata with version selector)
    metadata_header = create_metadata_header(metadata, last_modified, archive_version, version_selector_html)
    encoded_rel_dir = _encode_path_for_url(relative_dir)
    url = f"{base_url}/{encoded_rel_dir}"
    readme_text = patch_header(readme_text, url, metadata_header)

    # Append the text to the output
    output_text += readme_text
    
    # Append the version selector script at the end
    if version_script_html:
        output_text += "\n" + version_script_html

    # Write the mkdocs page
    with mkdocs_gen_files.open(dest_path, "w") as dest_file:
        dest_file.write(output_text)


def create_title_from_readme_title(readme_title: str, suffix: str = "") -> str:
    """Create a cleaned title from the README title by removing common component type suffixes.
    
    Args:
        readme_title: The title extracted from the README
        suffix: Optional suffix to append (e.g., language identifier)
    
    Returns:
        Cleaned title with suffix appended
    """
    title = re.sub(
        r"(Operator|Operators|Op|Application|App|Workflow)\b",
        "",
        readme_title,
        flags=re.IGNORECASE,
    ).strip()
    return title + suffix


def process_archived_versions(
    archives: dict,
    metadata: dict,
    metadata_path: Path,
    readme_path: Path,
    dest_dir: Path,
    project_type: str,
    git_repo_path: Path,
) -> str:
    """Process archived versions of documentation and return nav content.
    
    Args:
        archives: Dictionary mapping version names to git references
        metadata: Current metadata dictionary
        metadata_path: Path to the metadata file
        readme_path: Path to the README file (may be language-specific or parent)
        dest_dir: Destination directory for generated pages
        project_type: Type of project (operator, application, etc.)
        git_repo_path: Path to the Git repository root
    
    Returns:
        Navigation content string for archived versions
    """
    logger.info(f"Processing versioned documentation for {str(dest_dir)}")
    
    nav_content = """
nav:
  - README.md
"""
    
    for version in sorted(archives.keys(), reverse=True):
        git_ref = archives[version]
        
        # Get metadata and README from the specified git reference
        archived_metadata_content = get_file_from_git(metadata_path, git_ref, git_repo_path)
        archived_readme_content = get_file_from_git(readme_path, git_ref, git_repo_path)
        if not archived_metadata_content or not archived_readme_content:
            logger.error(f"Failed to retrieve archived content for {dest_dir} at {git_ref}")
            continue
        
        # Parse the archived metadata
        try:
            archived_metadata = json.loads(archived_metadata_content)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse archived metadata for {dest_dir.name} at {git_ref}")
            continue
        archived_metadata = archived_metadata[project_type]
        
        # Get commit date as last modified
        repo_str = str(git_repo_path)
        cmd = [
            "git",
            "-C",
            repo_str,
            "show",
            "-s",
            "--format=%ad",
            "--date=short",
            git_ref,
        ]
        archive_last_modified = subprocess.run(
            cmd, capture_output=True, text=True, check=True
        ).stdout.strip()
        archive_last_modified = format_date(archive_last_modified)
        
        # Create the archived version content
        archive_dest_path = dest_dir / f"{version}.md"
        create_page(
            archived_metadata,
            archived_readme_content,
            archive_dest_path,
            archive_last_modified,
            git_repo_path,
            archive={"version": version, "git_ref": git_ref},
            archives=archives,
        )
        
        # Add archives to nav file
        nav_content += f'  - "{version}": {version}.md\n'
    
    return nav_content


def write_nav_file(nav_path: Path, nav_content: str) -> None:
    """Write navigation file content to the specified path.
    
    Args:
        nav_path: Path where the navigation file should be written
        nav_content: Content to write to the navigation file
    """
    with mkdocs_gen_files.open(nav_path, "w") as nav_file:
        nav_file.write(nav_content)


def parse_metadata_path(
    metadata_path: Path, components, git_repo_path: Path, processed_parent_readmes=None
) -> None:
    """Copy README file from a sub-package to the user guide's developer guide directory.

    Args:
        metadata_path: Path to the metadata file
        components: Dictionary tracking unique components
        git_repo_path: Path to the Git repository root
        processed_parent_readmes: Set tracking which parent READMEs have been processed

    Returns:
        None
    """
    if processed_parent_readmes is None:
        processed_parent_readmes = set()
    # Disable application with {{ in the name
    if "{{" in str(metadata_path):
        return

    metadata_rel_path = metadata_path.relative_to(git_repo_path)
    logger.info(f"Processing: {metadata_rel_path}")

    # Parse the metadata
    with metadata_path.open("r") as metadata_file:
        metadata = json.load(metadata_file)

    project_type = list(metadata.keys())[0]
    metadata = metadata[project_type]

    # Check valid component type
    component_type = f"{project_type}s"
    if component_type not in COMPONENT_TYPES:
        logger.error(f"Skipping {metadata_rel_path}: unknown type '{component_type}'")
        return

    # Dirs & Paths
    metadata_dir = metadata_path.parent
    readme_dir = metadata_dir
    md_search_dir = readme_dir
    while md_search_dir.name not in COMPONENT_TYPES and md_search_dir != git_repo_path.parent:
        if (md_search_dir / "README.md").exists():
            readme_dir = md_search_dir
            break  # Found the relevant README, stop searching
        md_search_dir = md_search_dir.parent  # Not found, try parent
    readme_path = readme_dir / "README.md"
    dest_dir = readme_dir.relative_to(git_repo_path)
    language_agnostic_dir = metadata_dir
    nbr_language_dirs = 0
    if metadata_dir.name in ["cpp", "python"]:
        language_agnostic_dir = metadata_dir.parent
        nbr_language_dirs = len(list(language_agnostic_dir.glob("*/metadata.json")))

    # Track language agnostic components if in adequate directories
    # Ex:
    # - don't track operators under application folders
    # - only track once for cpp and python
    # - don't track metadata.json in parent directories of multiple operators
    if component_type == metadata_rel_path.parts[0]:
        if "Operators" not in metadata["name"]:
            components[component_type].add(language_agnostic_dir)

    # Process the README content and extract title
    readme_text = ""
    readme_title = metadata["name"] if "name" in metadata else metadata_path.name

    if readme_path.exists():
        with readme_path.open("r") as readme_file:
            readme_text = readme_file.read()

        # Extract title from README content
        header_info = extract_markdown_header(readme_text)
        if header_info:
            readme_title = header_info[1]  # header_text from the tuple
    else:
        logger.warning(f"No README available for {metadata_path}")
        readme_text = f"# {readme_title}\n\nNo documentation found."

    # Prepare suffix with language info if it's a language-specific component
    suffix = ""
    if readme_dir == metadata_dir and nbr_language_dirs > 1:
        suffix = "C++" if metadata_dir.name == "cpp" else f"{metadata_dir.name.capitalize()}"
        suffix = f" ({suffix})"
    logger.debug(f"suffix: {suffix}")

    title = create_title_from_readme_title(readme_title, suffix)

    # Generate page
    dest_path = dest_dir / "README.md"
    last_modified = get_last_modified_date(metadata_path, git_repo_path)
    # Check for archives in metadata for version selector
    archives = metadata["archives"] if "archives" in metadata else None
    create_page(metadata, readme_text, dest_path, last_modified, git_repo_path, archives=archives)

    # Initialize nav file content to set title
    nav_path = dest_dir / ".nav.yml"
    nav_content = f"""
title: "{title}"
"""

    # Check for archives in metadata
    archives = metadata["archives"] if "archives" in metadata else None
    if archives:
        nav_content += process_archived_versions(
            archives, metadata, metadata_path, readme_path, dest_dir, project_type, git_repo_path
        )

    # Write nav file
    write_nav_file(nav_path, nav_content)

    # If we're in a language-specific directory (cpp/python) and processed a language-specific README,
    # also check if there's a parent README to process as a language-agnostic page
    if (
        metadata_dir.name in ["cpp", "python"]
        and readme_dir == metadata_dir
        and nbr_language_dirs > 1
    ):

        parent_readme_path = language_agnostic_dir / "README.md"
        parent_readme_key = str(parent_readme_path.relative_to(git_repo_path))

        # Only process if parent README exists and hasn't been processed yet
        if parent_readme_path.exists() and parent_readme_key not in processed_parent_readmes:
            processed_parent_readmes.add(parent_readme_key)
            logger.info(f"Also processing parent README: {parent_readme_key}")

            # Read parent README
            with parent_readme_path.open("r") as parent_readme_file:
                parent_readme_text = parent_readme_file.read()

            # Extract title from parent README
            parent_readme_title = metadata["name"]
            parent_header_info = extract_markdown_header(parent_readme_text)
            if parent_header_info:
                parent_readme_title = parent_header_info[1]

            # Create title without language suffix (language-agnostic)
            parent_title = create_title_from_readme_title(parent_readme_title)

            # Generate parent page
            parent_dest_dir = language_agnostic_dir.relative_to(git_repo_path)
            parent_dest_path = parent_dest_dir / "README.md"
            parent_last_modified = get_last_modified_date(metadata_path, git_repo_path)
            create_page(
                metadata, parent_readme_text, parent_dest_path, parent_last_modified, git_repo_path, archives=archives
            )

            # Write parent nav file
            parent_nav_path = parent_dest_dir / ".nav.yml"
            parent_nav_content = f"""
title: "{parent_title}"
"""

            # Check for archives in metadata (parent page uses same metadata)
            if archives:
                parent_nav_content += process_archived_versions(
                    archives,
                    metadata,
                    metadata_path,
                    parent_readme_path,
                    parent_dest_dir,
                    project_type,
                    git_repo_path,
                )

            # Write parent nav file
            write_nav_file(parent_nav_path, parent_nav_content)


def generate_pages() -> None:
    """Generate pages for documentation.

    This function orchestrates the entire process of generating API references,
    copying README files for workflow, applications and operators.

    Returns:
        None
    """
    # Initialize Git repository path
    try:
        git_repo_path = get_git_root()
        logger.info(f"Git repository root: {git_repo_path}")
    except Exception as e:
        logger.error(f"Failed to find Git repository root: {e}")
        logger.error("This script requires Git and must be run from within a Git repository.")
        sys.exit(1)

    # Dirs
    src_dir = git_repo_path
    website_src_dir = Path(__file__).parent.parent

    # Initialize map of projects/component per type
    components = {key: set() for key in COMPONENT_TYPES}

    # Track processed parent READMEs to avoid duplicates
    processed_parent_readmes = set()

    for component_type in COMPONENT_TYPES:
        component_dir = src_dir / component_type
        if not component_dir.exists():
            logger.error(f"Component directory not found: {component_dir}")
            continue

        # Parse the metadata.json files
        for metadata_path in component_dir.rglob("metadata.json"):
            try:
                parse_metadata_path(
                    metadata_path, components, git_repo_path, processed_parent_readmes
                )
            except Exception:
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

    # Update all featured component pages with component counts
    featured_pages = [
        "featured-operators.html",
        "featured-applications.html",
        "featured-benchmarks.html",
        "featured-tutorials.html",
    ]

    for featured_page in featured_pages:
        homefile_path = website_src_dir / "overrides" / "_pages" / featured_page
        if homefile_path.exists():
            with homefile_path.open("r+") as home_file:
                home_text = home_file.read()

                # Replace the number of components in the featured page
                for component_type in COMPONENT_TYPES:
                    nbr_components = len(components[component_type])
                    home_text = home_text.replace(f"#{component_type}", str(nbr_components))

                # Write the updated content back to the file
                home_file.seek(0)  # Go to beginning
                home_file.truncate()  # Clear all content
                home_file.write(home_text)
                logger.info(f"Updated component counts in {featured_page}")
        else:
            logger.warning(f"Featured page not found: {featured_page}")

    # Write explicit navigation order for the root
    nav_content = """
nav:
- Home: index.md
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
