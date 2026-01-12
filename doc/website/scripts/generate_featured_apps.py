#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Generate featured content HTML based on the most recent metadata.json files for any component type."""

import os
import sys
from pathlib import Path

script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(script_dir)) if str(script_dir) not in sys.path else None

from common_utils import (  # noqa: E402
    COMPONENT_TYPES,
    extract_first_sentences,
    extract_image_from_readme,
    get_full_image_url,
    get_git_root,
    get_metadata_file_commit_date,
    get_recent_source_code_update_date,
    logger,
    parse_metadata_file,
)


def find_most_recent_metadata_files(
    git_repo_path: Path, component_type: str, count: int = 3
) -> list:
    """Find the most recently created/updated metadata.json files.

    Args:
        git_repo_path: Path to the Git repository root
        component_type: Component type to search for (e.g., 'applications', 'tutorials')
        count: Number of recent metadata files to retrieve

    Returns:
        List of tuples (metadata_path, datetime) for the most recent unique components
    """
    unique_components = {}
    component_dir = git_repo_path / component_type

    if not component_dir.exists():
        return []

    for metadata_path in component_dir.rglob("metadata.json"):
        # Skip certain non-regular components based on component type
        if component_type == "applications" and any(
            t in str(metadata_path) for t in ["datawriter", "operators", "xr_hello_holoscan"]
        ):
            continue  # non regular apps skip.

        component_dir_path = metadata_path.parent
        if component_dir_path.name in ["cpp", "python"]:
            component_dir_path = component_dir_path.parent

        component_name = component_dir_path.name
        # Get the creation date of the metadata.json file from git history
        # This uses 'git log --follow --format=%at --reverse' to find the first commit
        commit_date = get_metadata_file_commit_date(metadata_path, git_repo_path)
        logger.debug(
            f"Component '{component_name}' was created on {commit_date.strftime('%Y-%m-%d')}"
        )

        if (
            component_name not in unique_components
            or commit_date > unique_components[component_name][1]
        ):
            unique_components[component_name] = (metadata_path, commit_date)

    result = sorted(unique_components.values(), key=lambda x: x[1], reverse=True)
    return result[:count]


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


def generate_featured_component_card(
    metadata_path: Path, git_repo_path: Path, commit_date=None
) -> str:
    """Generate HTML for a featured component card."""
    metadata, component_type = parse_metadata_file(metadata_path)
    name = metadata.get("name", metadata_path.parent.name)
    description = metadata.get("description", "")
    tags = metadata.get("tags", [])

    # Detect all available languages by checking for metadata.json files
    available_languages = []
    current_lang = metadata.get("language", "")

    # Check if this is in a language-specific subdirectory (cpp/python)
    metadata_dir = metadata_path.parent
    parent_dir = metadata_dir.parent if metadata_dir.name in ["cpp", "python"] else metadata_dir

    # Check for Python version
    python_metadata = parent_dir / "python" / "metadata.json"
    if python_metadata.exists():
        available_languages.append("Python")

    # Check for C++ version
    cpp_metadata = parent_dir / "cpp" / "metadata.json"
    if cpp_metadata.exists():
        available_languages.append("C++")

    # If no language-specific subdirectories found, use the current language from metadata
    if not available_languages and current_lang:
        if isinstance(current_lang, list):
            available_languages = current_lang
        else:
            available_languages = [current_lang]

    # Create language badge HTML if languages are available
    language_badge_html = ""
    if available_languages:
        languages_str = ", ".join(available_languages)
        language_badge_html = f'<span style="display: inline-block; background-color: var(--md-code-bg-color); color: var(--md-code-fg-color); padding: 0.2rem 0.5rem; border-radius: 0.25rem; font-size: 0.65rem; font-weight: 600; margin-bottom: 0.5rem;">{languages_str}</span>'

    logger.info(
        f"Generating featured {component_type} card for {name} with languages: {', '.join(available_languages) if available_languages else 'None'}"
    )

    readme_path = find_readme_path(metadata_path.parent, git_repo_path)
    readme_content = ""
    image_url = None

    # If no description in metadata, look for first paragraph in README
    if readme_path and readme_path.exists():
        with readme_path.open("r") as f:
            readme_text = f.read()
            readme_content = readme_text
            if not description:
                description = extract_first_sentences(readme_text, 1, max_chars=120)
    if readme_content:
        image_path = extract_image_from_readme(readme_content)
        if image_path:
            image_url = get_full_image_url(image_path, readme_path)
            logger.info(f"Found image in README for {name}: {image_url}")
    component_path = get_component_path(metadata_path, git_repo_path)

    # Use the found image URL or fall back to default
    if not image_url:
        logger.info(f"No image found in README for {name}, using default")
        image_url = f"/holohub/assets/images/{component_type}_default.png"

    # Check if this is a recent contribution (within 45 days)
    # commit_date is the date when metadata.json was first committed to git
    from datetime import datetime

    is_recent_attr = ""
    badge_html = ""
    is_new = False

    if commit_date:
        days_old = (datetime.now() - commit_date).days
        if days_old <= 45:
            is_recent_attr = ' data-recent="true"'
            badge_html = '<span class="new-badge" style="position: absolute; top: 10px; right: 10px; background-color: #76b900; color: white; padding: 0.2rem 0.5rem; border-radius: 0.25rem; font-size: 0.65rem; font-weight: 600;">New</span>'
            is_new = True
            logger.info(
                f"✓ Marking '{name}' as NEW - created {days_old} days ago ({commit_date.strftime('%Y-%m-%d')})"
            )
        else:
            logger.debug(
                f"  '{name}' is {days_old} days old (created {commit_date.strftime('%Y-%m-%d')})"
            )

    # Check for recent source code updates (within 30 days) - only if not already marked as "New"
    if not is_new:
        update_date = get_recent_source_code_update_date(metadata_path, git_repo_path)
        if update_date:
            days_since_update = (datetime.now() - update_date).days
            if days_since_update <= 30:
                is_recent_attr = ' data-updated="true"'
                badge_html = '<span class="updated-badge" style="position: absolute; top: 10px; right: 10px; background-color: #ff9933; color: white; padding: 0.2rem 0.5rem; border-radius: 0.25rem; font-size: 0.65rem; font-weight: 600;">Updated</span>'
                logger.info(
                    f"✓ Marking '{name}' as UPDATED - source code modified {days_since_update} days ago ({update_date.strftime('%Y-%m-%d')})"
                )
            else:
                logger.debug(
                    f"  '{name}' last updated {days_since_update} days ago ({update_date.strftime('%Y-%m-%d')})"
                )

    # Generate tags HTML (hide first tag visually but keep it for filtering)
    tags_html = ""
    if tags:
        # First tag (category) - hidden but kept for filtering, all lowercase with spaces replaced by hyphens
        tag_items = []
        first_tag_display = tags[0].lower()
        first_tag_href = tags[0].lower().replace(" ", "-")
        tag_items.append(
            f'<a href="/holohub/tags/#tag:{first_tag_href}" class="md-tag" style="display: none;">{first_tag_display}</a>'
        )

        # Remaining tags - visible, all lowercase with spaces replaced by hyphens
        for tag in tags[1:]:
            tag_display = tag.lower()
            tag_href = tag.lower().replace(" ", "-")
            tag_items.append(
                f'<a href="/holohub/tags/#tag:{tag_href}" class="md-tag" style="display: inline-block; margin: 0.1rem; font-size: 0.55rem; padding: 0.05rem 0.25rem; cursor: pointer; transition: background-color 0.2s; text-decoration: none; line-height: normal; height: auto; width: auto;" onmouseover="this.style.backgroundColor=\'#5a9100\';" onmouseout="this.style.backgroundColor=\'\';" onclick="event.stopPropagation();">{tag_display}</a>'
            )

        tags_html = f'<div class="md-tags md-nav__link md-typeset" style="line-height: 1.5; display: block;" onclick="event.stopPropagation();">{" ".join(tag_items)}</div>'

    # Generate card HTML with the found image URL
    card_html = f"""
 <div class="col-xl-4 col-lg-6 col-sm-12 mb-1 feature-box"{is_recent_attr}>
                <div class="shadow padding-feature-box-item text-center d-block match-height app-card" style="cursor: pointer; position: relative;" onclick="window.location.href='/holohub/{component_path}';">
                    {badge_html}
                    <img src="{image_url}" alt="{name}" width="120" height="120">
                    <h3 class="mb-1 mt-0" style="font-size: 0.8rem;">{name}</h3>
                    {language_badge_html}
                    <p class="feature-card-desc">{description}</p>
                    {tags_html}
                    <!-- <p class="nv-teaser-text-link">Learn More <i class="fa-solid fa-angle-right"></i></p> -->
                </div>
            </div>"""

    return card_html


def generate_featured_content_html(component_type: str, output_path: str, count: int = 3):
    """Generate the featured content HTML for specified component_type.

    Args:
        component_type: Component type to feature (e.g., 'applications', 'tutorials')
        output_path: Path to the output HTML file
        count: Number of components to feature
    """
    git_repo_path = get_git_root()
    output_file = Path(output_path) / f"featured-{component_type}.html"
    recent_metadata_files = find_most_recent_metadata_files(git_repo_path, component_type, count)
    if not recent_metadata_files:
        logger.warning(
            f"No metadata files found to feature for {component_type} in {git_repo_path}"
        )
        return
    cards = []
    for metadata_path, commit_date in recent_metadata_files:
        # Skip if '/template/' is in the directory path
        if "/template/" in str(metadata_path).replace("\\", "/") + "/":
            continue
        card_html = generate_featured_component_card(metadata_path, git_repo_path, commit_date)
        cards.append(card_html)
    cards_html = "".join(cards)

    # Generate browse links for each component type
    browse_links = []
    browse_links.append(
        f"""
                <a href="{component_type}" title="Browse all {component_type}" class="md-button md-button--primary" style="padding-top: 0.7rem; padding-bottom: 0.7rem; margin-right: 0.5rem;">
                  Browse all {component_type} (#{component_type})
               </a>"""
    )

    browse_links_html = "".join(browse_links)
    featured_content_html = f"""
            {cards_html}
             <div class="tx-hero">
                {browse_links_html}
               </div>
"""
    with open(output_file, "w") as f:
        f.write(featured_content_html)
    logger.info(
        f"Generated featured content HTML with {len(recent_metadata_files)} components for {component_type}"
    )


def get_unique_first_tags(git_repo_path: Path, component_type: str) -> dict:
    """Get unique first tags from all metadata.json files for a component type with counts.

    This function counts how many applications have each tag (as first tag for categories,
    but counts all occurrences of each tag to match the filtering behavior).
    Applications with both cpp and python implementations are counted only once.

    Args:
        git_repo_path: Path to the Git repository root
        component_type: Component type to search (e.g., 'applications', 'tutorials')

    Returns:
        Dictionary mapping first tag names to their counts (counting all apps with that tag anywhere)
    """
    # First, collect all first tags (categories)
    first_tags = set()
    component_dir = git_repo_path / component_type

    if not component_dir.exists():
        return {}

    # Map plural component_type to singular key in metadata
    component_key_map = {
        "applications": "application",
        "operators": "operator",
        "tutorials": "tutorial",
        "benchmarks": "benchmark",
        "workflows": "workflow",
    }
    expected_key = component_key_map.get(component_type, component_type.rstrip("s"))

    # First pass: collect all first tags (these are the categories)
    for metadata_path in component_dir.rglob("metadata.json"):
        # Skip template files
        if "template" in str(metadata_path):
            continue

        try:
            metadata, parsed_type = parse_metadata_file(metadata_path)

            # Skip if this metadata doesn't match the expected component type
            if parsed_type != expected_key:
                continue

            tags = metadata.get("tags", [])
            if tags and len(tags) > 0:
                first_tags.add(tags[0])
        except Exception as e:
            logger.warning(f"Error reading {metadata_path}: {e}")

    # Helper function to normalize app names for deduplication
    def normalize_app_name(name: str) -> str:
        """Normalize app name by removing common variations."""
        # Convert to lowercase, remove extra spaces, and remove common words that might differ
        normalized = name.lower().strip()
        # Remove " and " to handle cases like "Tool and AR" vs "Tool AR"
        normalized = normalized.replace(" and ", " ")
        # Remove multiple spaces
        normalized = " ".join(normalized.split())
        return normalized

    # Second pass: for each category (first tag), count unique apps (by normalized name) that have that tag anywhere
    tag_counts = {}
    for category in first_tags:
        # Use a set to track unique app names for this category
        unique_app_names = set()

        for metadata_path in component_dir.rglob("metadata.json"):
            # Skip template files
            if "template" in str(metadata_path):
                continue

            try:
                metadata, parsed_type = parse_metadata_file(metadata_path)

                # Skip if this metadata doesn't match the expected component type
                if parsed_type != expected_key:
                    continue

                tags = metadata.get("tags", [])
                # Check if the category appears anywhere in the tags
                if category in tags:
                    app_name = metadata.get("name", "")
                    if app_name:
                        # Use normalized name for deduplication
                        normalized_name = normalize_app_name(app_name)
                        unique_app_names.add(normalized_name)
            except Exception:
                pass

        tag_counts[category] = len(unique_app_names)

    # Return sorted by tag name
    return dict(sorted(tag_counts.items()))


def generate_component_html(component_type: str, output_path: str):
    """Generate the featured content HTML for specified component types.

    Args:
        component_type: Component type to feature (e.g., 'applications', 'tutorials')
        output_path: Path to the output HTML file
        count: Number of components to feature
    """
    git_repo_path = get_git_root()
    output_file = Path(output_path) / f"{component_type}.html"
    recent_metadata_files = find_most_recent_metadata_files(git_repo_path, component_type, 500)
    if not recent_metadata_files:
        logger.warning(
            f"No metadata files found to feature for {component_type} in {git_repo_path}"
        )
        return

    # Map plural component_type to singular key in metadata
    component_key_map = {
        "applications": "application",
        "operators": "operator",
        "tutorials": "tutorial",
        "benchmarks": "benchmark",
        "workflows": "workflow",
    }
    expected_key = component_key_map.get(component_type, component_type.rstrip("s"))

    # Helper function to normalize app names for deduplication
    def normalize_app_name(name: str) -> str:
        """Normalize app name by removing common variations."""
        # Convert to lowercase, remove extra spaces, and remove common words that might differ
        normalized = name.lower().strip()
        # Remove " and " to handle cases like "Tool and AR" vs "Tool AR"
        normalized = normalized.replace(" and ", " ")
        # Remove multiple spaces
        normalized = " ".join(normalized.split())
        return normalized

    cards = []
    unique_app_names = set()  # Track unique app names for total count

    for metadata_path, commit_date in recent_metadata_files:
        # Skip if '/template/' is in the directory path
        if "/template/" in str(metadata_path).replace("\\", "/") + "/":
            continue

        # Verify this metadata matches the expected component type
        try:
            metadata, parsed_type = parse_metadata_file(metadata_path)
            if parsed_type != expected_key:
                logger.debug(
                    f"Skipping {metadata_path}: expected {expected_key}, got {parsed_type}"
                )
                continue

            # Track unique app name for counting (normalized to avoid counting cpp/python separately)
            app_name = metadata.get("name", "")
            if app_name:
                normalized_name = normalize_app_name(app_name)
                unique_app_names.add(normalized_name)
        except Exception as e:
            logger.warning(f"Error parsing {metadata_path}: {e}")
            continue

        card_html = generate_featured_component_card(metadata_path, git_repo_path, commit_date)
        cards.append(card_html)
    cards_html = "".join(cards)

    # Generate browse links for each component type
    content_html = f"""
            {cards_html}
"""
    with open(output_file, "w") as f:
        f.write(content_html)
    logger.info(
        f"Generated featured content HTML with {len(cards)} components for {component_type}"
    )

    # Generate navigation HTML based on unique first tags
    tag_counts = get_unique_first_tags(git_repo_path, component_type)
    if tag_counts:
        nav_output_file = Path(output_path) / f"{component_type}_nav.html"
        # Total count is the number of unique application names (not counting cpp/python separately)
        total_count = len(unique_app_names)
        generate_navigation_html(tag_counts, component_type, nav_output_file, total_count)


def generate_navigation_html(
    tag_counts: dict, component_type: str, output_file: Path, total_count: int
):
    """Generate navigation HTML for component categories.

    Args:
        tag_counts: Dictionary mapping tag names to their counts
        component_type: Component type (e.g., 'applications')
        output_file: Path to output navigation HTML file
        total_count: Total number of components
    """
    nav_items = [
        f'<a href="#all" style="display: block; padding: 0.75rem 1rem; text-decoration: none; color: var(--md-default-fg-color); font-size: 0.7rem; transition: all 0.2s;" onclick="filterByTag(\'all\'); return true;">All ({total_count})</a>'
    ]

    # Map long tag names to shorter display names
    tag_display_map = {
        "Computer Vision and Perception": "Computer Vision",
        "Natural Language and Conversational AI": "NLP & Conversational AI",
        "Networking and Distributed Computing": "Networking",
        "Tools And Other Specialized Applications": "Tools & Specialized",
    }

    for tag, count in tag_counts.items():
        tag_lower = tag.lower()
        tag_href = tag_lower.replace(" ", "-")
        display_name = tag_display_map.get(tag, tag)

        nav_item = f'<a href="#{tag_href}" style="display: block; padding: 0.75rem 1rem; text-decoration: none; color: var(--md-default-fg-color); font-size: 0.7rem; transition: all 0.2s;" onclick="filterByTag(\'{tag}\'); return true;">{display_name} ({count})</a>'
        nav_items.append(nav_item)

    nav_html = "\n                        ".join(nav_items)

    with open(output_file, "w") as f:
        f.write(nav_html)

    logger.info(f"Generated navigation HTML with {len(tag_counts)} categories for {component_type}")


def main():
    """Main function that generates featured content for operators, applications, benchmarks, and tutorials."""
    # Define the component types and their corresponding output files
    component_configs = [
        "operators",
        "applications",
        "benchmarks",
        "tutorials",
        "workflows",
    ]

    # Validate output file path
    output_path = "overrides/_pages"
    if not Path(output_path).parent.exists():
        logger.error(f"Output directory does not exist: {output_path.parent}")
        return

    # Generate featured content for each component type
    for component_type in component_configs:
        logger.info(f"Generating featured {component_type} HTML...")

        generate_featured_content_html(component_type, output_path, 3)
        generate_component_html(component_type, output_path)

    logger.info("Finished generating all featured content HTML files")


if __name__ in {"__main__", "<run_path>"}:
    main()
