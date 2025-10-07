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
    logger,
    parse_metadata_file,
)

def find_most_recent_metadata_files(git_repo_path: Path, component_types: list, count: int = 3) -> list:
    """Find the most recently created/updated metadata.json files.

    Args:
        git_repo_path: Path to the Git repository root
        component_types: List of component types to search for (e.g., ['applications', 'tutorials'])
        count: Number of recent metadata files to retrieve

    Returns:
        List of tuples (metadata_path, datetime) for the most recent unique components
    """
    unique_components = {}
    for component_type in component_types:
        component_dir = git_repo_path / component_type
        if not component_dir.exists():
            continue
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
            commit_date = get_metadata_file_commit_date(metadata_path, git_repo_path)
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


def generate_featured_component_card(metadata_path: Path, git_repo_path: Path) -> str:
    """Generate HTML for a featured component card."""
    metadata, component_type = parse_metadata_file(metadata_path)
    name = metadata.get("name", metadata_path.parent.name)
    description = metadata.get("description", "")
    logger.info(f"Generating featured {component_type} card for {name}")

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
        image_url = f"assets/images/{component_type}_default.png"

    # Generate card HTML with the found image URL
    card_html = f"""
 <div class="col-lg-4 col-sm-12 mb-1 feature-box">
                <a href="{component_path}" class="bg-white shadow padding-feature-box-item text-center d-block match-height" >
                    <img src="{image_url}" alt="{name}" width="120" height="120">
                    <h3 class="mb-1 mt-0">{name}</h3>
                    <p class="feature-card-desc">{description}</p>
                    <p class="nv-teaser-text-link">Learn More <i class="fa-solid fa-angle-right"></i></p>
                </a>
            </div>"""

    return card_html


def generate_featured_content_html(component_types: list, output_file: str, count: int = 3):
    """Generate the featured content HTML for specified component types.
    
    Args:
        component_types: List of component types to feature (e.g., ['applications', 'tutorials'])
        output_file: Path to the output HTML file
        count: Number of components to feature
    """
    git_repo_path = get_git_root()

    recent_metadata_files = find_most_recent_metadata_files(git_repo_path, component_types, count)
    if not recent_metadata_files:
        logger.warning(f"No metadata files found to feature for types {component_types} in {git_repo_path}")
        return
    cards = []
    for metadata_path, _ in recent_metadata_files:
        card_html = generate_featured_component_card(metadata_path, git_repo_path)
        cards.append(card_html)
    cards_html = "".join(cards)
    
    # Generate browse links for each component type
    browse_links = []
    for component_type in component_types:
        browse_links.append(f"""
                <a href="{component_type}" title="Browse all {component_type}" class="md-button md-button--primary" style="padding-top: 0.7rem; padding-bottom: 0.7rem; margin-right: 0.5rem;">
                  Browse all {component_type} (#{component_type})
               </a>""")
    
    browse_links_html = "".join(browse_links)
    featured_content_html = f"""
            {cards_html}
             <div class="tx-hero">
                {browse_links_html}
               </div>
"""
    with open(output_file, "w") as f:
        f.write(featured_content_html)
    logger.info(f"Generated featured content HTML with {len(recent_metadata_files)} components for types {component_types}")


def on_pre_build(config, **kwargs):
    """MkDocs hook for backward compatibility - generates featured applications."""
    generate_featured_content_html(["applications"], "overrides/_pages/featured-applications.html", 3)


def main():
    """Main function that generates featured content for operators, applications, benchmarks, and tutorials."""
    # Define the component types and their corresponding output files
    component_configs = [
        ("operators", "overrides/_pages/featured-operators.html"),
        ("applications", "overrides/_pages/featured-applications.html"),
        ("benchmarks", "overrides/_pages/featured-benchmarks.html"),
        ("tutorials", "overrides/_pages/featured-tutorials.html"),
        ("workflows", "overrides/_pages/featured-workflows.html"),
    ]
    
    # Generate featured content for each component type
    for component_type, output_file in component_configs:
        logger.info(f"Generating featured {component_type} HTML...")
        
        # Validate output file path
        output_path = Path(output_file)
        if not output_path.parent.exists():
            logger.error(f"Output directory does not exist: {output_path.parent}")
            continue
        
        generate_featured_content_html([component_type], output_file, 3)
    
    logger.info("Finished generating all featured content HTML files")


if __name__ in {"__main__", "<run_path>"}:
    main()