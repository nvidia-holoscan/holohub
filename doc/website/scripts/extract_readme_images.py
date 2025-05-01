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
"""
Script to extract images from component README.md files and update the featured-apps.html
thumbnails during the mkdocs build process.
"""

import os
import sys
from bs4 import BeautifulSoup
from pathlib import Path

# Add the script directory to the path to enable importing common_utils
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from common_utils import (
    logger,
    COMPONENT_TYPES,
    HOLOHUB_REPO_URL,
    extract_image_from_readme,
    get_git_root,
)

# Base directories and URLs
SCRIPT_DIR = Path(__file__).parent
WEBSITE_DIR = SCRIPT_DIR.parent
FEATURED_APPS_PATH = WEBSITE_DIR / "docs" / "_data" / "featured-apps.html"

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
        if href and '/' in href:
            href_path = href.strip('/')
            readme_paths.append(git_repo_path / href_path / "README.md")
            href_parent = Path(href_path).parent
            if str(href_parent) != '.':  # Only if parent is not root
                readme_paths.append(git_repo_path / href_parent / "README.md")

        readme_paths.extend([
            git_repo_path / component_type / component_name / "README.md",
            git_repo_path / component_type / component_name / "python" / "README.md",
            git_repo_path / component_type / component_name / "cpp" / "README.md",
        ])

        # Try each possible path
        for path in readme_paths:
            if path.exists():
                logger.info(f"Found README at {path}")
                with open(path, 'r', encoding='utf-8') as f:
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
    if relative_path.startswith('./'):
        relative_path = relative_path[2:]

    # Handle already absolute URLs
    if relative_path.startswith(('http://', 'https://')):
        return relative_path

    # Handle already raw GitHub URLs
    if relative_path.startswith('https://raw.githubusercontent.com/'):
        return relative_path

    # Convert GitHub blob URLs to raw URLs
    if relative_path.startswith(HOLOHUB_REPO_URL) and 'blob/' in relative_path:
        # Convert from:
        # https://github.com/nvidia-holoscan/holohub/blob/main/path/to/image.png
        # To:
        # https://raw.githubusercontent.com/nvidia-holoscan/holohub/main/path/to/image.png
        return relative_path.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')

    # Use the README path to resolve the relative path
    if readme_path:
        # Get the directory containing the README
        readme_dir = readme_path.parent

        # Resolve the image path relative to the README directory
        image_path = (readme_dir / relative_path).resolve()

        # Get the path relative to the git root
        git_root = get_git_root()
        try:
            rel_image_path = image_path.relative_to(git_root)
            # Use raw.githubusercontent.com instead of github.com/blob
            return f"https://raw.githubusercontent.com/nvidia-holoscan/holohub/main/{rel_image_path}"
        except ValueError:
            # If we can't make it relative to git root, use absolute path
            logger.warning(f"Could not make {image_path} relative to git root")

    # Fallback to the old method if we can't determine from README path
    components = readme_path.relative_to(get_git_root()).parts if readme_path else []
    if len(components) >= 2:
        component_type, component_name = components[0], components[1]
        return f"https://raw.githubusercontent.com/nvidia-holoscan/holohub/main/{component_type}/{component_name}/{relative_path}"

    # Last resort fallback
    logger.warning(f"Using direct URL without context: {relative_path}")
    return relative_path

def update_featured_apps_html():
    """Updates the featured-apps.html with images from READMEs."""
    if not FEATURED_APPS_PATH.exists():
        logger.error(f"Error: Featured apps file not found at {FEATURED_APPS_PATH}")
        return

    # Parse the featured-apps.html file
    with open(FEATURED_APPS_PATH, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    app_cards = soup.select('.featured-app-card')

    updated = False

    # Default image URLs for each component type
    default_images = {
        "applications": "assets/images/applications_default.png",
    }

    # Generic fallback image if component type is unknown
    generic_default = "assets/images/applications_default.png"

    for card in app_cards:
        # Extract component type and name from href
        href = card.get('href', '')
        if not href or '/' not in href:
            continue

        parts = href.split('/')
        if len(parts) < 2:
            continue

        component_type = parts[0]
        component_name = parts[-1]

        # Check if this is a valid component type
        if component_type not in COMPONENT_TYPES:
            continue

        # Get image from README, passing the href for more precise location
        readme_content, readme_path = get_readme_content(component_type, component_name, href)
        image_path = extract_image_from_readme(readme_content)

        # Get the image tag
        img_tag = card.select_one('.app-thumbnail img')
        if not img_tag:
            continue

        if image_path:
            # README image found - use it
            full_img_url = get_full_image_url(image_path, readme_path)
            logger.info(f"Found image in README for {component_type}/{component_name}: {full_img_url}")

            # Update the image src and data-src
            img_tag['src'] = full_img_url
            img_tag['data-src'] = full_img_url

            # Remove unnecessary attributes
            if 'data-app-name' in img_tag.attrs:
                del img_tag.attrs['data-app-name']

            updated = True
        else:
            # No README image found - use a default image based on component type
            logger.info(f"No image found in README for {component_type}/{component_name}, using default")

            # Select the appropriate default image
            default_img_url = default_images.get(component_type, generic_default)

            # Update the image src and data-src
            img_tag['src'] = default_img_url
            img_tag['data-src'] = default_img_url

            updated = True

    if updated:
        # Write back the updated HTML
        with open(FEATURED_APPS_PATH, 'w', encoding='utf-8') as f:
            f.write(str(soup))
        logger.info("Updated featured-apps.html with README images")
    else:
        logger.info("No changes made to featured-apps.html")

if __name__ in {"__main__", "<run_path>"}:
    update_featured_apps_html()
